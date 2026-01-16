#include <chrono>
#include <iostream>
#include <ctime>
#include <filesystem>
#include <unistd.h>
#include <algorithm>

#include "clipp.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/env.h"
#include "rocksdb/iostats_context.h"
#include "rocksdb/perf_context.h"
#include "rocksdb/compaction_filter.h"
#include "rocksdb/utilities/sim_cache.h"

// 自定义组件
#include "tmpdb/compactor.hpp"
#include "infrastructure/data_generator.hpp"
#include "tmpdb/progress_bar.hpp"

using namespace ROCKSDB_NAMESPACE;

#define PAGESIZE 4096

struct ExpResult
{
    // 实验参数配置
    size_t M;           // 总内存
    size_t N;           // 数据量
    size_t Q;           // 查询数
    double T;           // size ratio
    double skewness;    // Zipfian参数

    // 两个读写工作负载(如果相同，说明当前实验是单一工作负载)
    double read_ratio_1;  // 读比例
    double write_ratio_1; // 写比例
    double read_ratio_2;  // 读比例
    double write_ratio_2; // 写比例


    // 整体性能
    double H_cache;             // 总命中率 (混合读写测量)
    double write_io_kb_per_op;  // I/O 成本 (KB/op)
    double read_io_kb_per_op;
    double total_io_kb_per_op;
    double latency;  // 混合读写工作负载时间 (毫秒)

    std::string to_csv() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6)
            << (M / (1024.0 * 1024.0)) << ","
            << N << ","
            << Q << ","
            << T << ","
            << skewness << "," 
            << read_ratio_1 << "," << write_ratio_1 << ","
            << read_ratio_2 << "," << write_ratio_2 << ","
            << H_cache << ","
            << write_io_kb_per_op << ","
            << read_io_kb_per_op << ","
            << total_io_kb_per_op << ","
            << latency;
        return oss.str();
    }
};

typedef struct environment
{
    std::string db_path;

    // 工作负载配置
    double read_ratio_1 = 0.5;
    double write_ratio_1 = 0.5; 

    // 工作负载2的配置
    double read_ratio_2 = 0.5;
    double write_ratio_2 = 0.5; 

    size_t queries = 10;
    int sel = 2;
    int scaling = 1;
    std::string compaction_style = "level";
    double T = 10;
    double K = 0;
    size_t E = 1 << 7;
    double bpe = 5.0;
    size_t N = 1e6;
    size_t L = 0;
    size_t M = 0; // 总内存预算
     
    std::string exp_output_file = "/data/default/exp_result.txt";  // 实验输出文件

    // 其他配置
    int verbose = 0;
    bool destroy_db = true;
    int max_rocksdb_levels = 64;
    int parallelism = 1;
    int seed = 0;
    std::string dist_mode = "zipfian";
    double skew = 0.5;

} environment;

environment parse_args(int argc, char *argv[])
{
    using namespace clipp;
    using std::to_string;

    environment env;
    bool help = false;

    auto general_opt = "general options" % (
        (option("-v", "--verbose") & integer("level", env.verbose)) % "Logging levels",
        (option("-h", "--help").set(help, true)) % "prints this message"
    );

    auto build_opt = "build options:" % (
        (value("db_path", env.db_path)) % "path to the db",
        (option("-N", "--entries") & integer("num", env.N)) % "total entries",
        (option("-T", "--size-ratio") & number("ratio", env.T)) % "size ratio",
        (option("-M", "--total-memory-size") & integer("size", env.M)) % "total memory size",
        (option("-E", "--entry-size") & integer("size", env.E)) % "entry size",
        (option("-b", "--bpe") & number("bits", env.bpe)) % "bits per element",
        (option("-c", "--compaction") & value("mode", env.compaction_style)) % "compaction style",
        (option("-d", "--destroy").set(env.destroy_db)) % "destroy the DB if exists"
    );

    auto run_opt = "run options:" % (
        (option("-r1", "--reads1") & number("num", env.read_ratio_1)) % "Workload1 read ratio",
        (option("-w1", "--writes1") & number("num", env.write_ratio_1)) % "Workload1 write ratio",
        (option("-r2", "--reads2") & number("num", env.read_ratio_2)) % "Workload2 read ratio",
        (option("-w2", "--writes2") & number("num", env.write_ratio_2)) % "Workload2 write ratio",
        (option("-s", "--queries") & integer("num", env.queries)) % "queries",
        (option("--dist") & value("mode", env.dist_mode)) % "distribution mode",
        (option("--skew") & number("num", env.skew)) % "skewness for zipfian",
        (option("--sel") & number("num", env.sel)) % "selectivity of range query",
        (option("-o", "--output") & value("file", env.exp_output_file)) % "output CSV file"
    );

    auto minor_opt = "minor options:" % (
        (option("--max_rocksdb_level") & integer("num", env.max_rocksdb_levels)) % "max levels",
        (option("--parallelism") & integer("num", env.parallelism)) % "parallelism",
        (option("--seed") & integer("num", env.seed)) % "seed for generating data"
    );

    auto cli = (general_opt, build_opt, run_opt, minor_opt);

    if (!parse(argc, argv, cli))
        help = true;

    if (help)
    {
        auto fmt = doc_formatting{}.doc_column(42);
        std::cout << make_man_page(cli, "db_runner_with_tuner", fmt);
        exit(EXIT_FAILURE);
    }

    return env;
}

void print_db_status(rocksdb::DB *db)
{
    spdlog::debug("Files per level");
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    int level_idx = 1;
    for (auto &level : cf_meta.levels)
    {
        std::string level_str = "";
        for (auto &file : level.files)
        {
            level_str += file.name + ", ";
        }
        level_str = level_str == "" ? "EMPTY" : level_str.substr(0, level_str.size() - 2);
        spdlog::debug("Level {} : {} Files : {}", level_idx, level.files.size(), level_str);
        level_idx++;
    }
}

void wait_for_compactions(rocksdb::DB *db, tmpdb::Compactor *compactor)
{
    uint64_t num_running_flushes, num_pending_flushes;
    
    while (true)
    {
        db->GetIntProperty(DB::Properties::kNumRunningFlushes, &num_running_flushes);
        db->GetIntProperty(DB::Properties::kMemTableFlushPending, &num_pending_flushes);
        if (num_running_flushes == 0 && num_pending_flushes == 0)
            break;
    }
    
    while (compactor->compactions_left_count > 0)
        ;
    
    while (compactor->requires_compaction(db))
    {
        while (compactor->compactions_left_count > 0)
            ;
    }
}

void reset_all_statistics(rocksdb::Options &rocksdb_opt, 
                          tmpdb::Compactor *compactor)
{
    rocksdb_opt.statistics->Reset();
    rocksdb::get_iostats_context()->Reset();
    rocksdb::get_perf_context()->Reset();
    compactor->stats.reset_epoch();
}

int run_experiment(environment &env)
{
    // ==================== 设置已知Result ====================
    ExpResult result;
    result.M = env.M;
    result.N = env.N;
    result.Q = env.queries;
    result.T = env.T;
    result.skewness = env.skew;
    result.read_ratio_1 = env.read_ratio_1;
    result.write_ratio_1 = env.write_ratio_1;
    result.read_ratio_2 = env.read_ratio_2;
    result.write_ratio_2 = env.write_ratio_2;

    // ==================== 配置 RocksDB ====================
    std::string db_path = env.db_path + "_memory_tuner";
    // 销毁旧数据库
    rocksdb::DestroyDB(db_path, rocksdb::Options());
    std::string rm_db_cmd = "rm -rf " + db_path;
    int ret = system(rm_db_cmd.c_str());
    if (ret != 0) {
        spdlog::warn("Failed to execute: {}, return code: {}", rm_db_cmd, ret);
    }

    spdlog::info("Building DB: {}", db_path);
    rocksdb::Options rocksdb_opt;

    rocksdb_opt.create_if_missing = true;
    rocksdb_opt.error_if_exists = true;
    rocksdb_opt.IncreaseParallelism(env.parallelism);
    rocksdb_opt.compression = rocksdb::kNoCompression;
    rocksdb_opt.bottommost_compression = kNoCompression;
    rocksdb_opt.use_direct_reads = true;
    rocksdb_opt.use_direct_io_for_flush_and_compaction = true;
    rocksdb_opt.compaction_style = rocksdb::kCompactionStyleNone;
    rocksdb_opt.disable_auto_compactions = true;
    rocksdb_opt.max_total_wal_size = 128 * 1024 * 1024; // 128MB
    rocksdb_opt.max_bytes_for_level_multiplier = 5;  // 增加Compaction的数量

    // 写日志长度: rocksdb_opt.max_log_file_size

    // 设置初始write buffer大小
    size_t Mb = 64 * 1024 * 1024;  // 使用默认参数 64MB
    size_t Mc = env.M - Mb;  // 剩余内存全部分配给 Block Cache
    rocksdb_opt.write_buffer_size = Mb; // ✅ 实际配置写内存参数
    spdlog::info("Write buffer size: {} MB", Mb / (1024 * 1024));

    // ==================== 配置自定义 Compactor ====================
    tmpdb::Compactor *compactor = nullptr;
    tmpdb::CompactorOptions compactor_opt;
    
    compactor_opt.size_ratio = env.T;
    compactor_opt.buffer_size = Mb;
    compactor_opt.entry_size = env.E;
    compactor_opt.bits_per_element = env.bpe;
    compactor_opt.num_entries = env.N;

    if (env.compaction_style == "level")
        compactor_opt.K = 1;
    else if (env.compaction_style == "tier")
        compactor_opt.K = env.T;
    else
        compactor_opt.K = env.K;

    compactor_opt.levels = tmpdb::Compactor::estimate_levels(env.N, env.T, env.E, Mb) 
                           * compactor_opt.K + 1;
    rocksdb_opt.num_levels = compactor_opt.levels + 1;

    compactor = new tmpdb::Compactor(compactor_opt, rocksdb_opt);
    rocksdb_opt.listeners.emplace_back(compactor);

    // ==================== 配置 filter + Block Cache ====================
    rocksdb::BlockBasedTableOptions table_options;
    table_options.read_amp_bytes_per_bit = 32;

    table_options.filter_policy.reset(
        rocksdb::NewMonkeyFilterPolicy(
            env.bpe,
            compactor_opt.size_ratio,
            compactor_opt.levels));
    
    std::shared_ptr<Cache> block_cache = rocksdb::NewLRUCache(Mc);  
    table_options.block_cache = block_cache;     
    
    rocksdb_opt.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    rocksdb_opt.statistics = rocksdb::CreateDBStatistics();

    // ==================== 打开数据库 ====================
    rocksdb::DB *db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(rocksdb_opt, db_path, &db);
    if (!status.ok())
    {
        spdlog::error("Problems opening DB: {}", status.ToString());
        delete db;
        exit(EXIT_FAILURE);
    }

    // ==================== 初始化 + 预热 + 实际混合读写 ====================
    spdlog::info("Initializing data with {} entries...", env.N);

    ReadOptions read_options;
    read_options.total_order_seek = true;

    rocksdb::WriteOptions write_opt;
    write_opt.low_pri = true;
    write_opt.disableWAL = false; // ✅ 必须要开启写日志，因为其作为写成本导数系数出现

    std::string value, key, limit;
    std::pair<std::string, std::string> key_value;

    // 1️⃣ 初始化LSM-tree阶段：注入N个entry
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Initializing LSM-tree with {} entries", env.N);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    DataGenerator *data_gen = new YCSBGenerator(env.N, "uniform", 0.0);
    {
        ProgressBar progress(env.N, "📥 Init Data  ");
        for (size_t entry_num = 0; entry_num < env.N; entry_num += 1)
        {
            key_value = data_gen->gen_kv_pair(env.E);
            db->Put(write_opt, key_value.first, key_value.second);
            progress.update();
        }
        progress.finish();
    }

    spdlog::info("Waiting for initial compactions to finish...");
    wait_for_compactions(db, compactor);

    // print_db_status(db);
    delete data_gen;

    // 2️⃣ 预热阶段：执行1/4的总的操作数量
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Warming up cache with {} queries", env.queries / 4);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;  // 为进度条预留空行

    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);
    uint64_t warmup_queries = env.queries / 4;
    {
        ProgressBar progress(warmup_queries, "🔥 WarmUp Data  ");
        for (size_t i = 0; i < warmup_queries; i++)
        {
            std::string key = data_gen->gen_existing_key();
            db->Get(read_options, key, &value);
            progress.update();
        }
        progress.finish();
    }

    delete data_gen;

    // 3️⃣ 实际执行混合工作负载
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Mixed R/W workload with Online Tuning");
    spdlog::info("Workload1: R={:.0f}% | Workload2: R={:.0f}%", 
        env.read_ratio_1 * 100, env.read_ratio_2 * 100);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;

    // 清空前面的统计数据(唯一的一次重置❗️)
    reset_all_statistics(rocksdb_opt, compactor);

    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);

    std::mt19937 engine;
    if (env.seed != 0) {
        engine.seed(env.seed);
    } else {
        engine.seed(std::time(nullptr));
    }
    std::uniform_real_distribution<double> dist(0, 1);

    auto time_start = std::chrono::high_resolution_clock::now(); 

    int read_ops = 0;
    int write_ops = 0;

    double read_ratio = 0;
    // 混合负载主循环-进度条显示
    {
        ProgressBar progress(env.queries, "🔄 Mixed R/W  ");
        for (size_t i = 0; i < env.queries; i++)
        {
            double r = dist(engine);
            if(i < env.queries / 2){
                read_ratio = env.read_ratio_1;
            } else {
                read_ratio = env.read_ratio_2;
            }

            if(r < read_ratio){ // 执行读
                read_ops = read_ops + 1;
                key = data_gen->gen_existing_key();
                status = db->Get(read_options, key, &value);
            } else { // 执行写
                write_ops = write_ops + 1;
                key_value = data_gen->gen_existing_kv_pair(env.E);
                db->Put(write_opt, key_value.first, key_value.second);
            }
            progress.update();
        }
        progress.finish();
    }

    // ==================== Step 9: 等待后台操作完成 ====================
    spdlog::info("Waiting for background operations to complete...");
    wait_for_compactions(db, compactor);

    auto time_end = std::chrono::high_resolution_clock::now();
    
    result.latency = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start).count(); 
    
    // ✅计算 H_cache
    std::map<std::string, uint64_t> stats;
    rocksdb_opt.statistics->getTickerMap(&stats);
    
    uint64_t cache_hits = stats["rocksdb.block.cache.hit"];
    uint64_t cache_misses = stats["rocksdb.block.cache.miss"];
    result.H_cache = (cache_hits + cache_misses) > 0 ?
        static_cast<double>(cache_hits) / static_cast<double>(cache_hits + cache_misses) : 0.0;


    // ✅ 计算 I/O 成本(写成本 + 读成本)
    uint64_t flush_bytes = stats["rocksdb.flush.write.bytes"];
    uint64_t compaction_write_bytes = stats["rocksdb.compact.write.bytes"];
    uint64_t compaction_read_bytes = stats["rocksdb.compact.read.bytes"];

    uint64_t total_write_bytes = flush_bytes + compaction_write_bytes;
    auto perf_ctx = rocksdb::get_perf_context();
    uint64_t user_read_bytes = perf_ctx->block_read_byte;
    uint64_t total_read_bytes = compaction_read_bytes + user_read_bytes;
    
    result.write_io_kb_per_op = static_cast<double>(total_write_bytes) / (env.queries * 1024.0);
    result.read_io_kb_per_op = static_cast<double>(total_read_bytes) / (env.queries * 1024.0);
    result.total_io_kb_per_op = result.write_io_kb_per_op + result.read_io_kb_per_op;

    // ==================== 保存结果 ====================    
    std::ofstream ofs;
    ofs.open(env.exp_output_file, std::ios::app);
    if (ofs.is_open()) {
        ofs << result.to_csv() << "\n";
        ofs.close();
    }

    // ==================== 清理 ====================
    db->Close();
    delete db;
    delete data_gen;

    rocksdb::DestroyDB(db_path, rocksdb::Options());
    ret = system(rm_db_cmd.c_str());
    if (ret != 0) {
        spdlog::warn("Failed to execute: {}, return code: {}", rm_db_cmd, ret);
    }

    spdlog::info("=== Execution Completed ===");
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]){
    // // ==================== 配置日志信息 ====================
    // 配置双输出日志：控制台只输出警告以上，文件输出所有info
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::warn);  // 控制台只显示警告和错误
    std::string log_file = "./data/default/exp_log.txt";
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
    file_sink->set_level(spdlog::level::info);
    auto logger = std::make_shared<spdlog::logger>("tuner_logger", 
        spdlog::sinks_init_list{console_sink, file_sink});
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%Y-%m-%d %T.%e][%l] %v");
    environment env = parse_args(argc, argv);

    if (env.verbose == 1)
    {
        spdlog::info("Log level: DEBUG");
        spdlog::set_level(spdlog::level::debug);
    }
    else if (env.verbose == 2)
    {
        spdlog::info("Log level: TRACE");
        spdlog::set_level(spdlog::level::trace);
    }
    else
    {
        spdlog::set_level(spdlog::level::info);
    }

    if (env.destroy_db)
    {
        spdlog::info("Destroying DB: {}", env.db_path);
        rocksdb::DestroyDB(env.db_path, rocksdb::Options());
    }

    run_experiment(env);
}