#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>
#include <unistd.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

#include "clipp.h"
#include "spdlog/spdlog.h"

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
#include "tmpdb/memory_tuner.hpp"
#include "tmpdb/progress_bar.hpp"

using namespace ROCKSDB_NAMESPACE;

#define PAGESIZE 4096 // 4K  (B)

struct SamplingExpResult
{
    // 实验参数配置
    size_t M;           // 总内存
    size_t N;           // 数据量
    size_t Q;           // 查询数
    double T;           // size ratio
    double skewness;    // Zipfian参数
    double read_ratio;  // 读比例
    double write_ratio; // 写比例
    
    double alpha;

    // Stage 0: α → B (Write Memory Size)
    uint64_t Mbuf;
    // Stage 0': α → C (Cache Size) [并行路径]
    uint64_t Mcache;

    // Stage 1: B → F_flush (Flush 频率)
    uint64_t flush_count; // Flush 次数
    double flush_rate;    // Flush 频率 (次/秒)

    // Stage 2: F_flush → F_comp (Compaction 频率)
    uint64_t compaction_count;  // Compaction 次数
    double compaction_rate;     // Compaction 频率 (次/秒)

    // Stage 3: F_comp → τ_sst (SST 平均生命周期)
    uint64_t sst_inv_count;     // SST文件失效数量
    double sst_inv_rate;      // SST文件失效率 (个/秒)
    
    // Stage 4: τ_sst → I_inv (失效率)
    uint64_t cache_inv_count;   // 缓存失效数量
    double cache_inv_rate;      // 缓存失效率 (个/秒)

    // Stage 5: I_inv → (Cache Hit Rate)
    double H_cache;             // 总命中率 (混合读写测量)

    // I/O 成本 (KB/op)
    double write_io_kb_per_op;
    double read_io_kb_per_op;
    double total_io_kb_per_op;
    
    // 时间 - 用于计算速率
    double latency; // 混合读写工作负载时间 (毫秒)

    std::string to_csv() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6)
            << (M / (1024.0 * 1024.0)) << ","
            << N << ","
            << Q << ","
            << T << ","
            << skewness << "," 
            << read_ratio << ","
            << write_ratio << ","
            << alpha << ","
            << (Mbuf / (1024.0 * 1024.0)) << ","
            << (Mcache / (1024.0 * 1024.0)) << ","
            << flush_count << ","
            << flush_rate << ","
            << compaction_count << ","
            << compaction_rate << ","
            << sst_inv_count << ","
            << sst_inv_rate << ","
            << cache_inv_count << ","
            << cache_inv_rate << ","
            << H_cache << ","
            << write_io_kb_per_op << ","
            << read_io_kb_per_op << ","
            << total_io_kb_per_op << ","
            << latency;
        return oss.str();
    }
    
    static std::string csv_header() {
        return "M_MB,N,Q,T,skewness,read_ratio,write_ratio,"
           "alpha,Mbuf_MB,Mcache_MB,"
           "flush_count,flush_rate,compaction_count,compaction_rate,"
           "sst_inv_count,sst_inv_rate,cache_inv_count,cache_inv_rate,"
           "H_cache,write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op,latency";
    }
};

typedef struct environment
{
    std::string db_path;

    // 工作负载配置
    double non_empty_reads = 0.25;
    double empty_reads = 0.25; 
    double range_reads = 0.25;
    double writes = 0.25;
    double dels = 0.0;

    size_t queries = 10;
    int sel = 2;
    int scaling = 1;
    std::string compaction_style = "level";
    double alpha = 0.5;  // 写内存占总内存的比例
    double T = 10;
    double K = 0;
    size_t E = 1 << 7;
    double bpe = 5.0;
    size_t N = 1e6;
    size_t L = 0;
    size_t M = 0; // 总内存预算

    std::string exp_output_file = "/data/sampling_results.csv";  // 实验输出文件
              
    // 其他配置
    int verbose = 0;
    bool destroy_db = true;
    int max_rocksdb_levels = 64;
    int parallelism = 1;
    int seed = 0;
    std::string dist_mode = "zipfian";
    double skew = 0.99;                     // 默认 Zipfian skewness
    bool append_mode = false;

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
        (option("-a", "--alpha") & number("alpha", env.alpha)) % "write buffer ratio",
        (option("-E", "--entry-size") & integer("size", env.E)) % "entry size",
        (option("-b", "--bpe") & number("bits", env.bpe)) % "bits per element",
        (option("-c", "--compaction") & value("mode", env.compaction_style)) % "compaction style",
        (option("-d", "--destroy").set(env.destroy_db)) % "destroy the DB if exists"
    );

    auto run_opt = "run options:" % (
        (option("-e", "--empty_reads") & number("num", env.empty_reads)) % "empty queries",
        (option("-r", "--non_empty_reads") & number("num", env.non_empty_reads)) % "non-empty queries",
        (option("-q", "--range_reads") & number("num", env.range_reads)) % "range reads",
        (option("-w", "--writes") & number("num", env.writes)) % "writes",
        (option("-s", "--queries") & integer("num", env.queries)) % "queries",
        (option("--dist") & value("mode", env.dist_mode)) % "distribution mode",
        (option("--skew") & number("num", env.skew)) % "skewness for zipfian",
        (option("--sel") & number("num", env.sel)) % "selectivity of range query",
        (option("-o", "--output") & value("file", env.exp_output_file)) % "output CSV file",
        (option("--append").set(env.append_mode, true)) % "append to output file (no header)"
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
                          tmpdb::Compactor *compactor,
                          std::shared_ptr<rocksdb::FileCacheTracker> tracker)
{
    rocksdb_opt.statistics->Reset();
    rocksdb::get_iostats_context()->Reset();
    rocksdb::get_perf_context()->Reset();
    compactor->stats.reset_epoch();
    if (tracker) {
        tracker->ResetEpochStats();
        tracker->Clear();
    }
}

int run_experiment(environment &env)
{
    SamplingExpResult result;

    result.M = env.M;
    result.N = env.N;
    result.Q = env.queries;
    result.T = env.T;
    result.skewness = env.skew;
    result.read_ratio = env.non_empty_reads;
    result.write_ratio = env.writes;

    result.alpha = env.alpha;
    
    // 计算内存分配-写内存+块缓存
    size_t Mbuf = static_cast<size_t>(env.alpha * env.M);
    size_t Mcache = env.M - Mbuf;
    
    result.Mbuf = Mbuf;
    result.Mcache = Mcache;
    
    // 创建唯一的数据库路径和日志路径
    std::string db_path = env.db_path + "_alpha_" + std::to_string(static_cast<int>(env.alpha * 100));
    
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Running alpha = {:.3f} (Mbuf={:.1f} MB, Mcache={:.1f} MB)",
             env.alpha, Mbuf / (1024.0*1024.0), Mcache / (1024.0*1024.0));
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // 销毁旧数据库
    rocksdb::DestroyDB(db_path, rocksdb::Options());
    std::string rm_db_cmd = "rm -rf " + db_path;
    system(rm_db_cmd.c_str());

    // ==================== 配置 RocksDB ====================
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
    rocksdb_opt.max_bytes_for_level_multiplier = env.T; // 默认情况下是10
    rocksdb_opt.write_buffer_size = Mbuf;
    
    // ==================== 配置自定义Compactor ====================
    tmpdb::Compactor *compactor = nullptr;
    tmpdb::CompactorOptions compactor_opt;
    
    compactor_opt.size_ratio = env.T;
    compactor_opt.buffer_size = Mbuf;
    compactor_opt.entry_size = env.E;
    compactor_opt.bits_per_element = env.bpe;
    compactor_opt.num_entries = env.N;

    if (env.compaction_style == "level")
        compactor_opt.K = 1;
    else if (env.compaction_style == "tier")
        compactor_opt.K = env.T;
    else
        compactor_opt.K = env.K;

    compactor_opt.levels = tmpdb::Compactor::estimate_levels(env.N, env.T, env.E, Mbuf) 
                           * compactor_opt.K + 1;
    rocksdb_opt.num_levels = compactor_opt.levels + 1;

    compactor = new tmpdb::Compactor(compactor_opt, rocksdb_opt);
    rocksdb_opt.listeners.emplace_back(compactor);
    
    // ==================== 配置 Block Cache ====================
    rocksdb::BlockBasedTableOptions table_options;
    table_options.read_amp_bytes_per_bit = 32;

    table_options.filter_policy.reset(
        rocksdb::NewMonkeyFilterPolicy(
            compactor_opt.bits_per_element,
            compactor_opt.size_ratio,
            compactor_opt.levels));
    
    std::shared_ptr<Cache> block_cache = rocksdb::NewLRUCache(Mcache);
    // 启用 FileCacheTracker
    std::shared_ptr<rocksdb::FileCacheTracker> tracker = std::make_shared<rocksdb::FileCacheTracker>();

    table_options.block_cache = block_cache;
    table_options.file_cache_tracker = tracker; // ✅ 设置缓存跟踪器

    rocksdb_opt.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    rocksdb_opt.statistics = rocksdb::CreateDBStatistics();
    
    // ==================== 打开数据库 ====================
    rocksdb::DB *db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(rocksdb_opt, db_path, &db);
    if (!status.ok())
    {
        spdlog::error("Problems opening DB: {}", status.ToString());
        return EXIT_FAILURE;
    }

    rocksdb::WriteOptions write_opt;
    write_opt.low_pri = true;
    write_opt.disableWAL = true; //关闭写日志

    ReadOptions read_options;
    read_options.total_order_seek = true;

    std::string value, key, limit;
    std::pair<std::string, std::string> key_value;

    // 1️⃣ 初始化LSM-tree阶段：注入N个entry
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Initializing LSM-tree with {} entries", env.N);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;  // 为进度条预留空行
    
    DataGenerator *data_gen = new YCSBGenerator(env.N, "uniform", 0.0);
    {
        ProgressBar progress(env.N, "📥 Phase1: Init Data  ");
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
    

    print_db_status(db);
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
    spdlog::info("Mixed R/W workload: reads={:.0f}%, writes={:.0f}%", 
        env.non_empty_reads * 100, env.writes * 100);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;
    
    // 清空前面的统计数据
    reset_all_statistics(rocksdb_opt, compactor, tracker);

    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);

    rocksdb::Iterator *it = db->NewIterator(read_options);
    double p[] = {env.empty_reads, env.non_empty_reads, env.range_reads, env.writes};
    double cumprob[] = {p[0], p[0] + p[1], p[0] + p[1] + p[2], 1.0};

    std::mt19937 engine;
    if (env.seed != 0) {
        engine.seed(env.seed);
    } else {
        engine.seed(std::time(nullptr));
    }
    std::uniform_real_distribution<double> dist(0, 1);
    env.sel = PAGESIZE / env.E;
    auto time_start = std::chrono::high_resolution_clock::now();
    
    // 混合负载主循环-进度条显示
    {
        ProgressBar progress(env.queries, "🔄 Phase5: Mixed R/W  ");
        for (size_t i = 0; i < env.queries; i++)
        {
            double r = dist(engine);
            int outcome = 0;
            for (int j = 0; j < 4; j++)
            {
                if (r < cumprob[j])
                {
                    outcome = j;
                    break;
                }
            }
            
            switch (outcome)
            {
                case 0:  // Empty read
                {
                    key = data_gen->gen_new_dup_key();
                    status = db->Get(read_options, key, &value);
                    break;
                }
                case 1:  // Non-empty read ✅
                {
                    key = data_gen->gen_existing_key();
                    status = db->Get(read_options, key, &value);
                    break;
                }
                case 2:  // Range read
                {
                    key = data_gen->gen_existing_key();
                    limit = std::to_string(stoi(key) + 1 + env.sel);
                    for (it->Seek(rocksdb::Slice(key)); 
                        it->Valid() && it->key().ToString() < limit; 
                        it->Next())
                    {
                        value = it->value().ToString();
                    }
                    break;
                }
                case 3:  // Write ✅
                {
                    key_value = data_gen->gen_existing_kv_pair(env.E);
                    db->Put(write_opt, key_value.first, key_value.second);
                    break;
                }
                default:
                    break;
            }
            progress.update();
        }
        progress.finish();
    }

    delete it;
    wait_for_compactions(db, compactor);
    
    auto time_end = std::chrono::high_resolution_clock::now();
    result.latency = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start).count();
    double latency_sec = result.latency / 1000.0; // 转换为秒

    // ✅✅✅ 计算 H_cache
    std::map<std::string, uint64_t> stats;
    rocksdb_opt.statistics->getTickerMap(&stats);
    
    uint64_t cache_hits = stats["rocksdb.block.cache.hit"];
    uint64_t cache_misses = stats["rocksdb.block.cache.miss"];
    result.H_cache = (cache_hits + cache_misses) > 0 ?
        static_cast<double>(cache_hits) / static_cast<double>(cache_hits + cache_misses) : 0.0;

    // ✅✅✅ 收集五阶段链条指标
    // Stage 1-2: Flush 和 Compaction 统计 分别统计次数和字节数
    result.flush_count = compactor->stats.epoch_flush_count.load();
    result.compaction_count = compactor->stats.epoch_compaction_count.load();

    result.flush_rate = (latency_sec > 0) ? result.flush_count / latency_sec : 0.0;
    result.compaction_rate = (latency_sec > 0) ? result.compaction_count / latency_sec : 0.0;

    // Stage 3: SST失效数量
    result.sst_inv_count = compactor->stats.epoch_sst_files_invalidation.load();
    result.sst_inv_rate = (latency_sec > 0) ? result.sst_inv_count / latency_sec : 0.0;
    
    // Stage 4: 缓存失效数量
    result.cache_inv_count = tracker->GetStats().epoch_invalidated_entries.load();
    result.cache_inv_rate = (latency_sec > 0) ? result.cache_inv_count / latency_sec : 0.0;

    // I/O 成本(写成本 + 读成本)
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
    
    // ==================== 清理 ====================
    db->Close();
    delete db;
    delete data_gen;
    
    rocksdb::DestroyDB(db_path, rocksdb::Options());
    system(rm_db_cmd.c_str());
    // ==================== 保存结果 ====================
    std::ofstream ofs;
    if (env.append_mode) {
        ofs.open(env.exp_output_file, std::ios::app);  // 追加模式 ✅
    } else {
        ofs.open(env.exp_output_file);
        ofs << SamplingExpResult::csv_header() << "\n";  // 写表头
    }

    if (ofs.is_open()) {
        ofs << result.to_csv() << "\n";
        ofs.close();
    }
    
    return EXIT_SUCCESS;
}


int main(int argc, char *argv[])
{
    spdlog::set_pattern("[%T.%e]%^[%l]%$ %v");

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