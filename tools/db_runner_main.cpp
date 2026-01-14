#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>
#include <unistd.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cmath>

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
#include "tmpdb/memory_tuner.hpp"
#include "tmpdb/progress_bar.hpp"
#include "tmpdb/rl_tuner.hpp"

using namespace ROCKSDB_NAMESPACE;

#define PAGESIZE 4096 // 4K  (B)

// 累积统计快照，用于计算epoch增量
struct StatsSnapshot {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t flush_bytes;
    uint64_t compaction_write_bytes;
    uint64_t compaction_read_bytes;
    uint64_t user_read_bytes;
    std::chrono::high_resolution_clock::time_point timestamp;
    
    StatsSnapshot() : cache_hits(0), cache_misses(0), flush_bytes(0),
                      compaction_write_bytes(0), compaction_read_bytes(0),
                      user_read_bytes(0) {}
};

// 单个Epoch的性能统计
struct EpochStats {
    int epoch_id;               // Epoch编号
    double alpha;               // 当前alpha值
    size_t queries;             // 该epoch执行的查询数
    
    // 性能指标
    double H_cache;             // 缓存命中率
    double latency_ms;          // 延迟 (毫秒)
    double write_io_kb_per_op;  // 写IO成本 (KB/op)
    double read_io_kb_per_op;   // 读IO成本 (KB/op)
    double total_io_kb_per_op;  // 总IO成本 (KB/op)
    
    // 综合性能分数 (用于RL决策)
    double performance_score;

    std::string to_csv() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6)
            << epoch_id << ","
            << alpha << ","
            << queries << ","
            << H_cache << ","
            << latency_ms << ","
            << write_io_kb_per_op << ","
            << read_io_kb_per_op << ","
            << total_io_kb_per_op << ","
            << performance_score;
        return oss.str();
    }
    
    static std::string csv_header() {
        return "epoch_id,alpha,queries,H_cache,latency_ms,"
               "write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op,"
               "performance_score";
    }
};

// 实验整体结果(包含所有epoch性能统计)
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

    // Alpha相关
    double alpha_initial;
    double alpha_final;
    std::vector<double> alpha_history;

    uint64_t Mbuf;
    uint64_t Mcache;

    // 整体性能
    double H_cache;             // 总命中率 (混合读写测量)
    double write_io_kb_per_op;  // I/O 成本 (KB/op)
    double read_io_kb_per_op;
    double total_io_kb_per_op;
    double latency;  // 混合读写工作负载时间 (毫秒)

    // 每个epoch的性能记录
    std::vector<EpochStats> epoch_stats;

    // RL调优统计
    int rl_epochs_count;        // RL调优的epoch数
    int drift_count;            // 漂移检测次数
    bool converged;             // 是否收敛

    // 消融实验设置
    bool rl_agent_enabled;
    bool jump_start_enabled;

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
            << alpha_initial << "," 
            << alpha_final << ","
            << (Mbuf / (1024.0 * 1024.0)) << ","
            << (Mcache / (1024.0 * 1024.0)) << ","
            << H_cache << ","
            << write_io_kb_per_op << ","
            << read_io_kb_per_op << ","
            << total_io_kb_per_op << ","
            << latency << ","
            << rl_epochs_count << ","    // 🆕
            << drift_count << ","        // 🆕
            << (converged ? "true" : "false") << ","
            << (rl_agent_enabled ? "true" : "false") << ","
            << (jump_start_enabled ? "true" : "false");
        return oss.str();
    }
    
    static std::string csv_header() {
        return "M_MB,N,Q,T,skewness,"
            "read_ratio_1,write_ratio_1,read_ratio_2,write_ratio_2,"  // ✅ 4个字段
            "alpha_initial,alpha_final,Mbuf_MB,Mcache_MB,"
            "H_cache,write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op,"
            "latency,rl_epochs_count,drift_count,converged,rl_agent_enabled,jump_start_enabled";
    }

    // 输出alpha历史
    std::string alpha_history_str() const {
        std::ostringstream oss;
        for (size_t i = 0; i < alpha_history.size(); i++) {
            if (i > 0) oss << ";";
            oss << std::fixed << std::setprecision(4) << alpha_history[i];
        }
        return oss.str();
    }

    void save_epoch_details(const std::string& filepath) const {
        std::ofstream ofs;

        ofs.open(filepath, std::ios::app);
        
        if (ofs.is_open()) {
            for (const auto& es : epoch_stats) {
                ofs << es.to_csv() << "\n";
            }
            ofs.close();
        }
    }
};

typedef struct environment
{
    std::string db_path;

    // 工作负载1的配置
    double read_ratio_1 = 0.5;
    double write_ratio_1 = 0.5; 

    // 工作负载2的配置
    double read_ratio_2 = 0.5;
    double write_ratio_2 = 0.5; 
    
    size_t queries = 10;
    std::string compaction_style = "level";

    double alpha1 = 0.5;  // 针对工作负载1
    double alpha2 = 0.5;  // 针对工作负载2

    double T = 10;
    double K = 0;
    size_t E = 1 << 7;
    double bpe = 5.0;
    size_t N = 1e6;
    size_t L = 0;
    size_t M = 0; // 总内存预算

    size_t epoch_length = 10000; // 每多少次操作执行一次RL-tune

    // RL 配置
    double rl_step_size = 0.05;
    double rl_learning_rate = 0.1;   // Q-learning学习率
    double rl_discount_factor = 0.9;
    double rl_epsilon_start = 0.3;  // 初始探索率
    double rl_epsilon_decay = 0.95;  // 探索率衰减 
    double rl_epsilon_min = 0.05;  // 最小探索率
    double rl_ucb_c = 1.414; // UCB 探索系数

    // 性能权重(用于计算综合分数)
    double weight_hit_rate = 0.4;
    double weight_latency = 0.3;
    double weight_io = 0.3;

    double drift_threshold = 0.15;  // 读比例变化超过此值触发漂移

    // 消融实验
    bool enable_rl_tuning = true;     // 是否启用RL调优
    bool enable_jump_start = true;    // 是否启用漂移检测和Jump Start

    // 输出文件
    std::string exp_output_file = "/data/main_results.csv";  // 实验输出文件
    std::string epoch_output_file = "/data/epoch_results.csv";
              
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
        (option("-a1", "--alpha1") & number("alpha", env.alpha1)) % "Workload1 write buffer ratio",
        (option("-a2", "--alpha2") & number("alpha", env.alpha2)) % "Workload2 write buffer ratio",
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
        (option("-o1", "--output1") & value("file", env.exp_output_file)) % "output CSV file",
        (option("-o2", "--output2") & value("file", env.epoch_output_file)) % "output CSV file",
        (option("--append").set(env.append_mode, true)) % "append to output file (no header)",
        (option("--drift-threshold") & number("val", env.drift_threshold)) % "Drift detection threshold",
        (option("--rl-agent").set(env.enable_rl_tuning, true)) % "Enable RL agent tuning",
        (option("--no-rl-agent").set(env.enable_rl_tuning, false)) % "Disable RL agent tuning",
        (option("--jump-start").set(env.enable_jump_start, true)) % "Enable drift detection and jump start",
        (option("--no-jump-start").set(env.enable_jump_start, false)) % "Disable drift detection"
    );

    auto rl_opt = "RL tuning options:" % (
        (option("--rl-step") & number("step", env.rl_step_size)) % "RL step size",
        (option("--rl-learning-rate") & number("val", env.rl_learning_rate)) % "Q-learning rate",
        (option("--rl-discount") & number("val", env.rl_discount_factor)) % "Discount factor",
        (option("--rl-epsilon-start") & number("val", env.rl_epsilon_start)) % "Initial epsilon",
        (option("--rl-epsilon-decay") & number("val", env.rl_epsilon_decay)) % "Epsilon decay rate",
        (option("--rl-epsilon-min") & number("val", env.rl_epsilon_min)) % "Min epsilon",
        (option("--rl-ucb-c") & number("val", env.rl_ucb_c)) % "UCB coefficient",
        (option("--epoch-ops") & integer("num", env.epoch_length)) % "Queries per RL epoch"
    );

    auto minor_opt = "minor options:" % (
        (option("--max_rocksdb_level") & integer("num", env.max_rocksdb_levels)) % "max levels",
        (option("--parallelism") & integer("num", env.parallelism)) % "parallelism",
        (option("--seed") & integer("num", env.seed)) % "seed for generating data"
    );

    auto cli = (general_opt, build_opt, run_opt, rl_opt, minor_opt);

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

void apply_memory_allocation(rocksdb::DB* db, 
                             std::shared_ptr<rocksdb::Cache> block_cache,
                             tmpdb::Compactor* compactor,
                             size_t M, 
                             double alpha)
{
    // 计算新的写内存大小 + 块缓存大小
    size_t Mbuf = static_cast<size_t>(alpha * M);
    size_t Mcache = M - Mbuf;

    // 更新块缓存大小
    block_cache->SetCapacity(Mcache);
    spdlog::info("Block cache capacity set to {} MB", Mcache / (1024 * 1024));

    // 更新写内存大小
    rocksdb::Status s = db->SetOptions({{"write_buffer_size", std::to_string(Mbuf)}});
    if (s.ok()) {
        spdlog::info("Write buffer size set to {} MB", Mbuf / (1024 * 1024));
    } else {
        spdlog::info("Failed to set write buffer size: {}", s.ToString());
    }

    compactor->updateM(Mbuf);// 动态更新，后续使用新的B进行判断
}

// 获取当前的累积统计快照
StatsSnapshot get_current_stats_snapshot(rocksdb::Options& rocksdb_opt) {
    StatsSnapshot snap;
    snap.timestamp = std::chrono::high_resolution_clock::now();
    
    std::map<std::string, uint64_t> ticker_stats;
    rocksdb_opt.statistics->getTickerMap(&ticker_stats);
    
    snap.cache_hits = ticker_stats["rocksdb.block.cache.hit"];
    snap.cache_misses = ticker_stats["rocksdb.block.cache.miss"];
    snap.flush_bytes = ticker_stats["rocksdb.flush.write.bytes"];
    snap.compaction_write_bytes = ticker_stats["rocksdb.compact.write.bytes"];
    snap.compaction_read_bytes = ticker_stats["rocksdb.compact.read.bytes"];
    
    auto perf_ctx = rocksdb::get_perf_context();
    snap.user_read_bytes = perf_ctx->block_read_byte;
    
    return snap;
}

// 根据两个快照计算epoch的增量统计
EpochStats calculate_epoch_stats(
    const StatsSnapshot& start_snap,
    const StatsSnapshot& end_snap,
    int epoch_id,
    double alpha,
    size_t queries)
{
    EpochStats stats;
    stats.epoch_id = epoch_id;
    stats.alpha = alpha;
    stats.queries = queries;
    
    // 计算时间增量
    stats.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_snap.timestamp - start_snap.timestamp).count();
    
    // 计算缓存命中率增量
    uint64_t delta_hits = end_snap.cache_hits - start_snap.cache_hits;
    uint64_t delta_misses = end_snap.cache_misses - start_snap.cache_misses;
    stats.H_cache = (delta_hits + delta_misses > 0) ?
        static_cast<double>(delta_hits) / (delta_hits + delta_misses) : 0.0;
    
    // 计算IO增量
    uint64_t delta_flush = end_snap.flush_bytes - start_snap.flush_bytes;
    uint64_t delta_compact_write = end_snap.compaction_write_bytes - start_snap.compaction_write_bytes;
    uint64_t delta_compact_read = end_snap.compaction_read_bytes - start_snap.compaction_read_bytes;
    uint64_t delta_user_read = end_snap.user_read_bytes - start_snap.user_read_bytes;
    
    uint64_t delta_write_bytes = delta_flush + delta_compact_write;
    uint64_t delta_read_bytes = delta_compact_read + delta_user_read;
    
    stats.write_io_kb_per_op = (queries > 0) ? 
        static_cast<double>(delta_write_bytes) / (queries * 1024.0) : 0.0;
    stats.read_io_kb_per_op = (queries > 0) ? 
        static_cast<double>(delta_read_bytes) / (queries * 1024.0) : 0.0;
    stats.total_io_kb_per_op = stats.write_io_kb_per_op + stats.read_io_kb_per_op;
    
    return stats;
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
    // ==================== 配置日志信息 ====================
    // 配置双输出日志：控制台只输出警告以上，文件输出所有info
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::warn);  // 控制台只显示警告和错误

    string output_name;
    if(env.enable_rl_tuning && env.enable_jump_start){
        output_name = "rl_on_js_on";
    } else if(!env.enable_rl_tuning && !env.enable_jump_start){
        output_name = "rl_off_js_off";
    } else if(env.enable_rl_tuning && !env.enable_jump_start){
        output_name = "rl_on_js_off";
    } else {
        output_name = "rl_off_js_on";
    }
 

    std::string log_file = "./data/online_tuning_log_" + output_name + ".txt";
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
    file_sink->set_level(spdlog::level::info);  // 文件记录所有info

    auto logger = std::make_shared<spdlog::logger>("tuner_logger", 
        spdlog::sinks_init_list{console_sink, file_sink});
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(logger);
    spdlog::set_pattern("[%Y-%m-%d %T.%e][%l] %v");

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

    result.alpha_initial = env.alpha1;
    result.alpha_final = env.alpha1;
    result.rl_epochs_count = 0;
    result.drift_count = 0;
    result.converged = false;

    result.rl_agent_enabled = env.enable_rl_tuning;
    result.jump_start_enabled = env.enable_jump_start;

    // 计算初始情况下的内存分配-写内存+块缓存
    size_t Mbuf = static_cast<size_t>(env.alpha1 * env.M);
    size_t Mcache = env.M - Mbuf;
    
    result.Mbuf = Mbuf;
    result.Mcache = Mcache;
    
    // 创建唯一的数据库路径和日志路径
    std::string db_path = env.db_path + "_alpha_" + std::to_string(static_cast<int>(env.alpha1 * 100));
        
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
    
    // ==================== 配置 RL-Tuner ====================
    tmpdb::RLConfig rl_config;
    rl_config.step_size = env.rl_step_size;
    rl_config.learning_rate = env.rl_learning_rate;
    rl_config.discount_factor = env.rl_discount_factor;
    rl_config.epsilon_start = env.rl_epsilon_start;
    rl_config.epsilon_decay = env.rl_epsilon_decay;
    rl_config.epsilon_min = env.rl_epsilon_min;
    rl_config.ucb_c = env.rl_ucb_c;

    tmpdb::RLTuner rl_tuner(rl_config);
    rl_tuner.init(env.alpha1, env.read_ratio_1);
    if (env.seed != 0) {
        rl_tuner.set_seed(env.seed);
    }

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

    std::string value, key;
    std::pair<std::string, std::string> key_value;

    // 1️⃣ 初始化LSM-tree阶段：注入N个entry
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Initializing LSM-tree with {} entries", env.N);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;  // 为进度条预留空行
    
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
    spdlog::info("Mixed R/W workload with Online Tuning (epoch_size={})", env.epoch_length);
    spdlog::info("Workload1: R={:.0f}% | Workload2: R={:.0f}%", 
        env.read_ratio_1 * 100, env.read_ratio_2 * 100);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;

    // 清空前面的统计数据(唯一的一次重置❗️)
    reset_all_statistics(rocksdb_opt, compactor, tracker);

    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);

    std::mt19937 engine;
    if (env.seed != 0) {
        engine.seed(env.seed);
    } else {
        engine.seed(std::time(nullptr));
    }
    std::uniform_real_distribution<double> dist(0, 1);

    auto time_start = std::chrono::high_resolution_clock::now(); 

    int current_epoch = 0;
    size_t epoch_ops = 0;
    StatsSnapshot epoch_start_snap = get_current_stats_snapshot(rocksdb_opt);
    
    double read_ratio = 0;
    int curr_read_ops = 0;

    // 设置为1是因为我们的epoch足够大
    tmpdb::DriftDetector drift_detector(env.drift_threshold, 1);
    result.alpha_history.push_back(env.alpha1); // 初始化
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
                curr_read_ops = curr_read_ops + 1;
                key = data_gen->gen_existing_key();
                status = db->Get(read_options, key, &value);
            } else { // 执行写
                key_value = data_gen->gen_existing_kv_pair(env.E);
                db->Put(write_opt, key_value.first, key_value.second);
            }

            epoch_ops = epoch_ops + 1;

            if(epoch_ops >= env.epoch_length){
                // 获取当前累积的统计快照(end) + 计算本epoch内的增量统计(end - satrt)
                StatsSnapshot epoch_end_snap = get_current_stats_snapshot(rocksdb_opt);
                EpochStats epoch_stats = calculate_epoch_stats(
                    epoch_start_snap, epoch_end_snap,
                    current_epoch, rl_tuner.current_alpha(), epoch_ops);

                // 首先检查是否发生漂移 - 若漂移则JumpStart 不用RL-tune
                double curr_read_ratio = static_cast<double>(curr_read_ops) / epoch_ops;
                bool drift = drift_detector.observe(curr_read_ratio);

                double new_alpha;

                if(drift && env.enable_jump_start){
                   spdlog::info("🚨 Drift detected!");
                   rl_tuner.on_drift_detected(env.alpha2, curr_read_ratio);
                   new_alpha = env.alpha2;
                   result.drift_count++; 
                } else if(env.enable_rl_tuning){ // 正常RL更新
                    tmpdb::EpochPerf perf(epoch_stats.H_cache, epoch_stats.latency_ms,
                              epoch_stats.total_io_kb_per_op, curr_read_ratio);

                    if (rl_tuner.is_converged()) {
                        // 已收敛，保持当前alpha不变 - 不实际执行RL
                        new_alpha = rl_tuner.current_alpha();
                        spdlog::info("   ✅ Converged, keeping alpha={:.3f}", new_alpha);
                    } else {
                        new_alpha = rl_tuner.on_epoch_end(perf);
                        result.rl_epochs_count++;
                    }
                } else { // baseline模式 - 消融实验
                    new_alpha = rl_tuner.current_alpha();
                    spdlog::info("   📌 Static: α = {:.3f} (no tuning)", new_alpha);
                }

                // 应用新的alpha
                if (std::abs(new_alpha - result.alpha_history.back()) > 0.001) {
                    apply_memory_allocation(db, block_cache, compactor, env.M, new_alpha);
                    spdlog::info("   🔧 {} alpha: {:.3f} → {:.3f}", 
                                tmpdb::action_to_string(rl_tuner.last_action()),
                                result.alpha_history.back(), new_alpha);
                }

                if (drift) {
                    epoch_stats.performance_score = 0.0;  // drift时没有RL计算，设为0
                } else {
                    epoch_stats.performance_score = rl_tuner.last_reward();
                }
                
                spdlog::info("📊 Epoch[{}]: α={:.3f}, H={:.4f}, {}", 
                            current_epoch, new_alpha, epoch_stats.H_cache,
                            rl_tuner.get_stats_string());
                
                result.epoch_stats.push_back(epoch_stats);
                result.alpha_history.push_back(new_alpha);

                // 重置
                curr_read_ops = 0;
                epoch_ops = 0;
                current_epoch++;
                epoch_start_snap = epoch_end_snap;
            }
            progress.update();
        }
        progress.finish();
    }

    result.alpha_final = rl_tuner.current_alpha();
    result.converged = rl_tuner.is_converged();
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
        ofs << ExpResult::csv_header() << "\n";  // 写表头
    }

    if (ofs.is_open()) {
        ofs << result.to_csv() << "\n";
        ofs.close();
    }

    result.save_epoch_details(env.epoch_output_file);
    
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