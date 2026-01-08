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

struct MotivatingExpResult
{
    double alpha;

    // Stage 0: α → B (Write Memory Size)
    uint64_t Mbuf;
    // Stage 0': α → C (Cache Size) [并行路径]
    uint64_t Mcache;

    // Stage 1: B → F_flush / Cthreshold (Flush 频率)
    uint64_t flush_count; // Flush 次数
    uint64_t flush_bytes; // Flush 写入字节数
    uint64_t Cthreshold;  // L0(磁盘中的第一层容量阈值/触发Compaction的阈值) 
    double flush_rate;    // Flush 频率 (次/秒)

    // Stage 2: F_flush + Cthreshold → F_comp (Compaction 频率)
    uint64_t compaction_count;  // Compaction 次数
    uint64_t compaction_read_bytes;   // Compaction 读取字节数
    uint64_t compaction_write_bytes;  // Compaction 写入字节数
    double compaction_rate;     // Compaction 频率 (次/秒)

    // Stage 3: F_comp → τ_sst (SST 平均生命周期)
    uint64_t sst_inv_count;     // SST文件失效的数量
    // double tau_sst_estimated;   // SST文件平均生命周期估算 (秒)

    // Stage 4: τ_sst → I_inv (失效率)
    uint64_t cache_inv_count;      // 缓存失效数量
    // double invalidation_rate;   // 缓存失效率估算

    // Stage 5: I_inv → (Cache Hit Rate)
    double H_cap;               // 容量命中率 (纯读测量)
    double H_val;               // ✅ 有效性命中率 (计算得出) H_val = H_cache / H_cap
    double H_cache;             // 总命中率 (混合读写测量)

    // I/O 成本 (KB/op)
    double write_io_kb_per_op;
    double read_io_kb_per_op;
    double total_io_kb_per_op;
    
    
    std::string to_csv() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6)
            << alpha << ","
            << Mbuf << ","
            << Mcache << ","
            << flush_count << ","
            << compaction_count << ","
            << sst_inv_count << ","
            << cache_inv_count << ","
            << H_cache << ","
            << H_cap << ","
            << H_val << "," 
            << write_io_kb_per_op << ","
            << read_io_kb_per_op << ","
            << total_io_kb_per_op;
        return oss.str();
    }
    
    static std::string csv_header() {
        return "alpha,Mbuf,Mcache,"
                "flush_count,compaction_count,sst_inv_count,cache_inv_count,"
                "H_cache,H_cap,H_val,"
                "write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op";
    }
};

typedef struct environment
{
    std::string db_path;

    // 工作负载配置
    double non_empty_reads = 0.25; // ✅
    double empty_reads = 0.25; 
    double range_reads = 0.25;
    double writes = 0.25; // ✅
    double dels = 0.0;

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
    size_t M = 0;                                    // 总内存预算
    size_t initial_write_memory = 64 * 1024 * 1024; // 初始写内存大小
    
    // Memory Tuner配置
    bool enable_memory_tuner = true;
    double write_weight = 1.0;
    double read_weight = 1.0;
    size_t sim_cache_size = 128 * 1024 * 1024;
    size_t tuning_interval_seconds = 180;
    size_t min_tuning_interval_seconds = 20;

    // Motivating Experiment 配置
    bool motivating_exp_mode = false;       // 是否启用 Motivating Experiment 模式
    double alpha_start = 0.05;              // Alpha 扫描起始值
    double alpha_end = 0.95;                // Alpha 扫描结束值
    double alpha_step = 0.05;               // Alpha 扫描步长
    std::string exp_output_file = "/data/motivating_exp_results.csv";  // 实验输出文件
              
    // 其他配置
    int verbose = 0;
    bool destroy_db = true;
    int max_rocksdb_levels = 64;
    int parallelism = 1;
    int seed = 0;
    std::string dist_mode = "zipfian";
    double skew = 0.99;                     // 默认 Zipfian skewness
    std::string key_log_file;

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
        (option("-B", "--initial-buffer-size") & integer("size", env.initial_write_memory)) % "buffer size",
        (option("-M", "--total-memory-size") & integer("size", env.M)) % "total memory size",
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
        (option("--sel") & number("num", env.sel)) % "selectivity of range query"
    );

    // Motivating Experiment 选项
    auto motivating_opt = "motivating experiment options:" % (
        (option("--motivating-exp").set(env.motivating_exp_mode, true)) % "enable motivating experiment mode",
        (option("--alpha-start") & number("val", env.alpha_start)) % "alpha sweep start (default: 0.05)",
        (option("--alpha-end") & number("val", env.alpha_end)) % "alpha sweep end (default: 0.95)",
        (option("--alpha-step") & number("val", env.alpha_step)) % "alpha sweep step (default: 0.05)",
        (option("-o", "--output") & value("file", env.exp_output_file)) % "output CSV file"
    );

    auto minor_opt = "minor options:" % (
        (option("--max_rocksdb_level") & integer("num", env.max_rocksdb_levels)) % "max levels",
        (option("--parallelism") & integer("num", env.parallelism)) % "parallelism",
        (option("--seed") & integer("num", env.seed)) % "seed for generating data"
    );

    auto cli = (general_opt, build_opt, run_opt, motivating_opt, minor_opt);

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

void print_final_statistics(rocksdb::DB *db, 
                            rocksdb::Options &rocksdb_opt,
                            tmpdb::Compactor *compactor,
                            std::chrono::milliseconds latency,
                            size_t total_operations)
{
    std::map<std::string, uint64_t> stats;
    rocksdb_opt.statistics->getTickerMap(&stats);
    
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);
    
    std::string files_per_level = "[";
    std::string size_per_level = "[";
    for (auto &level : cf_meta.levels)
    {
        files_per_level += std::to_string(level.files.size()) + ", ";
        size_per_level += std::to_string(level.size) + ", ";
    }
    files_per_level = files_per_level.substr(0, files_per_level.size() - 2) + "]";
    size_per_level = size_per_level.substr(0, size_per_level.size() - 2) + "]";
    
    spdlog::info("=== Final Statistics ===");
    spdlog::info("files_per_level : {}", files_per_level);
    spdlog::info("size_per_level : {}", size_per_level);
    
    spdlog::info("(l0, l1, l2plus) : ({}, {}, {})",
                 stats["rocksdb.l0.hit"],
                 stats["rocksdb.l1.hit"],
                 stats["rocksdb.l2andup.hit"]);
    
    spdlog::info("(bf_true_neg, bf_pos, bf_true_pos) : ({}, {}, {})",
                 stats["rocksdb.bloom.filter.useful"],
                 stats["rocksdb.bloom.filter.full.positive"],
                 stats["rocksdb.bloom.filter.full.true.positive"]);
    
    spdlog::info("(bytes_written, compact_read, compact_write, flush_write) : ({}, {}, {}, {})",
                 stats["rocksdb.bytes.written"],
                 stats["rocksdb.compact.read.bytes"],
                 stats["rocksdb.compact.write.bytes"],
                 stats["rocksdb.flush.write.bytes"]);

    spdlog::info("(total_latency) : ({})", latency.count());

    double cache_hit_rate = stats["rocksdb.block.cache.miss"] == 0 ? 0 : 
        double(stats["rocksdb.block.cache.hit"]) / 
        double(stats["rocksdb.block.cache.hit"] + stats["rocksdb.block.cache.miss"]);
    
    spdlog::info("(cache_hit_rate) : ({})", cache_hit_rate);
    spdlog::info("(cache_hit) : ({})", stats["rocksdb.block.cache.hit"]);
    spdlog::info("(cache_miss) : ({})", stats["rocksdb.block.cache.miss"]);

    spdlog::info("=== Compactor Statistics ===");
    spdlog::info("total_flush_count: {}", compactor->stats.total_flush_count.load());
    spdlog::info("total_compaction_count: {}", compactor->stats.total_compaction_count.load());

    // I/O 成本计算
    uint64_t flush_write_bytes = stats["rocksdb.flush.write.bytes"];
    uint64_t compact_write_bytes = stats["rocksdb.compact.write.bytes"];
    uint64_t total_write_bytes = flush_write_bytes + compact_write_bytes;

    uint64_t merge_read_bytes = stats["rocksdb.compact.read.bytes"];
    auto perf_ctx = rocksdb::get_perf_context();
    uint64_t query_read_bytes = perf_ctx->block_read_byte;
    uint64_t total_read_bytes = merge_read_bytes + query_read_bytes;
    
    double write_cost_kb_per_op = static_cast<double>(total_write_bytes) / (total_operations * 1024.0);
    double read_cost_kb_per_op = static_cast<double>(total_read_bytes) / (total_operations * 1024.0);
    double total_io_cost_kb_per_op = write_cost_kb_per_op + read_cost_kb_per_op;
    
    spdlog::info("I/O Cost (KB/op):");
    spdlog::info("(write_io_kb_per_op, read_io_kb_per_op, total_io_kb_per_op) : ({:.4f}, {:.4f}, {:.4f})",
             write_cost_kb_per_op, read_cost_kb_per_op, total_io_cost_kb_per_op);
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

// 单次的Alpha的实验：分离测量
MotivatingExpResult run_single_alpha_experiment(environment &env, double alpha)
{
    MotivatingExpResult result;
    result.alpha = alpha;
    
    // 计算内存分配-写内存+块缓存
    size_t Mbuf = static_cast<size_t>(alpha * env.M);
    size_t Mcache = env.M - Mbuf;
    
    result.Mbuf = Mbuf;
    result.Mcache = Mcache;
    
    // 创建唯一的数据库路径和日志路径
    std::string db_path = env.db_path + "_alpha_" + std::to_string(static_cast<int>(alpha * 100));
    // std::string wal_path = "/mnt/essd/wal_alpha_" + std::to_string(static_cast<int>(alpha * 100));
    
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Running alpha = {:.3f} (Mbuf={:.1f} MB, Mcache={:.1f} MB)",
             alpha, Mbuf / (1024.0*1024.0), Mcache / (1024.0*1024.0));
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // 销毁旧数据库
    rocksdb::DestroyDB(db_path, rocksdb::Options());
    std::string rm_db_cmd = "rm -rf " + db_path;
    // std::string rm_wal_cmd = "rm -rf " + wal_path;
    system(rm_db_cmd.c_str());
    // system(rm_wal_cmd.c_str());
    // std::string mkdir_wal_cmd = "mkdir -p " + wal_path;
    // system(mkdir_wal_cmd.c_str());
    
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
    // rocksdb_opt.max_total_wal_size = 128 * 1024 * 1024; // 128MB
    rocksdb_opt.max_bytes_for_level_multiplier = env.T; // 默认情况下是10
    rocksdb_opt.write_buffer_size = Mbuf;
    // rocksdb_opt.wal_dir = wal_path;
    
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
        return result;
    }

    rocksdb::WriteOptions write_opt;
    write_opt.low_pri = true;
    write_opt.disableWAL = true;
    // write_opt.disableWAL = false; //开启写日志

    ReadOptions read_options;
    read_options.total_order_seek = true;

    std::string value, key, limit;
    std::pair<std::string, std::string> key_value;

    // 1️⃣ 初始化LSM-tree阶段：注入N个entry
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Phase 1: Initializing LSM-tree with {} entries", env.N);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;  // 为进度条预留空行
    
    DataGenerator *data_gen = new YCSBGenerator(env.N, "uniform", 0.0);

    auto initial_time_start = std::chrono::high_resolution_clock::now();

    // 初始化阶段-进度条显示
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
    
    auto initial_time_end = std::chrono::high_resolution_clock::now();
    auto initial_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        initial_time_end - initial_time_start).count();
    spdlog::info("(init_time) : ({})", initial_time);

    print_db_status(db);
    delete data_gen;
    
    // 2️⃣ 预热阶段：执行1/4的总的操作数量
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Phase 2: Warming up cache with {} queries", env.queries / 4);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;  // 为进度条预留空行

    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);
    auto warmup_start = std::chrono::high_resolution_clock::now();
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
    auto warmup_end = std::chrono::high_resolution_clock::now();
    auto warmup_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        warmup_end - warmup_start).count();

    spdlog::info("(warmup_time) : ({})", warmup_time);
    delete data_gen;

    // 3️⃣ 执行纯读工作负载H_cap
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Phase 3: Read-Only workload to measure H_cap");
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;  // 为进度条预留空行

    reset_all_statistics(rocksdb_opt, compactor, tracker); // 重置所有统计信息 丢弃预热的命中率影响

    // 重新创建数据生成器以保证访问模式一致性
    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);
    auto read_start = std::chrono::high_resolution_clock::now();
    uint64_t read_queries = env.queries * env.non_empty_reads / (env.non_empty_reads + env.writes);
    {
        ProgressBar progress(read_queries, "📖 Read-Only  ");
        for (size_t i = 0; i < read_queries; i++)
        {
            std::string key = data_gen->gen_existing_key();
            db->Get(read_options, key, &value);
            progress.update();
        }
        progress.finish();
    }

    auto read_end = std::chrono::high_resolution_clock::now();
    auto read_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        read_end - read_start).count();

    spdlog::info("(read_only_time) : ({})", read_time);

    // 计算 H_cap
    std::map<std::string, uint64_t> read_stats;
    rocksdb_opt.statistics->getTickerMap(&read_stats);

    uint64_t read_phase_hits = read_stats["rocksdb.block.cache.hit"];
    uint64_t read_phase_misses = read_stats["rocksdb.block.cache.miss"];
    
    result.H_cap = (read_phase_hits + read_phase_misses) > 0 ?
        static_cast<double>(read_phase_hits) / static_cast<double>(read_phase_hits + read_phase_misses) : 0.0;

    spdlog::info("[Phase 3] H_cap = {:.4f} (hits={}, misses={})",
                 result.H_cap, read_phase_hits, read_phase_misses);
    delete data_gen;

    // 4️⃣ 清空缓存-重新预热
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Phase 4: Clear cache and re-warm");
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;  // 为进度条预留空行

    block_cache->SetCapacity(0);
    block_cache->SetCapacity(Mcache);
    spdlog::info("Cache cleared. Current usage: {} MB", block_cache->GetUsage() / (1024*1024));

    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);
    auto warmup_start2 = std::chrono::high_resolution_clock::now();
    {
        ProgressBar progress(warmup_queries, "🔥 WarmUp Data Again  ");
        for (size_t i = 0; i < warmup_queries; i++)
        {
            std::string key = data_gen->gen_existing_key();
            db->Get(read_options, key, &value);
            progress.update();
        }
        progress.finish();
    }
    auto warmup_end2 = std::chrono::high_resolution_clock::now();
    auto warmup_time2 = std::chrono::duration_cast<std::chrono::milliseconds>(
        warmup_end2 - warmup_start2).count();

    spdlog::info("(warmup_time2) : ({})", warmup_time2);
    delete data_gen;
    
    // 5️⃣ 执行混合工作负载
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Phase 5: Mixed R/W workload to measure H_cache");
    spdlog::info("  Workload: reads={:.0f}%, writes={:.0f}%", 
                 env.non_empty_reads * 100, env.writes * 100);
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    std::cout << std::endl;

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
    auto mix_time_start = std::chrono::high_resolution_clock::now();
    
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
    
    auto mix_time_end = std::chrono::high_resolution_clock::now();
    auto mix_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        mix_time_end - mix_time_start).count();

    spdlog::info("(mix_time) : ({})", mix_time);
    
    // ✅✅✅ 计算 H_cache
    std::map<std::string, uint64_t> mixed_stats;
    rocksdb_opt.statistics->getTickerMap(&mixed_stats);
    
    uint64_t mixed_phase_hits = mixed_stats["rocksdb.block.cache.hit"];
    uint64_t mixed_phase_misses = mixed_stats["rocksdb.block.cache.miss"];
    result.H_cache = (mixed_phase_hits + mixed_phase_misses) > 0 ?
        static_cast<double>(mixed_phase_hits) / static_cast<double>(mixed_phase_hits + mixed_phase_misses) : 0.0;

    // 计算 H_val
    result.H_val = (result.H_cap > 0.0) ?
        result.H_cache / result.H_cap : 0.0;
        
    spdlog::info("  [Phase 5] H_cache = {:.4f}, H_val = H_cache/H_cap = {:.4f}",
                 result.H_cache, result.H_val);

    // ✅✅✅ 收集五阶段链条指标
    // Stage 1-2: Flush 和 Compaction 统计 分别统计次数和字节数
    result.flush_count = compactor->stats.epoch_flush_count.load();
    result.flush_bytes = mixed_stats["rocksdb.flush.write.bytes"];
    // result.Cthreshold = compactor->get_L0_capacity_threshold();
    result.compaction_count = compactor->stats.epoch_compaction_count.load();
    result.compaction_read_bytes = mixed_stats["rocksdb.compact.read.bytes"];
    result.compaction_write_bytes = mixed_stats["rocksdb.compact.write.bytes"];

    double mixed_workload_latency_sec = mix_time / 1000.0;
    result.flush_rate = (mixed_workload_latency_sec > 0) ? 
        result.flush_count / mixed_workload_latency_sec : 0.0;
    result.compaction_rate = (mixed_workload_latency_sec > 0) ? 
        result.compaction_count / mixed_workload_latency_sec : 0.0;

    // Stage 3: SST失效数量
    result.sst_inv_count = compactor->stats.epoch_sst_files_invalidation.load();
    
    // Stage 4: 缓存失效数量
    result.cache_inv_count = tracker->GetStats().epoch_invalidated_entries.load();

    // I/O 成本(写成本 + 读成本)
    uint64_t total_write_bytes = result.flush_bytes + result.compaction_write_bytes;
    auto perf_ctx = rocksdb::get_perf_context();
    uint64_t user_read_bytes = perf_ctx->block_read_byte;
    uint64_t total_read_bytes = result.compaction_read_bytes + user_read_bytes;
    
    result.write_io_kb_per_op = static_cast<double>(total_write_bytes) / (env.queries * 1024.0);
    result.read_io_kb_per_op = static_cast<double>(total_read_bytes) / (env.queries * 1024.0);
    result.total_io_kb_per_op = result.write_io_kb_per_op + result.read_io_kb_per_op;
    
    // ✅ 添加五阶段链条验证日志
    spdlog::info("=== Five-Stage Chain Verification ===");
    spdlog::info("Stage 0:  α={:.3f} → Mbuf={:.1f} MB", result.alpha, result.Mbuf/(1024.0*1024.0));
    spdlog::info("Stage 0': α={:.3f} → Mcache={:.1f} MB", result.alpha, result.Mcache/(1024.0*1024.0));
    spdlog::info("Stage 1:  Flush count={}, bytes={} KB", 
                result.flush_count, result.flush_bytes/1024);
    spdlog::info("Stage 2:  Compaction count={}, read={} KB, write={} KB", 
                result.compaction_count, 
                result.compaction_read_bytes/1024, 
                result.compaction_write_bytes/1024);
    spdlog::info("Stage 3:  SST files invalidated={}", result.sst_inv_count);
    spdlog::info("Stage 4:  Cache entries invalidated={}", result.cache_inv_count);
    spdlog::info("Stage 5:  H_cap={:.4f}, H_val={:.4f}, H_cache={:.4f}", 
                result.H_cap, result.H_val, result.H_cache);
    // ==================== 清理 ====================
    db->Close();
    delete db;
    delete data_gen;
    
    rocksdb::DestroyDB(db_path, rocksdb::Options());
    system(rm_db_cmd.c_str());
    // system(rm_wal_cmd.c_str());
    
    return result;
}

int run_motivating_experiment(environment &env)
{
    // spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    // spdlog::info("Motivating Experiment: Five-Stage Coupling Model Verification");
    // spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // 总内存预算M + 初始化总数据量 + 工作负载读写比例
    spdlog::info("Configuration:");
    spdlog::info("  Total Memory (M): {} MB", env.M / (1024*1024));
    spdlog::info("  Entries (N): {} (~{} GB data)", env.N, (env.N * env.E) / (1024*1024*1024));
    spdlog::info("  Entry Size (E): {} bytes", env.E);
    spdlog::info("  Size Ratio (T): {}", env.T);
    spdlog::info("  Workload: empty_reads={:.2f}, non_empty_reads={:.2f}, range={:.2f}, writes={:.2f}",
                 env.empty_reads, env.non_empty_reads, env.range_reads, env.writes);
    spdlog::info("  Distribution: {} (skew={})", env.dist_mode, env.skew);
    spdlog::info("  Total Queries: {}", env.queries);
    spdlog::info("  Alpha Sweep: [{}, {}] step {}", env.alpha_start, env.alpha_end, env.alpha_step);
    spdlog::info("");
    
    std::vector<MotivatingExpResult> results;
    
    // Alpha扫描，依次执行每一个实验
    for (double alpha = env.alpha_start; alpha <= env.alpha_end + 0.001; alpha += env.alpha_step)
    {
        MotivatingExpResult r = run_single_alpha_experiment(env, alpha);
        results.push_back(r);
    }
    
    // ==================== 保存结果 ====================
    std::ofstream ofs(env.exp_output_file);
    if (ofs.is_open())
    {
        ofs << MotivatingExpResult::csv_header() << "\n";
        for (const auto& r : results)
        {
            ofs << r.to_csv() << "\n";
        }
        ofs.close();
        spdlog::info("Results saved to: {}", env.exp_output_file);
    }
    else
    {
        spdlog::error("Failed to open output file: {}", env.exp_output_file);
    }
    
    spdlog::info("");
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    spdlog::info("Motivating Experiment Completed");
    spdlog::info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

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

    run_motivating_experiment(env);

}