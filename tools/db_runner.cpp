#include <chrono>
#include <iostream>
#include <ctime>
#include <filesystem>
#include <unistd.h>
#include <algorithm>

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
#include "tmpdb/compactor.hpp"
#include "infrastructure/data_generator.hpp"
#include "tmpdb/performance_monitor.hpp"
#include "tmpdb/decision_engine.hpp"
#include "tmpdb/memory_allocator.hpp"
#include "tmpdb/epoch_logger.hpp"

using namespace ROCKSDB_NAMESPACE;

#define PAGESIZE 4096

typedef struct environment
{
    std::string db_path;

    double non_empty_reads = 0.25;
    double empty_reads = 0.25;
    double range_reads = 0.25;
    double writes = 0.25;
    double dels = 0.0;
    size_t prime_reads = 0;

    size_t steps = 10;
    int sel = 2;
    int scaling = 1;
    std::string compaction_style = "level";
    // Build mode
    double T = 10;
    double K = 0;

    size_t B = 1 << 18;         //> 1 KB
    size_t E = 1 << 7;          //> 128 B
    size_t file_size = 4 * 1024 * 1024; //> 4M B;
    double bits_per_element = 5.0;
    size_t N = 1e6;
    size_t L = 0;

    // ===== 动态调参相关配置 =====
    size_t M = 0;      // ✅总内存预算
    double initial_alpha = 0.5;          // ✅初始alpha值(CAMAL得到的静态最优alpha)
    bool enable_dynamic_tuning = false;  // 是否启用动态调参
    size_t epoch_size = 1000;            // 每个epoch的操作数
    

    // Decision Engine 阈值配置
    double cache_hit_rate_low = 0.80;
    double write_stall_rate_high = 0.05;
    double compaction_rate_high = 5.0;
    double alpha_step = 0.05;
    int stability_window = 3;

    int verbose = 0;
    bool destroy_db = true;

    int max_rocksdb_levels = 64;
    int parallelism = 1;

    int seed = 0;

    std::string dist_mode = "zipfian";
    double skew = 0.5;

    size_t cache_cap = 0;
    bool use_cache = true;

    std::string key_log_file;
    bool use_key_log = true;

    // 增加日志文件相关配置
    std::string epoch_log_file; // 日志文件路径
    bool enable_epoch_log = false; // 是否启用日志记录

} environment;

environment parse_args(int argc, char *argv[])
{
    using namespace clipp;
    using std::to_string;

    size_t minimum_entry_size = 32;

    environment env;
    bool help = false;

    auto general_opt = "general options" % ((option("-v", "--verbose") & integer("level", env.verbose)) % ("Logging levels (DEFAULT: INFO, 1: DEBUG, 2: TRACE)"),
                                            (option("-h", "--help").set(help, true)) % "prints this message");

    auto build_opt = ("build options:" % ((value("db_path", env.db_path)) % "path to the db",
                                          (option("-N", "--entries") & integer("num", env.N)) % ("total entries, default pick [default: " + to_string(env.N) + "]"),
                                          (option("-T", "--size-ratio") & number("ratio", env.T)) % ("size ratio, [default: " + fmt::format("{:.0f}", env.T) + "]"),
                                          (option("-K", "--runs-number") & number("runs", env.K)) % ("size ratio, [default: " + fmt::format("{:.0f}", env.K) + "]"),
                                          (option("-f", "--file-size") & integer("size", env.file_size)) % ("file size (in bytes), [default: " + to_string(env.file_size) + "]"),
                                          (option("-B", "--buffer-size") & integer("size", env.B)) % ("buffer size (in bytes), [default: " + to_string(env.B) + "]"),
                                          (option("-M", "--total-memory-size") & integer("size", env.M)) % ("total memory size (in bytes), [default: " + to_string(env.M) + "]"), // ✅xxq新增，总内存预算
                                          (option("-E", "--entry-size") & integer("size", env.E)) % ("entry size (bytes) [default: " + to_string(env.E) + ", min: 32]"),
                                          (option("-b", "--bpe") & number("bits", env.bits_per_element)) % ("bits per entry per bloom filter [default: " + fmt::format("{:.1f}", env.bits_per_element) + "]"),
                                          (option("-c", "--compaction") & value("mode", env.compaction_style)) % "set level or tier compaction",
                                          (option("-d", "--destroy").set(env.destroy_db)) % "destroy the DB if it exists at the path"));

    auto run_opt = ("run options:" % ((option("-e", "--empty_reads") & number("num", env.empty_reads)) % ("empty queries, [default: " + to_string(env.empty_reads) + "]"),
                                      (option("-r", "--non_empty_reads") & number("num", env.non_empty_reads)) % ("non-empty queries, [default: " + to_string(env.non_empty_reads) + "]"),
                                      (option("-q", "--range_reads") & number("num", env.range_reads)) % ("range reads, [default: " + to_string(env.range_reads) + "]"),
                                      (option("-w", "--writes") & number("num", env.writes)) % ("writes, [default: " + to_string(env.writes) + "]"),
                                      (option("--dels") & number("num", env.writes)) % ("deletes, [default: " + to_string(env.dels) + "]"),
                                      (option("-s", "--steps") & integer("num", env.steps)) % ("steps, [default: " + to_string(env.steps) + "]"), //✅对应queries
                                      (option("--dist") & value("mode", env.dist_mode)) % ("distribution mode ['uniform', 'zipf']"),
                                      (option("--skew") & number("num", env.skew)) % ("skewness for zipfian [0, 1)"),
                                      (option("--sel") & number("num", env.sel)) % ("selectivity of range query"),
                                      (option("--scaling") & number("num", env.scaling)) % ("scaling"),
                                      (option("--cache").set(env.use_cache, true) & number("cap", env.cache_cap)) % "use block cache",
                                      (option("--key-log-file").set(env.use_key_log, true) & value("file", env.key_log_file)) % "use keylog to record each key"));
    // ===== ✅新增：动态调参选项 =====
    auto tuning_opt = ("dynamic tuning options:" % ((option("--enable-tuning").set(env.enable_dynamic_tuning, true)) % "enable dynamic memory tuning",
                                                    (option("--epoch-size") & integer("num", env.epoch_size)) % ("operations per epoch [default: " + to_string(env.epoch_size) + "]"),
                                                    (option("--initial-alpha") & number("alpha", env.initial_alpha)) % ("initial alpha value [default: " + fmt::format("{:.2f}", env.initial_alpha) + "]"),
                                                    (option("--hit-rate-low") & number("rate", env.cache_hit_rate_low)) % ("cache hit rate low threshold [default: " + fmt::format("{:.2f}", env.cache_hit_rate_low) + "]"),
                                                    (option("--stall-rate-high") & number("rate", env.write_stall_rate_high)) % ("write stall rate high threshold [default: " + fmt::format("{:.2f}", env.write_stall_rate_high) + "]"),
                                                    (option("--compaction-rate-high") & number("rate", env.compaction_rate_high)) % ("compaction rate high threshold [default: " + fmt::format("{:.1f}", env.compaction_rate_high) + "]"),
                                                    (option("--alpha-step") & number("step", env.alpha_step)) % ("alpha adjustment step [default: " + fmt::format("{:.2f}", env.alpha_step) + "]"),
                                                    (option("--stability-window") & integer("window", env.stability_window)) % ("stability window size [default: " + to_string(env.stability_window) + "]"),
                                                    (option("--enable-epoch-log").set(env.enable_epoch_log, true)) % "enable epoch logging",
                                                    // (option("--epoch-log-file") & value("file", env.epoch_log_file)) % "epoch log file path"));

    auto minor_opt = ("minor options:" % ((option("--max_rocksdb_level") & integer("num", env.max_rocksdb_levels)) % ("limits the maximum levels rocksdb has [default: " + to_string(env.max_rocksdb_levels) + "]"),
                                          (option("--parallelism") & integer("num", env.parallelism)) % ("parallelism for writing to db [default: " + to_string(env.parallelism) + "]"),
                                          (option("--seed") & integer("num", env.seed)) % "seed for generating data [default: random from time]"));

    auto cli = (general_opt,
                build_opt,
                run_opt,
                tuning_opt, //✅新增
                minor_opt);

    if (!parse(argc, argv, cli))
        help = true;

    if (env.E < minimum_entry_size)
    {
        help = true;
        spdlog::error("Entry size is less than {} bytes", minimum_entry_size);
    }

    if (help)
    {
        auto fmt = doc_formatting{}.doc_column(42);
        std::cout << make_man_page(cli, "db_builder", fmt);
        exit(EXIT_FAILURE);
    }

    return env;
}


void print_db_status(rocksdb::DB *db)
{
    spdlog::debug("Files per level");
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::vector<std::string> file_names;
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

// 新增函数
void wait_for_compactions(rocksdb::DB *db, tmpdb::Compactor *compactor)
{
    uint64_t num_running_flushes, num_pending_flushes;
    
    while (true)
    {
        db->GetIntProperty(DB::Properties::kNumRunningFlushes, &num_running_flushes);
        db->GetIntProperty(DB::Properties::kMemTableFlushPending, &num_pending_flushes);
        if (num_running_flushes == 0 && num_pending_flushes == 0)
            break;
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    while (compactor->compactions_left_count > 0)
        ;
    
    while (compactor->requires_compaction(db))
    {
        while (compactor->compactions_left_count > 0)
            ;
    }
}

// 新增函数：打印最终统计信息
void print_final_statistics(rocksdb::DB *db, 
                            rocksdb::Options &rocksdb_opt,
                            tmpdb::Compactor *compactor,
                            std::chrono::milliseconds latency)
{
    std::map<std::string, uint64_t> stats;
    rocksdb_opt.statistics->getTickerMap(&stats);
    
    // 获取每层文件分布
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

    spdlog::info("(total_read, estimate_read) : ({}, {})",
                stats["rocksdb.read.amp.total.read.bytes"],
                stats["rocksdb.read.amp.estimate.useful.bytes"]);

    spdlog::info("(total_latency) : ({})", latency.count());

    // 命中率相关
    double cache_hit_rate = stats["rocksdb.block.cache.miss"] == 0 ? 0 : double(stats["rocksdb.block.cache.hit"]) / double(stats["rocksdb.block.cache.hit"] + stats["rocksdb.block.cache.miss"]);
    if (cache_hit_rate < 1e-3)
        spdlog::info("(cache_hit_rate) : ({})", 0.0);
    else
        spdlog::info("(cache_hit_rate) : ({})", cache_hit_rate);
    spdlog::info("(cache_hit) : ({})", stats["rocksdb.block.cache.hit"]);
    spdlog::info("(cache_miss) : ({})", stats["rocksdb.block.cache.miss"]);


    // Compactor 统计
    spdlog::info("=== Compactor Statistics ===");
    spdlog::info("total_flush_count: {}", compactor->stats.total_flush_count.load());
    spdlog::info("total_compaction_count: {}", compactor->stats.total_compaction_count.load());
    spdlog::info("total_compaction_input_files: {}", compactor->stats.total_input_files.load());
    spdlog::info("total_compaction_read_bytes: {}", compactor->stats.total_compaction_read_bytes.load());
    spdlog::info("total_compaction_write_bytes: {}", compactor->stats.total_compaction_write_bytes.load());
}

int main(int argc, char *argv[])
{
    // ==================== Step 1: 初始化 ====================
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

    // ==================== Step 2: 配置 RocksDB ====================
    spdlog::info("Building DB: {}", env.db_path);
    rocksdb::Options rocksdb_opt; // 设置options

    rocksdb_opt.create_if_missing = true;
    rocksdb_opt.error_if_exists = true;
    rocksdb_opt.IncreaseParallelism(env.parallelism);
    rocksdb_opt.compression = rocksdb::kNoCompression;
    rocksdb_opt.bottommost_compression = kNoCompression;
    rocksdb_opt.use_direct_reads = true;
    rocksdb_opt.use_direct_io_for_flush_and_compaction = true;
    rocksdb_opt.target_file_size_base = env.scaling * env.file_size;

    // 禁用 RocksDB 原生 Compaction
    rocksdb_opt.compaction_style = rocksdb::kCompactionStyleNone;
    rocksdb_opt.disable_auto_compactions = true;
    rocksdb_opt.write_buffer_size = env.B / 2;
    rocksdb_opt.max_write_buffer_number = 2; // MemTable数量 默认也是2

    // ==================== Step 3: 配置自定义 Compactor ====================
    tmpdb::Compactor *compactor = nullptr; //配置自定义的Compactor(CAMAL自行模拟)
    tmpdb::CompactorOptions compactor_opt;
    compactor_opt.size_ratio = env.T;
    compactor_opt.buffer_size = env.B / 2;
    compactor_opt.entry_size = env.E;
    compactor_opt.bits_per_element = env.bits_per_element;
    compactor_opt.num_entries = env.N;

    if (env.compaction_style == "level")
        compactor_opt.K = 1;
    else if (env.compaction_style == "tier")
        compactor_opt.K = env.T;
    else
        compactor_opt.K = env.K;

    compactor_opt.levels = tmpdb::Compactor::estimate_levels(env.N, env.T, env.E, env.B / 2) * compactor_opt.K + 1;
    rocksdb_opt.num_levels = compactor_opt.levels + 1;

    compactor = new tmpdb::Compactor(compactor_opt, rocksdb_opt);
    rocksdb_opt.listeners.emplace_back(compactor);

    // ==================== Step 4: 配置 Block Cache ====================
    rocksdb::BlockBasedTableOptions table_options;
    table_options.read_amp_bytes_per_bit = 32; // 启用读放大统计功能

    // 配置Monkey Bloom filter(在不同的level调整bpe)
    table_options.filter_policy.reset(
        rocksdb::NewMonkeyFilterPolicy(
            env.bits_per_element,
            compactor_opt.size_ratio,
            compactor_opt.levels));
    
    std::shared_ptr<Cache> block_cache = nullptr;
    // 配置Block Cache
    if (env.cache_cap == 0)
        table_options.no_block_cache = true;
    else{
        block_cache = rocksdb::NewLRUCache(env.cache_cap);
        table_options.block_cache = block_cache;
    }
    rocksdb_opt.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    rocksdb_opt.statistics = rocksdb::CreateDBStatistics();

    // ==================== Step 5: 打开数据库 ====================
    rocksdb::DB *db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(rocksdb_opt, env.db_path, &db);
    if (!status.ok())
    {
        spdlog::error("Problems opening DB");
        spdlog::error("{}", status.ToString());
        delete db;
        exit(EXIT_FAILURE);
    }

    // ==================== Step 6: 初始化动态调参组件 ====================
    tmpdb::PerformanceMonitor* perf_monitor = nullptr;
    tmpdb::DecisionEngine* decision_engine = nullptr;
    tmpdb::MemoryAllocator* memory_allocator = nullptr;
    tmpdb::EpochLogger* epoch_logger = nullptr;

    if (env.enable_dynamic_tuning)
    {
        // 检查必要条件
        // 实际在我们的执行过程中enable_dynamic_tuning=true时二者均不会为0
        if (env.M == 0) {
            spdlog::error("Total memory budget (-M) must be specified for dynamic tuning");
            exit(EXIT_FAILURE);
        }
        if (block_cache == nullptr) {
            spdlog::error("Block cache must be enabled for dynamic tuning");
            exit(EXIT_FAILURE);
        }

        spdlog::info("=== Dynamic Tuning Enabled ===");
        
        // 输出配置信息
        spdlog::info("Total memory budget: {} MB", env.M >> 20); // 单位MB
        spdlog::info("Initial alpha: {:.2f}", env.initial_alpha);
        spdlog::info("Epoch size: {} operations", env.epoch_size);
        
        // 创建 Performance Monitor
        perf_monitor = new tmpdb::PerformanceMonitor(
            db, rocksdb_opt.statistics.get(), compactor);
        
        // 创建 Decision Engine
        tmpdb::ThresholdConfig threshold_config;
        threshold_config.cache_hit_rate_low = env.cache_hit_rate_low;
        threshold_config.write_stall_rate_high = env.write_stall_rate_high;
        threshold_config.compaction_rate_high = env.compaction_rate_high;
        threshold_config.alpha_step = env.alpha_step;
        threshold_config.stability_window = env.stability_window;

        decision_engine = new tmpdb::DecisionEngine(
            env.initial_alpha, threshold_config);
        
        // 创建 Memory Allocator
        memory_allocator = new tmpdb::MemoryAllocator(
            db, block_cache, compactor,
            env.M, env.initial_alpha);
        
        memory_allocator->print_allocation();

        // 创建 Epoch Logger（如果启用）
        if (env.enable_epoch_log)
        {
            env.epoch_log_file = "data/log_file.csv";
            spdlog::info("Using default epoch log file: {}", env.epoch_log_file);
            epoch_logger = new tmpdb::EpochLogger();
            if (!epoch_logger->open(env.epoch_log_file))
            {
                spdlog::error("Failed to create epoch log file, continuing without logging");
                delete epoch_logger;
                epoch_logger = nullptr;
            }
        }

    }

    // ==================== Step 7: 初始化数据 ====================
    spdlog::info("Initializing data with {} entries...", env.N);

    rocksdb::WriteOptions write_opt;
    write_opt.low_pri = true;
    write_opt.disableWAL = true;

    // 这一部分在构造YCSB时的mode和skewness是随意设置的
    // 因为gen_kv_pair递归调用的gen_key并未使用dist_new，而是顺序生成"0", "1", ..., "N-1"
    DataGenerator *data_gen = new YCSBGenerator(env.N, "uniform", 0.0);
    std::pair<std::string, std::string> key_value;

    auto write_time_start = std::chrono::high_resolution_clock::now();
    for (size_t entry_num = 0; entry_num < env.N; entry_num += 1)
    {
        key_value = data_gen->gen_kv_pair(env.E); // 生成由[0, N-1]N个key组成的kv对
        db->Put(write_opt, key_value.first, key_value.second);

        // 打印生成的kv对进行验证(前10个和后10个)
        // if (entry_num < 10 || entry_num >= env.N - 10)
        // {
        //     spdlog::info("Insert[{}]: key='{}', value_size={}", 
        //                 entry_num, key_value.first, key_value.second.size());
        // }
    }

    // 等待初始化完成
    spdlog::info("Waiting for initial compactions to finish...");
    wait_for_compactions(db, compactor);

    auto write_time_end = std::chrono::high_resolution_clock::now();
    auto write_time = std::chrono::duration_cast<std::chrono::milliseconds>(write_time_end - write_time_start).count();
    // spdlog::info("Initialization completed in {} ms", write_time);
    spdlog::info("(init_time) : ({})", write_time);

    print_db_status(db);

    // 重置统计信息
    rocksdb_opt.statistics->Reset();
    rocksdb::get_iostats_context()->Reset();
    rocksdb::get_perf_context()->Reset();
    compactor->stats.reset_epoch();

    // ==================== Step 8: 执行 Workload ====================
    spdlog::info("=== Starting Workload Execution ===");
    spdlog::info("Total steps: {}, Epoch size: {}", env.steps, env.epoch_size);

    double p[] = {env.empty_reads, env.non_empty_reads, env.range_reads, env.writes};
    double cumprob[] = {p[0], p[0] + p[1], p[0] + p[1] + p[2], 1.0};
    
    std::string value, key, limit;
    delete data_gen;
    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);

    ReadOptions read_options;
    read_options.total_order_seek = true;
    rocksdb::Iterator *it = db->NewIterator(read_options);

    std::mt19937 engine;
    if (env.seed != 0)
    {
        engine.seed(env.seed); //使用用户指定的种子
    }
    else
    {
        engine.seed(std::time(nullptr));
    }
    std::uniform_real_distribution<double> dist(0, 1);

    env.sel = PAGESIZE / env.E;

    // 统计变量
    size_t epoch_count = 0;
    size_t total_adjustments = 0;

    auto time_start = std::chrono::high_resolution_clock::now();
    
    // ==================== 主循环 ====================
    for (size_t i = 0; i < env.steps; i++)
    {
        double r = dist(engine); // 生成一个0-1的随机数
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
        case 0:
        {
            key = data_gen->gen_new_dup_key();
            status = db->Get(read_options, key, &value);
            break;
        }
        case 1:
        {
            key = data_gen->gen_existing_key();
            status = db->Get(read_options, key, &value);
            break;
        }
        case 2:
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
        case 3:
        {
            key_value = data_gen->gen_existing_kv_pair(env.E);
            db->Put(write_opt, key_value.first, key_value.second);
            break;
        }
        default:
            break;
        }

        // ===== 动态调参：每个 epoch 结束时执行 =====
        if (env.enable_dynamic_tuning && ((i + 1) % env.epoch_size == 0))
        {
            epoch_count++;
            
            // Step 1: 采集性能指标
            tmpdb::PerformanceMetrics metrics = perf_monitor->collect();
            
            spdlog::info("--- Epoch {} (step {}/{}) ---", epoch_count, i + 1, env.steps);
            metrics.print();
            
            // Step 2: 诊断瓶颈（用于日志记录）
            spdlog::info("DEBUG: Before diagnose");
            bool write_bottleneck = decision_engine->is_write_bottleneck(metrics);
            bool read_bottleneck = decision_engine->is_read_bottleneck(metrics);

            // Step 3: 决策引擎做出决策
            spdlog::info("DEBUG: Before decide");
            double alpha_before = decision_engine->get_current_alpha();
            tmpdb::AdjustmentDecision decision = decision_engine->decide(metrics);
            
            // Step 4: 如果需要调整，执行调整
            spdlog::info("DEBUG: Before apply decision");
            double alpha_after = alpha_before;
            bool adjustment_applied = false;

            if (decision.action != tmpdb::AdjustmentAction::NO_CHANGE)
            {
                spdlog::info("DEBUG: Applying adjustment");
                double new_alpha = decision_engine->apply_decision(decision);
                
                // Step 4: 应用新的内存分配
                if (memory_allocator->adjust_allocation(new_alpha))
                {
                    total_adjustments++;
                    adjustment_applied = true;
                    alpha_after = memory_allocator->get_current_alpha();
                    memory_allocator->print_allocation();
                }
            }
            else
            {   
                spdlog::debug("No adjustment needed: {}", decision.reason);
            }

            // 将epoch的相关信息记录到CSV文件中
            spdlog::info("DEBUG: Before log_epoch"); 
            if (epoch_logger && epoch_logger->is_open())
            {
                epoch_logger->log_epoch(
                    epoch_count,                    // epoch 编号
                    i + 1,                          // 当前步数
                    metrics,                        // 性能指标
                    write_bottleneck,               // 写瓶颈
                    read_bottleneck,                // 读瓶颈
                    decision.action,                // 调整动作
                    decision.reason,                // 调整原因
                    alpha_before,                   // 调整前 alpha
                    alpha_after,                    // 调整后 alpha
                    adjustment_applied              // 是否应用
                );
            }
            spdlog::info("DEBUG: Epoch {} completed", epoch_count);
        }
    }
    delete it;

    // ==================== Step 9: 等待后台操作完成 ====================
    spdlog::info("Waiting for background operations to complete...");
    wait_for_compactions(db, compactor);

    auto time_end = std::chrono::high_resolution_clock::now();
    auto total_latency = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    
    // ==================== Step 10: 打印最终统计 ====================
    print_final_statistics(db, rocksdb_opt, compactor, total_latency);

    if (env.enable_dynamic_tuning) // 动态调整相关信息
    {
        spdlog::info("=== Dynamic Tuning Summary ===");
        spdlog::info("Total epochs: {}", epoch_count);
        spdlog::info("Total adjustments: {}", total_adjustments);
        spdlog::info("Final alpha: {:.2f}", decision_engine->get_current_alpha());
        memory_allocator->print_allocation();
    }
    
    // ==================== Step 11: 清理 ====================
    db->Close();
    delete db;
    //delete key_log;
    delete data_gen;

    delete perf_monitor;
    delete decision_engine;
    delete memory_allocator;
    delete epoch_logger;

    spdlog::info("=== Execution Completed ===");
    return EXIT_SUCCESS;
}