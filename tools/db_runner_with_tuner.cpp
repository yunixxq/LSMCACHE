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
#include "rocksdb/utilities/sim_cache.h"

// 自定义组件
#include "tmpdb/compactor.hpp"
#include "infrastructure/data_generator.hpp"
#include "tmpdb/memory_tuner.hpp"  // ✅ 添加Memory Tuner

using namespace ROCKSDB_NAMESPACE;

#define PAGESIZE 4096

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
    double T = 10;
    double K = 0;
    size_t E = 1 << 7;
    double bpe = 5.0;
    size_t N = 1e6;
    size_t L = 0;
    size_t M = 0; // 总内存预算
    size_t initial_write_memory = 64 * 1024 * 1024; // 初始写内存大小
    
    // ===== Memory Tuner配置（论文Breaking Walls） =====
    bool enable_memory_tuner = false;   // 是否启用Memory Tuner
    double write_weight = 1.0;          // ω: 写成本权重 默认均为1
    double read_weight = 1.0;           // γ: 读成本权重
    size_t sim_cache_size = 128 * 1024 * 1024;  // SimCache大小 128M 单位:bytes
    size_t tuning_interval_seconds = 600;       // 调优间隔（秒）
    size_t min_tuning_interval_seconds = 60;    // 最小调优间隔
              
    // 其他配置
    int verbose = 0;
    bool destroy_db = true;
    int max_rocksdb_levels = 64;
    int parallelism = 1;
    int seed = 0;
    std::string dist_mode = "zipfian";
    double skew = 0.5;
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

    // 计算总的I/O成本，单位: KB/op (与Breaking Walls原文中的figure15对应)
    // 写I/O统计 Bytes
    uint64_t flush_write_bytes = stats["rocksdb.flush.write.bytes"];
    uint64_t compact_write_bytes = stats["rocksdb.compact.write.bytes"];
    uint64_t total_write_bytes = flush_write_bytes + compact_write_bytes;

    // 读I/O统计 Bytes (后台Compaction + 户线程的磁盘读取)
    uint64_t merge_read_bytes = stats["rocksdb.compact.read.bytes"];
    auto perf_ctx = rocksdb::get_perf_context();
    uint64_t query_read_bytes = perf_ctx->block_read_byte;
    uint64_t total_read_bytes = merge_read_bytes + query_read_bytes;
    
    // 转换为标准单位
    double write_cost_kb_per_op = 0.0;
    double read_cost_kb_per_op = 0.0;

    write_cost_kb_per_op = static_cast<double>(total_write_bytes) / (total_operations * 1024.0);
    read_cost_kb_per_op = static_cast<double>(total_read_bytes) / (total_operations * 1024.0);

    double total_io_cost_kb_per_op = write_cost_kb_per_op + read_cost_kb_per_op;
    spdlog::info("I/O Cost (KB/op):");
    spdlog::info("(write_io_kb_per_op, read_io_kb_per_op, total_io_kb_per_op) : ({:.4f}, {:.4f}, {:.4f})",
             write_cost_kb_per_op, read_cost_kb_per_op, total_io_cost_kb_per_op);
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

    // 写日志长度: rocksdb_opt.max_log_file_size

    // ===== 设置初始write buffer大小 =====
    size_t initial_write_memory = env.initial_write_memory;  // 默认使用-B参数
    if (env.M > 0 && env.enable_memory_tuner) {
        // 使用总内存和alpha计算初始write memory
        initial_write_memory = env.initial_write_memory;
    }
    rocksdb_opt.write_buffer_size = initial_write_memory; // ✅ 实际配置写内存参数
    spdlog::info("Initial write buffer size: {} MB", initial_write_memory / (1024 * 1024));

    // ==================== Step 3: 配置自定义 Compactor ====================
    tmpdb::Compactor *compactor = nullptr;
    tmpdb::CompactorOptions compactor_opt;
    
    compactor_opt.size_ratio = env.T;
    compactor_opt.buffer_size = initial_write_memory;
    compactor_opt.entry_size = env.E;
    compactor_opt.bits_per_element = env.bpe;
    compactor_opt.num_entries = env.N;

    if (env.compaction_style == "level")
        compactor_opt.K = 1;
    else if (env.compaction_style == "tier")
        compactor_opt.K = env.T;
    else
        compactor_opt.K = env.K;

    compactor_opt.levels = tmpdb::Compactor::estimate_levels(env.N, env.T, env.E, initial_write_memory) 
                           * compactor_opt.K + 1;
    rocksdb_opt.num_levels = compactor_opt.levels + 1;

    compactor = new tmpdb::Compactor(compactor_opt, rocksdb_opt);
    rocksdb_opt.listeners.emplace_back(compactor);

    // ==================== Step 4: 配置 Block Cache ====================
    rocksdb::BlockBasedTableOptions table_options;
    table_options.read_amp_bytes_per_bit = 32;

    table_options.filter_policy.reset(
        rocksdb::NewMonkeyFilterPolicy(
            env.bpe,
            compactor_opt.size_ratio,
            compactor_opt.levels));
    
    // ✅ env.sim_cache_size 指的是额外增加的块缓存大小
    std::shared_ptr<Cache> block_cache = nullptr;
    std::shared_ptr<rocksdb::SimCache> sim_cache = nullptr;  // ✅ 模拟缓存
    
    // ✅ 初始块缓存设置
    // 4GB = 4 * 1024 MB = 4096MB 20GB = 20 * 1024MB = 20480MB
    // 初始块缓存 = 4096 - 64 = 4032MB / 20480 - 64 = 20416MB
    size_t initial_cache_size = env.M - env.initial_write_memory;
    if (env.M > 0 && env.enable_memory_tuner) {
        // 使用总内存和alpha计算初始cache大小
        initial_cache_size = env.M - initial_write_memory; // ✅ 计算初始分配的块缓存大小
    }
    
    if (initial_cache_size == 0) {
        table_options.no_block_cache = true;
    } else {
        block_cache = rocksdb::NewLRUCache(initial_cache_size);
        
        // ✅ SimCache容量 = 实际缓存 + 128MB，模拟"如果缓存再大128MB"
        if (env.enable_memory_tuner && env.sim_cache_size > 0) {
            size_t sim_total_size = initial_cache_size + env.sim_cache_size;
            sim_cache = rocksdb::NewSimCache(block_cache, sim_total_size, -1);
            table_options.block_cache = sim_cache;
            spdlog::info("Block cache: {} MB, SimCache total: {} MB (= cache + {} MB)",
                         initial_cache_size / (1024 * 1024),
                         sim_total_size / (1024 * 1024),
                         env.sim_cache_size / (1024 * 1024));
        } else {
            table_options.block_cache = block_cache;
            spdlog::info("Initial block cache size: {} MB", initial_cache_size / (1024 * 1024));
        }
    }
    
    rocksdb_opt.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    rocksdb_opt.statistics = rocksdb::CreateDBStatistics();

    // ==================== Step 5: 打开数据库 ====================
    rocksdb::DB *db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(rocksdb_opt, env.db_path, &db);
    if (!status.ok())
    {
        spdlog::error("Problems opening DB: {}", status.ToString());
        delete db;
        exit(EXIT_FAILURE);
    }

    // ==================== Step 6: 初始化Memory Tuner ====================
    tmpdb::MemoryTuner* memory_tuner = nullptr;

    if (env.enable_memory_tuner)
    {
        // 验证必要条件
        if (env.M == 0) {
            spdlog::error("Total memory budget (-M) must be specified for memory tuning");
            exit(EXIT_FAILURE);
        }
        if (block_cache == nullptr) {
            spdlog::error("Block cache must be enabled for memory tuning");
            exit(EXIT_FAILURE);
        }

        spdlog::info("=== Memory Tuner Enabled (Breaking Walls Paper) ===");
        
        // 配置Memory Tuner
        tmpdb::MemoryTunerConfig tuner_config;
        tuner_config.total_memory = env.M; // ✅ 外部传参 总内存大小设置(4GB/16GB)
        tuner_config.initial_write_memory = env.initial_write_memory; // ✅ 默认的写内存初始值
        tuner_config.sim_cache_size = env.sim_cache_size; // ✅ 默认128MB 指的是在当前Block Cache基础上补充的容量
        tuner_config.write_weight = env.write_weight;
        tuner_config.read_weight = env.read_weight;
        tuner_config.tuning_interval_seconds = env.tuning_interval_seconds;
        tuner_config.min_tuning_interval_seconds = env.min_tuning_interval_seconds;
        tuner_config.K = 3;
        tuner_config.max_step_ratio = 0.10;
        tuner_config.min_step_size = 32 * 1024 * 1024;
        tuner_config.min_cost_reduction_ratio = 0.001;
        tuner_config.page_size = PAGESIZE;
        tuner_config.size_ratio = env.T;
        
        // 边界配置
        tuner_config.min_write_memory = 64 * 1024 * 1024;   // 最小64MB
        tuner_config.min_buffer_cache = 64 * 1024 * 1024;   // 最小64MB
        
        // ✅ 创建Memory Tuner
        memory_tuner = new tmpdb::MemoryTuner(
            db,
            block_cache,
            sim_cache,
            rocksdb_opt.statistics,
            compactor, 
            tuner_config
        );

        // 建立反向连接，将tuner再传递给compactor二者互相进行调用
        compactor->set_memory_tuner(memory_tuner);
        
        spdlog::info("Memory Tuner Configuration:");
        spdlog::info("  Total memory: {} MB", env.M / (1024 * 1024));
        spdlog::info("  Write weight (ω): {:.2f}", env.write_weight);
        spdlog::info("  Read weight (γ): {:.2f}", env.read_weight);
        spdlog::info("  SimCache size: {} MB", env.sim_cache_size / (1024 * 1024));
    }

    // ==================== Step 7: 初始化数据 ====================
    spdlog::info("Initializing data with {} entries...", env.N);

    rocksdb::WriteOptions write_opt;
    write_opt.low_pri = true;
    write_opt.disableWAL = false; // ✅ 必须要开启写日志，因为其作为写成本导数系数出现

    DataGenerator *data_gen = new YCSBGenerator(env.N, "uniform", 0.0);
    std::pair<std::string, std::string> key_value;

    auto write_time_start = std::chrono::high_resolution_clock::now();
    for (size_t entry_num = 0; entry_num < env.N; entry_num += 1)
    {
        key_value = data_gen->gen_kv_pair(env.E);
        db->Put(write_opt, key_value.first, key_value.second);
    }

    spdlog::info("Waiting for initial compactions to finish...");
    wait_for_compactions(db, compactor);

    auto write_time_end = std::chrono::high_resolution_clock::now();
    auto write_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        write_time_end - write_time_start).count();
    spdlog::info("(init_time) : ({})", write_time);

    print_db_status(db);

    // 重置统计信息
    rocksdb_opt.statistics->Reset();
    rocksdb::get_iostats_context()->Reset();
    rocksdb::get_perf_context()->Reset();
    compactor->stats.reset_epoch();
    
    // 重置Memory Tuner统计
    if (memory_tuner) {
        memory_tuner->reset_statistics();
    }

    // ==================== Step 8: 执行 Workload ====================
    spdlog::info("=== Starting Workload Execution ===");
    spdlog::info("Total queries: {}", env.queries);

    double p[] = {env.empty_reads, env.non_empty_reads, env.range_reads, env.writes};
    double cumprob[] = {p[0], p[0] + p[1], p[0] + p[1] + p[2], 1.0};
    
    std::string value, key, limit;
    delete data_gen;
    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);

    ReadOptions read_options;
    read_options.total_order_seek = true;
    rocksdb::Iterator *it = db->NewIterator(read_options);

    std::mt19937 engine;
    if (env.seed != 0) {
        engine.seed(env.seed);
    } else {
        engine.seed(std::time(nullptr));
    }
    std::uniform_real_distribution<double> dist(0, 1);

    env.sel = PAGESIZE / env.E;

    // 统计变量
    size_t tuning_count = 0;
    size_t total_adjustments = 0;

    auto time_start = std::chrono::high_resolution_clock::now();
    
    // ==================== 主循环 ====================
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
            case 1:  // Non-empty read
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
            case 3:  // Write
            {
                key_value = data_gen->gen_existing_kv_pair(env.E);
                db->Put(write_opt, key_value.first, key_value.second);
                break;
            }
            default:
                break;
        }
        
        if (memory_tuner) {
            memory_tuner->record_operation(1);
        }
        
        // ===== Memory Tuner调优检查 =====
        if (memory_tuner && memory_tuner->should_tune())
        {
            tuning_count++;
            spdlog::info("--- Memory Tuning Cycle {} (step {}/{}) ---", 
                         tuning_count, i + 1, env.queries);
            
            // 执行调优
            bool adjusted = memory_tuner->tune();
            
            if (adjusted) {
                total_adjustments++;
                spdlog::info("Memory allocation adjusted");
                spdlog::info("  Write memory: {} MB", 
                             memory_tuner->get_write_memory_size() / (1024 * 1024));
                spdlog::info("  Buffer cache: {} MB", 
                             memory_tuner->get_buffer_cache_size() / (1024 * 1024));
                
                // 更新Compactor的buffer_size配置
                compactor->updateM(memory_tuner->get_write_memory_size());
            }
        }
    }
    delete it;

    // ==================== Step 9: 等待后台操作完成 ====================
    spdlog::info("Waiting for background operations to complete...");
    wait_for_compactions(db, compactor);

    auto time_end = std::chrono::high_resolution_clock::now();
    auto total_latency = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    
    // ==================== Step 10: 打印最终统计 ====================
    print_final_statistics(db, rocksdb_opt, compactor, total_latency, env.queries);

    // Memory Tuner统计
    if (memory_tuner)
    {
        spdlog::info("=== Memory Tuner Summary ===");
        spdlog::info("Total tuning cycles: {}", tuning_count);
        spdlog::info("Total adjustments: {}", total_adjustments);
        spdlog::info("Final write memory: {} MB", 
                     memory_tuner->get_write_memory_size() / (1024 * 1024));
        spdlog::info("Final buffer cache: {} MB", 
                     memory_tuner->get_buffer_cache_size() / (1024 * 1024));
        memory_tuner->print_status();
    }
    
    // ==================== Step 11: 清理 ====================
    db->Close();
    delete db;
    delete data_gen;
    delete memory_tuner;

    spdlog::info("=== Execution Completed ===");
    return EXIT_SUCCESS;
}
