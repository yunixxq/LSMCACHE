#include <math.h>
#include <chrono>
#include <iostream>
#include <ctime>
#include <filesystem>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <fstream>

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
using namespace ROCKSDB_NAMESPACE;

#define PAGESIZE 4096

typedef struct environment
{
    std::string db_path;

    double non_empty_reads = 0.25;
    double empty_reads = 0.25;
    double range_reads = 0.25;
    double writes = 0.25;

    // next workload
    double non_empty_reads_next = 0.25;
    double empty_reads_next = 0.25;
    double range_reads_next = 0.25;
    double writes_next = 0.25;
    size_t prime_reads = 0;

    size_t steps = 10;
    int sel = 2;
    int scaling = 1;
    std::string compaction_style = "level";
    // Build mode
    double T = 10;

    size_t M = 1 << 18;
    size_t B = 1 << 18; //> 1 KB
    size_t E = 1 << 7;  //> 128 B
    size_t file_size = 4 * 1024 * 1024; //> 4MB;
    double bits_per_element = 5.0;
    double ratio = 1.0; // xxq's补充 增加Mb与Mc分配比例
    size_t N = 1e6;
    size_t L = 0;

    int verbose = 0;
    bool destroy_db = true;

    int max_rocksdb_levels = 16;
    int parallelism = 1;

    int seed = 0;
    tmpdb::file_size_policy file_size_policy_opt = tmpdb::file_size_policy::INCREASING;
    uint64_t fixed_file_size = std::numeric_limits<uint64_t>::max();

    std::string dist_mode = "zipfian";
    double skew = 0.5;

    size_t cache_cap = 0;
    bool use_cache = true;

    std::string key_log_file;
    bool use_key_log = true;
    bool tuning_T = false;
    bool tuning_h = false;
    bool rocksdb_default_config = false;
    int w = 10000;
    double r = 0.05;

} environment;

// ============================== Step1.命令行参数解析 =====================================
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
                                          (option("-M", "--memory-size") & integer("size", env.M)) % ("total memory size (in bytes), [default: " + to_string(env.M) + "]"), //❌这里后面一开始写的是env.B，显然是错误的
                                          (option("-E", "--entry-size") & integer("size", env.E)) % ("entry size (bytes) [default: " + to_string(env.E) + ", min: 32]"),
                                          (option("-c", "--compaction") & value("mode", env.compaction_style)) % "set level or tier compaction",
                                          (option("-d", "--destroy").set(env.destroy_db)) % "destroy the DB if it exists at the path",
                                          (option("--default-config").set(env.rocksdb_default_config)) % "whether use rocksdb default config"));

    auto run_opt = ("run options:" % ((option("-s", "--steps") & integer("num", env.steps)) % ("steps, [default: " + to_string(env.steps) + "]"),
                                      (option("-w") & integer("num", env.w)) % ("monitor period"),
                                      (option("-r") & number("num", env.r)) % ("threshold for reconfiguration"),
                                      (option("--dist") & value("mode", env.dist_mode)) % ("distribution mode ['uniform', 'zipf']"),
                                      (option("--skew") & number("num", env.skew)) % ("skewness for zipfian [0, 1)"),
                                      (option("--sel") & number("num", env.sel)) % ("selectivity of range query"),
                                      (option("--scaling") & number("num", env.scaling)) % ("scaling"),
                                      (option("--cache").set(env.use_cache, true) & number("cap", env.cache_cap)) % "use block cache",
                                      (option("--key-log-file").set(env.use_key_log, true) & value("file", env.key_log_file)) % "use keylog to record each key",
                                      (option("--tuning-T").set(env.tuning_T)) % "whether tuning T dynamically",
                                      (option("--tuning-h").set(env.tuning_h)) % "whether tuning h dynamically"));

    auto minor_opt = ("minor options:" % ((option("--max_rocksdb_level") & integer("num", env.max_rocksdb_levels)) % ("limits the maximum levels rocksdb has [default: " + to_string(env.max_rocksdb_levels) + "]"),
                                          (option("--parallelism") & integer("num", env.parallelism)) % ("parallelism for writing to db [default: " + to_string(env.parallelism) + "]"),
                                          (option("--seed") & integer("num", env.seed)) % "seed for generating data [default: random from time]"));

    auto file_size_policy_opt =
        ("file size policy (pick one)" %
         one_of(
             (
                 option("--increasing_files").set(env.file_size_policy_opt, tmpdb::file_size_policy::INCREASING) % "file size will match run size as LSM tree grows (default)",
                 (option("--fixed_files").set(env.file_size_policy_opt, tmpdb::file_size_policy::FIXED) & opt_integer("size", env.fixed_file_size)) % "fixed file size specified after fixed_files flag [default size MAX uint64]",
                 option("--buffer_files").set(env.file_size_policy_opt, tmpdb::file_size_policy::BUFFER) % "file size matches the buffer size")));

    auto cli = (general_opt,
                build_opt,
                run_opt,
                minor_opt,
                file_size_policy_opt);

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

void model_serving(std::vector<double> workload, double &T, double &h, double &ratio)
{
    ofstream fout;
    // 给Python的model_server发请求
    fout.open("workloads.in", std::ios::out);
    for (int i = 0; i < 4; ++i)
        fout << workload[i] << " ";
    fout.close();

    // 死循环等待optimal_params.in出现并读出结果
    int ret = -1;
    while (ret < 3)
    {
        auto fparams = freopen("optimal_params.in", "r", stdin);
        if (fparams != nullptr)
        {
            ret = scanf("%lf%lf%lf", &T, &h, &ratio); // 成功读到三个字段跳出循环
            fclose(fparams);
        }
    }
    remove("optimal_params.in"); // 删掉方便后续继续创建该文件
}

void update_param(environment env, std::vector<double> workload, int cur_N,
                  rocksdb::DB *db, tmpdb::Compactor *compactor, rocksdb::FilterPolicy *filter_policy)
{
    double T, h, ratio;
    model_serving(workload, T, h, ratio);

    auto tune_time_start = std::chrono::high_resolution_clock::now();
    if (env.tuning_T && env.T != T)
    {
        spdlog::info("update T: from {} to {}", env.T, T);
        env.T = T;
        compactor->updateT(env.T);
    }

    if (env.tuning_h && env.bits_per_element != h)
    {
        spdlog::info("update h: from {} to {}", env.bits_per_element, h);
        env.bits_per_element = h;
        filter_policy->Update_bpe(env.bits_per_element);//动态更新

        env.ratio = ratio;
        env.B = env.ratio * (env.M - env.bits_per_element * cur_N) / 8;
        env.B = int(env.B);

        rocksdb::Status status1 = db->SetOptions({{"write_buffer_size", std::to_string(env.B/2)}});
        if (!status1.ok())
        {
            printf("Set write_buffer_size fail: code=%d\n", status1.code());
        }
        compactor->updateM(env.B);// 动态更新，后续使用新的B进行判断
    }

    // 似乎没必要，但是判断一次也无妨
    while (compactor->requires_compaction(db))
    {
        while (compactor->compactions_left_count > 0)
            ;
    }
    auto tune_time_end = std::chrono::high_resolution_clock::now();
    auto latency_t = std::chrono::duration_cast<std::chrono::milliseconds>(tune_time_end - tune_time_start).count();
    spdlog::info("tunning time: {}", latency_t);
}

int main(int argc, char *argv[])
{
    // ============================== Step2.初始化RocksDB =====================================
    spdlog::set_pattern("[%T.%e]%^[%l]%$ %v");
    environment env = parse_args(argc, argv); //解析命令行

    std::vector<std::vector<double>> workloads;
    auto fworkloads = freopen("test_workloads.in", "r", stdin);
    if (fworkloads == nullptr)
        return 0;

    double z0, z1, q, w;
    while (~scanf("%lf %lf %lf %lf\n", &z0, &z1, &q, &w))
        workloads.push_back({z0, z1, q, w});

    fclose(fworkloads);

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

    // false表示不使用默认参数配置
    if (env.rocksdb_default_config)
    {
        env.T = 10.0;
        env.bits_per_element = 10.0;
    }
    else
    {
        model_serving(workloads[0], env.T, env.bits_per_element, env.ratio);
    }

    env.B = env.ratio * (env.M - env.bits_per_element * env.N) / 8 ;
    env.B = int(env.B);

    env.cache_cap = (1 - env.ratio) * (env.M - env.bits_per_element * env.N) / 8;
    env.cache_cap = int(env.cache_cap);

    spdlog::info("env.T: {}, env.bits_per_element: {}, env.B: {}, cache_cap: {}", env.T, env.bits_per_element, env.B, env.cache_cap);

    spdlog::info("Building DB: {}", env.db_path);

    // =========== RocksDB Options 基础配置 =========== 
    rocksdb::Options rocksdb_opt;

    rocksdb_opt.create_if_missing = true;
    rocksdb_opt.error_if_exists = true;
    //rocksdb_opt.IncreaseParallelism(env.parallelism); // db_runner=32
    rocksdb_opt.compression = rocksdb::kNoCompression;
    rocksdb_opt.bottommost_compression = kNoCompression;
    rocksdb_opt.use_direct_reads = true;
    rocksdb_opt.use_direct_io_for_flush_and_compaction = true;
    rocksdb_opt.max_open_files = 512;
    rocksdb_opt.advise_random_on_open = false;
    rocksdb_opt.random_access_max_buffer_size = 0;
    rocksdb_opt.avoid_unnecessary_blocking_io = true;
    rocksdb_opt.target_file_size_base = env.scaling * env.file_size;
    rocksdb_opt.compaction_style = rocksdb::kCompactionStyleNone;
    rocksdb_opt.disable_auto_compactions = true;
    rocksdb_opt.write_buffer_size = env.B / 2;

    // 输出默认的max_write_buffer_number检查是否为2
    std::cout << rocksdb_opt.max_write_buffer_number << std::endl;

    // 自定义压缩参数
    tmpdb::Compactor *compactor = nullptr;
    tmpdb::CompactorOptions compactor_opt;
    compactor_opt.size_ratio = env.T;
    compactor_opt.buffer_size = env.B;
    compactor_opt.entry_size = env.E;
    compactor_opt.bits_per_element = env.bits_per_element;
    compactor_opt.num_entries = env.N;
    compactor_opt.levels = tmpdb::Compactor::estimate_levels(env.N, env.T, env.E, env.B) + 1;
    rocksdb_opt.num_levels = compactor_opt.levels + 1;
    compactor = new tmpdb::Compactor(compactor_opt, rocksdb_opt);
    rocksdb_opt.listeners.emplace_back(compactor);

    rocksdb::BlockBasedTableOptions table_options;

    // 配置Monkey Bloom filter(在不同的level调整bpe)
    rocksdb::FilterPolicy *filter_policy =
        rocksdb::NewMonkeyFilterPolicy(
            env.bits_per_element,
            compactor_opt.size_ratio,
            compactor_opt.levels);

    table_options.filter_policy.reset(filter_policy);      

    // 配置Block Cache
    if (env.cache_cap == 0)
        table_options.no_block_cache = true;
    else
        table_options.block_cache = rocksdb::NewLRUCache(env.cache_cap);
    rocksdb_opt.table_factory.reset(
        rocksdb::NewBlockBasedTableFactory(table_options));

    rocksdb_opt.statistics = rocksdb::CreateDBStatistics();

    // 打开RocksDB
    rocksdb::DB *db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(rocksdb_opt, env.db_path, &db);
    if (!status.ok())
    {
        spdlog::error("Problems opening DB");
        spdlog::error("{}", status.ToString());
        delete db;
        exit(EXIT_FAILURE);
    }

    // ============================== Step3.初始化数据 =====================================
    std::map<std::string, uint64_t> stats;
    uint64_t num_running_flushes, num_pending_flushes;
    KeyLog *key_log = new KeyLog(env.key_log_file);
    rocksdb::WriteOptions write_opt;
    write_opt.low_pri = true;
    write_opt.disableWAL = true;

    // 插入env.N个KV对
    DataGenerator *data_gen = new YCSBGenerator(env.N, "uniform", 0.0);
    std::pair<std::string, std::string> key_value;
    auto write_time_start = std::chrono::high_resolution_clock::now();

    for (size_t entry_num = 0; entry_num < env.N; entry_num += 1)
    {
        key_value = data_gen->gen_kv_pair(env.E);
        db->Put(write_opt, key_value.first, key_value.second);
    }

    // 等待所有的flush/Compaction完成-确保数据库在进入查询前处于稳定状态
    spdlog::info("Waiting for all compactions to finish before running");
    rocksdb::FlushOptions flush_opt;
    {
        while (true)
        {
            // 正在进行flush的Memtable数量
            db->GetIntProperty(DB::Properties::kNumRunningFlushes,
                               &num_running_flushes);
            // 标记为待刷写的Memtable数量
            db->GetIntProperty(DB::Properties::kMemTableFlushPending,
                               &num_pending_flushes);
            if (num_running_flushes == 0 && num_pending_flushes == 0)
                break;
        }
        while (compactor->compactions_left_count > 0) //检查是否有尚未完成的Compaction
            ;
    }
    while (compactor->requires_compaction(db)) //检查当前是否需要执行Compaction操作
    {
        while (compactor->compactions_left_count > 0)
            ;
    }
    auto write_time_end = std::chrono::high_resolution_clock::now();
    auto write_time = std::chrono::duration_cast<std::chrono::milliseconds>(write_time_end - write_time_start).count();
    spdlog::info("(init_time) : ({})", write_time);

    // 获取每层文件分布
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::string run_per_level = "[";
    for (auto &level : cf_meta.levels)
    {
        run_per_level += std::to_string(level.files.size()) + ", ";
    }
    run_per_level = run_per_level.substr(0, run_per_level.size() - 2) + "]";
    spdlog::info("files_per_level_build : {}", run_per_level);

    std::string size_per_level = "[";
    for (auto &level : cf_meta.levels)
    {
        size_per_level += std::to_string(level.size) + ", ";
    }
    size_per_level = size_per_level.substr(0, size_per_level.size() - 2) + "]";
    spdlog::info("size_per_level_build : {}", size_per_level);

    // 重置统计计数器
    rocksdb_opt.statistics->Reset();
    rocksdb::get_iostats_context()->Reset();
    rocksdb::get_perf_context()->Reset();

    std::mt19937 engine; //1️⃣
    std::uniform_real_distribution<double> dist(0, 1); // 2️⃣

    // ============================== Step4.执行查询(workload执行逻辑) =====================================
    std::string value, key, limit; //limit是范围查询的终止key
    data_gen = new YCSBGenerator(env.N, env.dist_mode, env.skew);
    rocksdb::Iterator *it = db->NewIterator(rocksdb::ReadOptions());
    env.sel = PAGESIZE / env.E;

    size_t cur_N = env.N;
    size_t num_workloads = workloads.size();
    std::vector<int64_t> latency; // 存储每一段workload的执行时间
    bool tuning = (env.tuning_T || env.tuning_h);

    auto cur_workload = workloads[0];
    std::vector<int> freq = {0, 0, 0, 0}; // 统计过去env.w次操作中4类操作频率
    int num_ops = 0;

    for (size_t k = 0; k < num_workloads - 1; ++k)
    {
        auto time_start = std::chrono::high_resolution_clock::now();
        auto workload = workloads[k];
        // 线性插值
        std::vector<double> stride;
        for (int j = 0; j < 4; ++j)
            stride.push_back((workloads[k + 1][j] - workloads[k][j]) / env.steps);

        for (size_t i = 0; i < env.steps; i++)
        {
            double r = dist(engine); //3️⃣生成一个0-1的随机数
            int outcome = 0;
            double cumprob = 0;
            for (int j = 0; j < 4; j++)
            {
                cumprob += workload[j];
                if (r < cumprob)
                {
                    outcome = j;
                    break;
                }
            }
            for (int j = 0; j < 4; ++j) // 模拟工作负载之间的平滑过渡
                workload[j] += stride[j];

            // 动态调参触发逻辑
            if (tuning)
            {
                freq[outcome] += 1;
                num_ops += 1;
                if (num_ops == env.w)
                {
                    // delta用于统计四种操作类型“增长率”
                    double delta = 0, ratio;
                    for (int j = 0; j < 4; ++j)
                    {
                        ratio = 1.0 * freq[j] / env.w;
                        if (ratio > cur_workload[j])
                            delta += ratio - cur_workload[j];
                    }
                    if (delta > env.r)
                    {
                        for (int j = 0; j < 4; ++j)
                            cur_workload[j] = 1.0 * freq[j] / env.w; // 更新工作负载

                        update_param(env, cur_workload, cur_N, db, compactor, filter_policy);
                    }
                    
                    // 复位，重新统计
                    num_ops = 0;
                    for (int j = 0; j < 4; ++j)
                        freq[j] = 0;
                }
            }

            switch (outcome)
            {
            case 0:
            {
                key = data_gen->gen_new_dup_key();
                key_log->log_key(key);
                status = db->Get(rocksdb::ReadOptions(), key, &value);
                break;
            }
            case 1:
            {
                key = data_gen->gen_existing_key();
                key_log->log_key(key);
                status = db->Get(rocksdb::ReadOptions(), key, &value);
                break;
            }
            case 2:
            {
                key = data_gen->gen_existing_key();
                key_log->log_key(key);
                limit = std::to_string(stoi(key) + 1 + env.sel);
                // 范围对应 [start, limit）实际上实现的是一个顺序扫描
                for (it->Seek(rocksdb::Slice(key)); it->Valid() && it->key().ToString() < limit; it->Next())
                {
                    value = it->value().ToString();
                }
                break;
            }
            case 3:
            {
                key_value = data_gen->gen_new_kv_pair(compactor_opt.entry_size);
                key_log->log_key(key_value.first);
                db->Put(write_opt, key_value.first, key_value.second);
                ++cur_N;
                break;
            }
            default:
                break;
            }
        }

        // 确保所有的后台操作(flush + compaction)完成
        while (true)
        {
            db->GetIntProperty(DB::Properties::kNumRunningFlushes,
                               &num_running_flushes);
            db->GetIntProperty(DB::Properties::kMemTableFlushPending,
                               &num_pending_flushes);
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
        auto time_end = std::chrono::high_resolution_clock::now();
        auto latency_t = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        latency.push_back(latency_t);
    }

    std::string latency_per_workload = "[";
    for (auto &lt : latency)
    {
        latency_per_workload += std::to_string(lt) + ", ";
    }
    latency_per_workload = latency_per_workload.substr(0, latency_per_workload.size() - 2) + "]";
    spdlog::info("latency_per_workload : {}", latency_per_workload);

    delete it;
    db->Close();
    delete db;
    delete key_log;
    delete data_gen;
    return EXIT_SUCCESS;
}