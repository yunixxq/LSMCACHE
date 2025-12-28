#ifndef TMPDB_MEMORY_TUNER_HPP
#define TMPDB_MEMORY_TUNER_HPP

// Memory Tuner实现 - 复现VLDB'21论文
// "Breaking Down Memory Walls: Adaptive Memory Management in LSM-based Storage Systems"
// 核心思想：通过在线梯度下降方法，自适应调整write memory和buffer cache之间的内存分配，最小化总体I/O成本。

#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <cmath>
#include <memory>
#include <chrono>
#include <string>
#include <unordered_map>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"
#include "rocksdb/utilities/sim_cache.h"
#include "rocksdb/cache.h"
#include "spdlog/spdlog.h"
#include "tmpdb/compactor.hpp"

namespace tmpdb {

class Compactor;

// ✅ 一个调优周期内收集的统计信息
struct TuningCycleStats {
    // 基本统计
    size_t operations = 0;                 // op: 观察到的操作数
    size_t total_writes = 0;               // 总写入操作数
    size_t total_reads = 0;                // 总读取操作数
    size_t total_range_queries = 0;        // 总范围查询数
    
    // I/O统计(pages)
    size_t total_merge_writes = 0;         // 总merge写入页数
    size_t total_merge_reads = 0;          // 总merge读取页数
    size_t total_query_reads = 0;          // 查询读取页数
    size_t total_flush_writes = 0;         // flush写入页数
    
    // SimCache统计
    double saved_query_reads = 0.0;
    double saved_merge_reads = 0.0;
    
    // Cache统计
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    double cache_hit_rate = 0.0;
    
    // Flush统计
    size_t log_triggered_flushes = 0;
    size_t memory_triggered_flushes = 0;
    
    // 时间戳
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    
    void reset() {
        operations = 0;
        total_writes = 0;
        total_reads = 0;
        total_range_queries = 0;
        total_merge_writes = 0;
        total_merge_reads = 0;
        total_query_reads = 0;
        total_flush_writes = 0;
        saved_query_reads = 0.0;
        saved_merge_reads = 0.0;
        cache_hits = 0;
        cache_misses = 0;
        cache_hit_rate = 0.0;
        log_triggered_flushes = 0;
        memory_triggered_flushes = 0;
        start_time = std::chrono::steady_clock::now();
    }
};

// 记录一个调优点的信息，用于Newton-Raphson方法的线性拟合
struct TuningPoint {
    size_t write_memory_size;              // x: write memory大小
    double cost_derivative;                // cost'(x): 成本函数的导数
    double io_cost;                        // cost(x): 当前的I/O成本
    double write_cost;                     // write(x): 写成本
    double read_cost;                      // read(x): 读成本
};

// Memory Tuner的配置参数 
// 1️⃣初始写内存 64MB 2️⃣模拟缓存 128MB 3️⃣NewtonâĂŞRaphson k 默认值 3
struct MemoryTunerConfig {
    // 内存边界
    size_t total_memory = 0; // M: 总内存预算
    size_t initial_write_memory = 64 * 1024 * 1024;  // ✅ 初始写内存大小(默认64MB 6.3.1)
    size_t min_write_memory = 64 * 1024 * 1024;
    size_t min_buffer_cache = 64 * 1024 * 1024;
    
    // SimCache配置
    size_t sim_cache_size = 128 * 1024 * 1024;     // 扩展的模拟cache大小(默认128MB 6.3.1)
    
    // 权重配置
    double write_weight = 1.0;             // ω: 写成本权重
    double read_weight = 1.0;              // γ: 读成本权重
    
    // Newton-Raphson参数 & 三个启发式函数
    size_t K = 3;        // K: 用于线性拟合的样本数
    double max_step_ratio = 0.10;          // 最大步长比例(限制最大调整量)
    size_t min_step_size = 32 * 1024 * 1024;  // 最小步长(停止条件1) 32MB
    double min_cost_reduction_ratio = 0.001;   // 最小成本降低比例(停止条件2) 0.1%
    size_t fixed_step_size = 0;            // 固定步长(回退)
    
    // 调优周期配置 
    size_t min_tuning_interval_seconds = 60; // 最小的调优时间间隔
    size_t tuning_interval_seconds = 600;  // 基于时间的调优间隔（10分钟）
    
    // LSM参数
    size_t page_size = 4096;               // ✅ P: 页大小，相当于data block大小，默认为4KB = 4 * 1024 B
    double size_ratio = 10.0;              // T: size ratio
    
    // 调试开关
    bool verbose = false;
};

// ✅ 内存调优器 - 自适应调整write memory和buffer cache的分配
class MemoryTuner {
public:
    /**
     * @brief 构造函数
     * @param db RocksDB实例指针
     * @param block_cache Block cache指针
     * @param sim_cache 模拟块缓存指针
     * @param statistics RocksDB统计信息指针
     * @param config 配置参数
     */
    MemoryTuner(rocksdb::DB* db,
                std::shared_ptr<rocksdb::Cache> block_cache,
                std::shared_ptr<rocksdb::SimCache> sim_cache,
                std::shared_ptr<rocksdb::Statistics> statistics,
                Compactor* compactor,
                const MemoryTunerConfig& config);
    
    ~MemoryTuner();

    bool tune();
    
    void notify_log_triggered_flush() {
        log_flush_occurred_.store(true);
    }

    bool should_tune();
    
    
    size_t get_write_memory_size() const { return current_write_memory_; }
    
    size_t get_buffer_cache_size() const { 
        return config_.total_memory - current_write_memory_; 
    }

    // ✅ 新增：获取初始写内存大小(通常不需要使用)
    size_t get_initial_write_memory() const {
        return config_.initial_write_memory;
    }

    void reset_statistics();
    
    void print_status() const;
    
    // ✅ 计算一个调优周期内的操作数
    void record_operation(size_t count = 1) {
        current_stats_.operations += count;
    }

private:
    // 总成本导数: cost'(x) = ω·write'(x) + γ·read'(x)
    // 写成本导数(eq5): write'(x) = Σ [-merge_i(x) / (x·ln(|L_Ni|/(a_i·x)))] · [flushmem_i / (flushmem_i + flushlog_i)]
    double estimate_write_cost_derivative(size_t write_memory);
    
    // 读成本导数(eq6): read'(x) = (saved_q + saved_m)/sim + write'(x)·read_m(x)/merge(x)
    double estimate_read_cost_derivative(size_t write_memory, double write_derivative);
    
    // Newton-Raphson调优(返回值是建议的x值)
    size_t newton_raphson_step(size_t current_x, double current_derivative);
    
    // 使用线性拟合估算成本导数函数
    // 用最近K个样本拟合 cost'(x) = Ax + B
    bool fit_linear_model(double& A, double& B);
    
    // 应用启发式规则确保稳定性
    size_t apply_stability_heuristics(size_t proposed_x, size_t current_x);
    
    // 检查是否满足停止条件(1️⃣步长过小2️⃣预期成本降低太小)
    bool should_stop_tuning(size_t step_size, double expected_reduction);
    
    // 应用新的内存分配
    bool apply_memory_allocation(size_t new_write_memory);
    
    // 调整block cache大小
    void resize_block_cache(size_t new_size);
    
    // 调整write buffer大小
    void resize_write_buffer(size_t new_size);
    
    // 收集统计信息
    void collect_from_statistics();

    // 初始化上一个epoch结束状态时的信息
    void init_prev_values();
     
private:
    // RocksDB组件
    rocksdb::DB* db_;
    std::shared_ptr<rocksdb::Cache> block_cache_;
    std::shared_ptr<rocksdb::Statistics> statistics_;
    std::shared_ptr<rocksdb::SimCache> sim_cache_;
    Compactor* compactor_;
    std::atomic<bool> log_flush_occurred_{false};
    
    // 配置
    MemoryTunerConfig config_;
    
    // 当前内存分配
    size_t current_write_memory_;
    
    // 统计信息
    TuningCycleStats current_stats_;
    
    // 调优历史(用于Newton-Raphson)
    std::deque<TuningPoint> tuning_history_;
    
    // 调优周期管理
    size_t last_tuning_ops_;
    std::chrono::steady_clock::time_point last_tuning_time_;
    size_t tuning_count_;
    
    // ✅ 上一周期的统计快照(用于计算增量) 包含 Statistics + PerfContext
    std::unordered_map<std::string, uint64_t> prev_ticker_values_;
    
    // 线程安全
    mutable std::mutex mutex_;
    
    // 调优状态
    bool is_tuning_enabled_;
    bool has_converged_;
};


// 计算LSM-tree的层数
inline size_t estimate_lsm_levels(size_t data_size, size_t write_memory, double size_ratio) {
    if (data_size <= write_memory) return 1;
    return static_cast<size_t>(std::ceil(std::log(static_cast<double>(data_size) / write_memory) 
                                          / std::log(size_ratio)));
}


} // namespace tmpdb

#endif // TMPDB_MEMORY_TUNER_HPP