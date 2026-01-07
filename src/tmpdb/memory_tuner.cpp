/**
 * @file memory_tuner.cpp
 * @brief Memory Tuner实现 - 复现VLDB'21论文
 *        "Breaking Down Memory Walls: Adaptive Memory Management in LSM-based Storage Systems"
 */

#include "memory_tuner.hpp"
#include "tmpdb/compactor.hpp"
#include "rocksdb/perf_context.h"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace tmpdb {

// ✅ 构造函数定义
MemoryTuner::MemoryTuner(
    rocksdb::DB* db,
    std::shared_ptr<rocksdb::Cache> block_cache,
    std::shared_ptr<rocksdb::SimCache> sim_cache,
    std::shared_ptr<rocksdb::Statistics> statistics,
    Compactor* compactor,
    const MemoryTunerConfig& config)
    : db_(db)
    , block_cache_(block_cache)
    , statistics_(statistics)
    , sim_cache_(sim_cache)
    , compactor_(compactor)
    , config_(config)
    , current_write_memory_(config.initial_write_memory)
    , last_tuning_ops_(0)
    , tuning_count_(0)
    , is_tuning_enabled_(true)
    , has_converged_(false)
{
    // 验证配置
    if (config_.total_memory == 0) {
        spdlog::error("MemoryTuner: total_memory must be specified");
        is_tuning_enabled_ = false;
        return;
    }
    
    // 在调用这个构造函数时，config已经赋值了，即我们已经拥有一些配置
    if (config_.fixed_step_size == 0) {
        config_.fixed_step_size = config_.total_memory / 20;  // 5% of total memory
    }

    // 初始化统计信息
    current_stats_.reset(); // 作为epoch起点，设置其start_time
    
    // // 初始化时间戳
    // last_tuning_time_ = std::chrono::steady_clock::now();

    // 初始化所有快照信息
    init_prev_values();
    
    spdlog::info("MemoryTuner initialized:");
    spdlog::info("  Total memory: {} MB", config_.total_memory / (1024 * 1024));
    spdlog::info("  Initial write memory: {} MB", current_write_memory_ / (1024 * 1024));
    spdlog::info("  Initial buffer cache: {} MB", get_buffer_cache_size() / (1024 * 1024));
    spdlog::info("  SimCache size: {} MB", config_.sim_cache_size / (1024 * 1024));
    spdlog::info("  Write weight (ω): {:.2f}", config_.write_weight);
    spdlog::info("  Read weight (γ): {:.2f}", config_.read_weight);
}

MemoryTuner::~MemoryTuner() {
    spdlog::info("MemoryTuner destroyed. Total tuning cycles: {}", tuning_count_);
}

// ✅新增：初始化快照的辅助函数
void MemoryTuner::init_prev_values() {
    prev_ticker_values_.clear();
    
    // Statistics 快照
    if (statistics_) {
        std::vector<uint32_t> tickers = {
            rocksdb::BLOCK_CACHE_HIT,
            rocksdb::BLOCK_CACHE_MISS,
            rocksdb::SIM_BLOCK_CACHE_HIT,
            rocksdb::SIM_BLOCK_CACHE_MISS,
            rocksdb::BLOCK_CACHE_DATA_MISS,
            rocksdb::BLOCK_CACHE_INDEX_MISS,
            rocksdb::BLOCK_CACHE_FILTER_MISS,
            rocksdb::BLOOM_FILTER_USEFUL,
            rocksdb::BYTES_WRITTEN,
            rocksdb::BYTES_READ,
            rocksdb::COMPACT_READ_BYTES,
            rocksdb::COMPACT_WRITE_BYTES,
            rocksdb::FLUSH_WRITE_BYTES,
            rocksdb::STALL_MICROS,
        };
        
        for (auto ticker : tickers) {
            prev_ticker_values_[std::to_string(ticker)] = 
                statistics_->getTickerCount(ticker);
        }
    }
    
    // PerfContext快照(使用特殊key进行区分)
    auto perf_ctx = rocksdb::get_perf_context();
    prev_ticker_values_["perf_block_read_byte"] = perf_ctx->block_read_byte;
}

// ✅在一个epoch结束时调用tune
bool MemoryTuner::tune() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!is_tuning_enabled_) {
        return false;
    }
    
    tuning_count_++;
    spdlog::info("=== Memory Tuning Cycle {} ===", tuning_count_);
    
    // Step1: 收集统计信息
    collect_from_statistics();

    // 在计算成本导数之前，输出收集到的原始统计数据
    spdlog::info("Raw statistics collected:");
    spdlog::info("  operations: {}", current_stats_.operations);
    spdlog::info("  total_merge_writes: {} pages", current_stats_.total_merge_writes);
    spdlog::info("  total_merge_reads: {} pages", current_stats_.total_merge_reads);
    spdlog::info("  total_query_reads: {} pages", current_stats_.total_query_reads);
    spdlog::info("  total_flush_writes: {} pages", current_stats_.total_flush_writes);
    spdlog::info("  memory_triggered_flushes: {}", current_stats_.memory_triggered_flushes);
    spdlog::info("  log_triggered_flushes: {}", current_stats_.log_triggered_flushes);
    spdlog::info("  saved_query_reads: {:.6f} pages/op", current_stats_.saved_query_reads);
    spdlog::info("  saved_merge_reads: {:.6f} pages/op", current_stats_.saved_merge_reads);

    
    // Step2: 计算成本导数(必须先写成本后读成本)
    // cost'(x) = ω·write'(x) + γ·read'(x)
    double write_derivative = estimate_write_cost_derivative(current_write_memory_);
    double read_derivative = estimate_read_cost_derivative(write_derivative);
    double cost_derivative = config_.write_weight * write_derivative + 
                         config_.read_weight * read_derivative;

    // 新增：分别输出读写导数
    spdlog::info("Cost derivatives at x={} MB:", current_write_memory_ / (1024 * 1024));
    spdlog::info("  write'(x) = {:.6e} pages/(op·MB)", write_derivative);
    spdlog::info("  read'(x)  = {:.6e} pages/(op·MB)", read_derivative);
    spdlog::info("  cost'(x)  = {:.6e} pages/(op·MB)", cost_derivative);

    
    // Step3: 计算I/O成本 单位：pages/op      
    double write_cost = static_cast<double>(current_stats_.total_merge_writes + 
                                            current_stats_.total_flush_writes) / 
                       current_stats_.operations;
    double read_cost = static_cast<double>(current_stats_.total_query_reads + 
                                       current_stats_.total_merge_reads) / 
                   current_stats_.operations;
    double current_io_cost = config_.write_weight * write_cost + 
                         config_.read_weight * read_cost;    

    spdlog::info("Cost at x={} MB:", current_write_memory_ / (1024 * 1024));
    spdlog::info("  write(x) = {:.6e}", write_cost);
    spdlog::info("  read(x)  = {:.6e}", read_cost);
    spdlog::info("  cost(x)  = ω*{:.6e} + γ*{:.6e} = {:.6e}", 
                 write_cost, read_cost, current_io_cost);

    
    // Step 4: 记录当前调优点
    TuningPoint current_point;
    current_point.write_memory_size = current_write_memory_;
    current_point.cost_derivative = cost_derivative; // 总成本导数
    current_point.io_cost = current_io_cost; // 总成本
    current_point.write_cost = write_cost;
    current_point.read_cost = read_cost;
    tuning_history_.push_back(current_point);
    
    // 保持历史记录在合理大小，及时清除过期的历史记录
    while (tuning_history_.size() > config_.K * 2) {
        tuning_history_.pop_front();
    }
    
    // Step 5: 使用Newton-Raphson方法计算新的write memory大小
    size_t new_write_memory = newton_raphson_step(current_write_memory_, cost_derivative);
    
    // Step 6: 应用稳定性启发式规则
    new_write_memory = apply_stability_heuristics(new_write_memory, current_write_memory_);
    
    // Step 7: 检查停止条件
    size_t step_size = (new_write_memory > current_write_memory_) ?
                       (new_write_memory - current_write_memory_) :
                       (current_write_memory_ - new_write_memory);
    
    double expected_reduction = std::abs(cost_derivative) * step_size;
    
    if (should_stop_tuning(step_size, expected_reduction)) {
        spdlog::info("Tuning stopped: step size or expected reduction too small");
        has_converged_ = true;  // 设置为true表示后续无需再进行调优
        return false;
    }
    
    // Step 8: 应用新的内存分配
    spdlog::info("Adjusting memory allocation:");
    spdlog::info("  Write memory: {} MB -> {} MB", 
                    current_write_memory_ / (1024 * 1024),
                    new_write_memory / (1024 * 1024));
    spdlog::info("  Buffer cache: {} MB -> {} MB",
                    get_buffer_cache_size() / (1024 * 1024),
                    (config_.total_memory - new_write_memory) / (1024 * 1024));
    
    bool success = apply_memory_allocation(new_write_memory);
    
    reset_statistics();

    return success;
}

// ✅ 判断是否应该执行调优(即结束一个epoch)
std::pair<bool, TuningTrigger> MemoryTuner::should_tune() {
    if(has_converged_){
        return {false, TuningTrigger::NONE}; 
    }

    auto now = std::chrono::steady_clock::now();
    // 计算距离上一次调优的时间间隔
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
    now - last_tuning_time_).count();

    // 无论什么原因触发，距离上次调优太近就直接拒绝
    if (elapsed < static_cast<long>(config_.min_tuning_interval_seconds)) {
        // 同时清除log标志，避免累积
        if (log_flush_occurred_.load()) {
            log_flush_occurred_.store(false);
        }

        return {false, TuningTrigger::NONE}; // ← 直接拒绝，不调优
    }

    // ✅只有在满足调优条件而实际触发调优时才需要更新last_tuning_time_时间
    // 条件1: log-triggered flush
    if (log_flush_occurred_.load()) {
        log_flush_occurred_.store(false); // 立即重置，等待下一次触发
        last_tuning_time_ = now;
        return {true, TuningTrigger::LOG_FLUSH};
    }
    
    // 条件2: 时间兜底
    if (elapsed >= static_cast<long>(config_.tuning_interval_seconds)) {
        last_tuning_time_ = now;
        return {true, TuningTrigger::TIME_BASED};
    }
    
    return {false, TuningTrigger::NONE};
}

// ✅ 重置当前epoch统计信息，保留上一周期结束时候的统计信息
void MemoryTuner::reset_statistics() {
    // 由于其在每一次成功执行后均执行保留
    // 因此，最后的一个就是最终结果
    if (statistics_) {
        prev_ticker_values_.clear();
        
        // 保存所有相关的ticker值
        std::vector<uint32_t> tickers = {
            rocksdb::BLOCK_CACHE_HIT,
            rocksdb::BLOCK_CACHE_MISS,
            rocksdb::SIM_BLOCK_CACHE_HIT,   // 添加模拟缓存相关统计信息
            rocksdb::SIM_BLOCK_CACHE_MISS,
            // rocksdb::BLOCK_CACHE_DATA_MISS, // 用于估算磁盘查询读成本
            // rocksdb::BLOCK_CACHE_INDEX_MISS,
            // rocksdb::BLOCK_CACHE_FILTER_MISS,  
            rocksdb::BLOOM_FILTER_USEFUL,
            rocksdb::BYTES_WRITTEN,
            rocksdb::BYTES_READ,
            rocksdb::COMPACT_READ_BYTES,
            rocksdb::COMPACT_WRITE_BYTES,
            rocksdb::FLUSH_WRITE_BYTES,
            rocksdb::STALL_MICROS,
        };
        
        // 记录当前时刻的累积值，作为下一个epoch的起点，用于计算增量值
        for (auto ticker : tickers) {
            prev_ticker_values_[std::to_string(ticker)] = statistics_->getTickerCount(ticker);
        }

        auto perf_ctx = rocksdb::get_perf_context();
        prev_ticker_values_["perf_block_read_byte"] = perf_ctx->block_read_byte;
    }
    
    // 重置周期统计
    current_stats_.reset();

    // 重置Compactor中的epoch统计
    if (compactor_) {
        compactor_->stats.reset_epoch();
    }

}

void MemoryTuner::print_status() const {
    spdlog::info("=== Memory Tuner Status ===");
    spdlog::info("Tuning enabled: {}", is_tuning_enabled_);
    spdlog::info("Has converged: {}", has_converged_);
    spdlog::info("Tuning count: {}", tuning_count_);
    spdlog::info("Write memory: {} MB", current_write_memory_ / (1024 * 1024));
    spdlog::info("Buffer cache: {} MB", get_buffer_cache_size() / (1024 * 1024));
    
    if (!tuning_history_.empty()) {
        const auto& last = tuning_history_.back();
        spdlog::info("Last I/O cost: {:.4f} pages/op", last.io_cost);
        spdlog::info("Last cost derivative: {:.2e}", last.cost_derivative);
    }
}

// ✅ 写成本导数估计 修改12.30
double MemoryTuner::estimate_write_cost_derivative(size_t write_memory) {
    // write'(x) = -merge(x) / x·ln(|LN|/(x)) · [flushmem / (flushmem + flushlog)]

    if (write_memory == 0) {
        return -std::numeric_limits<double>::max();
    }
    
    // ✅ 写内存转换为MB
    double write_memory_mb = static_cast<double>(write_memory) / (1024.0 * 1024.0);

    // 1. 获取最后一层LN的大小 
    size_t last_level_size = 0;
    rocksdb::ColumnFamilyMetaData cf_meta;
    db_->GetColumnFamilyMetaData(&cf_meta);
    // 反向遍历寻找最后一个非空层(实际存储数据的最深层)，
    // 如果只是获取配置的最后一层，该层可能会是空的
    for (auto it = cf_meta.levels.rbegin(); it != cf_meta.levels.rend(); ++it) {
        if (!it->files.empty()) {
            last_level_size = it->size;
            break;
        }
    }

    double last_level_size_mb = static_cast<double>(last_level_size) / (1024.0 * 1024.0);

    // 2. 判断LN和x的关系，对数为0/负结果出错/没意义
    if (last_level_size_mb <= write_memory_mb) {
        return 0.0;
    }

    // 3. 计算merge(x) 单位: pages/op
    double merge_pages_per_op = 0.0;
    if (current_stats_.operations > 0) {
        merge_pages_per_op = static_cast<double>(current_stats_.total_merge_writes) / 
                            current_stats_.operations;
    }

    // 4. 计算导数主体部分
    // double ratio = static_cast<double>(last_level_size) / write_memory;
    double ratio = last_level_size_mb / write_memory_mb;
    double ln_ratio = std::log(ratio);

    // 防止除以接近0的数
    if (std::abs(ln_ratio) < 0.1) {  // 防止除以接近0的数
        return 0.0;
    }

    // ✅ 导数单位: pages/(op·MB)
    // double derivative = -merge_pages_per_op / (write_memory * ln_ratio);
    double derivative = -merge_pages_per_op / (write_memory_mb * ln_ratio);

    // 5. 计算系数部分 flushmem / flushmem + flushlog
    double scale_factor = 1.0;
    size_t total_flush = current_stats_.memory_triggered_flushes + 
                        current_stats_.log_triggered_flushes;
    if (total_flush > 0) {
        scale_factor = static_cast<double>(current_stats_.memory_triggered_flushes) / 
                      total_flush;
    }

    spdlog::debug("Write derivative calculation:");
    spdlog::debug("  merge_pages_per_op = {:.6f}", merge_pages_per_op);
    spdlog::debug("  ratio = |LN|/x = {:.2f}", ratio);
    spdlog::debug("  ln_ratio = {:.4f}", ln_ratio);
    spdlog::debug("  raw_derivative = {:.6e}", derivative);
    spdlog::debug("  scale_factor = {:.4f} (mem_flush={}, log_flush={})",
                  scale_factor, current_stats_.memory_triggered_flushes,
                  current_stats_.log_triggered_flushes);

    double result = derivative * scale_factor;
    // spdlog::debug("Write cost derivative: {:.2e}", result);
    return result;
}

// ✅ 读成本导数估计
double MemoryTuner::estimate_read_cost_derivative(double write_derivative) {
    // read'(x) = (saved_q + saved_m)/sim + write'(x) · read_m(x)/merge(x)

    // 使用SimCache(扩展128MB)后的模拟缓存可以节省的I/O统计
    // saved_q / saved_m 单位: pages / ops
    double saved_q = 0.0, saved_m = 0.0;
    saved_q = current_stats_.saved_query_reads;
    saved_m = current_stats_.saved_merge_reads;
    
    // 第一项：(saved_q + saved_m) / sim
    // ✅ 首先将sim_cache_size转换为MB 第一项单位: pages/(op·MB)
    double sim_cache_mb = static_cast<double>(config_.sim_cache_size) / (1024.0 * 1024.0);
    double term1 = 0.0;
    if (config_.sim_cache_size > 0) {
        // term1 = (saved_q + saved_m) / static_cast<double>(config_.sim_cache_size);
        term1 = (saved_q + saved_m) / sim_cache_mb;
    }
    
    // 第二项：write'(x) · read_m(x) / merge(x) 其中的write_derivative已经是pages/(op·MB)了
    double term2 = 0.0;
    if (current_stats_.operations > 0 && current_stats_.total_merge_writes > 0) {
        double read_m_per_op = static_cast<double>(current_stats_.total_merge_reads) / 
                               current_stats_.operations;
        double merge_per_op = static_cast<double>(current_stats_.total_merge_writes) / 
                              current_stats_.operations;
        
        term2 = write_derivative * (read_m_per_op / merge_per_op);
    }
    
    double total_derivative = term1 + term2;
    
    spdlog::debug("Read cost derivative: {:.2e} (term1={:.2e}, term2={:.2e})", 
                  total_derivative, term1, term2);
    return total_derivative;
}

// ✅ 使用最小二乘法拟合线性模型 cost'(x) = Ax + B
bool MemoryTuner::fit_linear_model(double& A, double& B) {
    // 确保当前的样本已经符合拟合数量要求
    if (tuning_history_.size() < config_.K) {
        return false;
    }

    size_t n = config_.K;
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    
    auto it = std::prev(tuning_history_.end(), n); // 指向倒数第n(3)个元素
    
    for (size_t i = 0; i < n; ++i, ++it) {
        // ✅ 单位转换为 MB
        // double x = static_cast<double>(it->write_memory_size);
        double x = static_cast<double>(it->write_memory_size) / (1024.0 * 1024.0);
        double y = it->cost_derivative; //单位已经是pages/(op·MB)
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    
    double denom = n * sum_xx - sum_x * sum_x;
    
    // if (std::abs(denom) < 1e-20) {
    //     return false;
    // }
    // ✅ 调整阈值，现在 x 是 MB 量级（几十到几千）
    if (std::abs(denom) < 1e-6) {
        spdlog::debug("Linear fit failed: denom={:.6e} too small", denom);
        return false;
    }
    
    A = (n * sum_xy - sum_x * sum_y) / denom;
    B = (sum_y - A * sum_x) / n;
    
    spdlog::debug("Linear fit: cost'(x) = {:.2e}*x + {:.2e}", A, B);
    return true;
}

// ✅ Newton-Raphson调优实现 计算下一个写内存变量x的大小
// ❗️当读写成本导数均转换为pages/op*MB的单位之后，我们的拟合调优也要更新为MB
size_t MemoryTuner::newton_raphson_step(size_t current_x, double current_derivative) {
    // 首先将当前写内存转换为MB
    double current_x_mb = static_cast<double>(current_x) / (1024.0 * 1024.0);

    double A, B;
    
    // 尝试使用线性拟合 xi+1 = xi - cost'(xi) / A 
    if (fit_linear_model(A, B) && std::abs(A) > 1e-20) {
        // double step = current_derivative / A;
        double step_mb = current_derivative / A;
        
        // 检查变化幅度是否合理
        if (std::isfinite(step_mb) && std::abs(step_mb) < config_.total_memory) {
            double new_x_mb = current_x_mb - step_mb;

            // 防止负数转size_t的未定义行为
            if (new_x_mb < 0) {
                new_x_mb = 0;
            }
            
            // 再将其转换为bytes
            size_t new_x = static_cast<size_t>(new_x_mb * 1024.0 * 1024.0);

            spdlog::debug("Newton-Raphson: x={} MB, derivative={:.2e}, A={:.2e}, step={:.0f} bytes",
                          current_x_mb / (1024 * 1024), current_derivative, A, step_mb);
            return new_x;
        }
    }
    
    // 若不满足要求(通常是因为fit函数返回false)，回退到固定步长(启发式规则1)
    spdlog::debug("Falling back to fixed step size");
    
    // cost'(x) > 0 意味着增加write memory会增加成本，所以应该减少x
    // cost'(x) < 0 意味着增加write memory会减少成本，所以应该增加x
    if (current_derivative > 0) { // 避免下溢 直接使用最小值
        return current_x > config_.fixed_step_size ? 
               current_x - config_.fixed_step_size : config_.min_write_memory;
    } else if (current_derivative < 0) {
        return current_x + config_.fixed_step_size;
    }
    
    return current_x; // 隐式处理了cost'(x) == 0的情况
}

// ✅ 最大步长限制确保稳定性
size_t MemoryTuner::apply_stability_heuristics(size_t proposed_x, size_t current_x) {
    size_t new_x = proposed_x;

    // 确保在有效范围内
    new_x = std::max(new_x, config_.min_write_memory); // new_x应该>最小值
    new_x = std::min(new_x, config_.total_memory - config_.min_buffer_cache); // new_x应该<最大值
    
    // 最大步长限制 max_step_ratio=0.1
    size_t max_decrease_write = static_cast<size_t>(current_x * config_.max_step_ratio);
    size_t max_decrease_cache = static_cast<size_t>(get_buffer_cache_size() * config_.max_step_ratio);

    if (new_x < current_x) {
        // 减少write memory
        size_t decrease = current_x - new_x;
        if (decrease > max_decrease_write) {
            new_x = current_x - max_decrease_write;
            spdlog::debug("Limited write memory decrease to {} MB", 
                          max_decrease_write / (1024 * 1024));
        }
    } else if (new_x > current_x) {
        // 增加write memory（减少buffer cache）
        size_t increase = new_x - current_x;
        if (increase > max_decrease_cache) {
            new_x = current_x + max_decrease_cache;
            spdlog::debug("Limited buffer cache decrease to {} MB",
                          max_decrease_cache / (1024 * 1024));
        }
    }

    // 再次确保在有效范围内
    new_x = std::max(new_x, config_.min_write_memory);
    new_x = std::min(new_x, config_.total_memory - config_.min_buffer_cache);
    
    return new_x;
}

// ✅ 检查两个停止条件
bool MemoryTuner::should_stop_tuning(size_t step_size, double expected_reduction) {
    // 条件1：步长过小 32MB
    if (step_size < config_.min_step_size) {
        spdlog::debug("Step size {} MB < min {} MB", 
                      step_size / (1024 * 1024), 
                      config_.min_step_size / (1024 * 1024));
        return true;
    }
    
    // 条件2：预期成本降低过小 0.1%
    if (!tuning_history_.empty()) {
        double current_cost = tuning_history_.back().io_cost;
        if (current_cost > 0 && 
            expected_reduction / current_cost < config_.min_cost_reduction_ratio) {
            spdlog::debug("Expected reduction {:.2e} < {:.2f}% of current cost",
                          expected_reduction, config_.min_cost_reduction_ratio * 100);
            return true;
        }
    }
    
    return false;
}

// ✅ 应用新的内存分配(写内存 + 块缓存)
// 写内存的应用是从下一个memtable开始调整；block cache则是直接更改(可能驱逐对象)
bool MemoryTuner::apply_memory_allocation(size_t new_write_memory) {
    size_t old_write_memory = current_write_memory_;
    size_t new_buffer_cache = config_.total_memory - new_write_memory;
    
    // 调整block cache大小
    resize_block_cache(new_buffer_cache);
    
    // 调整write buffer大小（通过SetOptions）
    resize_write_buffer(new_write_memory);
    
    // 更新当前状态
    current_write_memory_ = new_write_memory;
    
    spdlog::info("Memory allocation applied:");
    spdlog::info("  Write memory: {} MB -> {} MB",
                 old_write_memory / (1024 * 1024),
                 new_write_memory / (1024 * 1024));
    spdlog::info("  Buffer cache: {} MB -> {} MB",
                 (config_.total_memory - old_write_memory) / (1024 * 1024),
                 new_buffer_cache / (1024 * 1024));
    
    return true;
}

// ✅ 动态调整块缓存的大小，并递归调整模拟缓存的大小
// 模拟缓存大小始终等于(动态变化的)块缓存容量 + (固定的)sim_cache_size 128MB
void MemoryTuner::resize_block_cache(size_t new_size) {
    if (block_cache_) {
        block_cache_->SetCapacity(new_size);
        spdlog::debug("Block cache capacity set to {} MB", new_size / (1024 * 1024));
    }
    // 同时调整SimCache的模拟容量 = 实际缓存 + 128MB
    if (sim_cache_) {
        size_t new_sim_capacity = new_size + config_.sim_cache_size;
        sim_cache_->SetSimCapacity(new_sim_capacity);
        spdlog::debug("SimCache sim_capacity set to {} MB (= {} + {} MB)",
                    new_sim_capacity / (1024 * 1024),
                    new_size / (1024 * 1024),
                    config_.sim_cache_size / (1024 * 1024));
    }
}

// ✅ 动态调整写内存大小，其只会影响到新的memtable，旧的memtable并不会受到影响
void MemoryTuner::resize_write_buffer(size_t new_size) {
    if (db_) {
        std::unordered_map<std::string, std::string> new_options;
        new_options["write_buffer_size"] = std::to_string(new_size);
        
        rocksdb::Status s = db_->SetOptions(new_options);
        if (s.ok()) {
            spdlog::debug("Write buffer size set to {} MB", new_size / (1024 * 1024));
        } else {
            spdlog::warn("Failed to set write buffer size: {}", s.ToString());
        }

        compactor_->updateM(new_size);   // 更新compactor的配置
    }
}

// ✅ 从RocksDB Statistics收集统计信息
void MemoryTuner::collect_from_statistics() {
    if (!statistics_) {
        return;
    }
    
    // 统一的增量计算工具，用于Statistics
    auto get_delta = [this](uint32_t ticker) -> uint64_t {
        uint64_t current = statistics_->getTickerCount(ticker);
        std::string key = std::to_string(ticker);
        uint64_t prev = 0;
        if (prev_ticker_values_.count(key)) {
            prev = prev_ticker_values_[key];
        }
        return current - prev;
    };

    // PerfContext增量计算
    auto perf_ctx = rocksdb::get_perf_context();
    uint64_t query_read_bytes = perf_ctx->block_read_byte - prev_ticker_values_["perf_block_read_byte"];

    // 1.I/O统计 bytes➡️pages
    uint64_t compact_read = get_delta(rocksdb::COMPACT_READ_BYTES);
    uint64_t compact_write = get_delta(rocksdb::COMPACT_WRITE_BYTES);
    uint64_t flush_write = get_delta(rocksdb::FLUSH_WRITE_BYTES);
    uint64_t cache_hits = get_delta(rocksdb::BLOCK_CACHE_HIT);
    uint64_t cache_misses = get_delta(rocksdb::BLOCK_CACHE_MISS);
    // uint64_t data_miss = get_delta(rocksdb::BLOCK_CACHE_DATA_MISS);
    // uint64_t index_miss = get_delta(rocksdb::BLOCK_CACHE_INDEX_MISS);
    // uint64_t filter_miss = get_delta(rocksdb::BLOCK_CACHE_FILTER_MISS);

    spdlog::info("I/O Collection Result: compact_read={}, compact_write={}, flush_write={}, "
                  "cache_hits={}, cache_misses={}",
                  compact_read,
                  compact_write,
                  flush_write,
                  cache_hits,
                  cache_misses);

    // 总的读写Bytes / 每个page的Bytes =  (单位)读写的pages/blocks 
    current_stats_.total_merge_writes = compact_write / config_.page_size;
    current_stats_.total_flush_writes = flush_write / config_.page_size;
    current_stats_.total_merge_reads = compact_read / config_.page_size;
    current_stats_.total_query_reads = query_read_bytes / config_.page_size;
    current_stats_.total_reads = current_stats_.total_query_reads + current_stats_.total_merge_reads;
    
    // 2.验证操作数(由外部通过record_operation记录)
    if (current_stats_.operations == 0) {
        spdlog::warn("operations is 0, this may indicate record_operation() was not called");
        current_stats_.operations = 1;  // 避免除零，但这是异常情况
    }

    // 3.Cache统计
    current_stats_.cache_hits = cache_hits;
    current_stats_.cache_misses = cache_misses;
    if (cache_hits + cache_misses > 0) {
        current_stats_.cache_hit_rate = static_cast<double>(cache_hits) / 
                                        (cache_hits + cache_misses);
    }
    
    // 4. SimCache统计
    if(sim_cache_){
        uint64_t delta_sim_hits = get_delta(rocksdb::SIM_BLOCK_CACHE_HIT);
        uint64_t delta_actual_hits = cache_hits;  // 复用上面已计算的值

        // saved_hits (单位：pages)
        int64_t saved_hits = static_cast<int64_t>(delta_sim_hits) - 
                             static_cast<int64_t>(delta_actual_hits);
        if (saved_hits < 0) {
            saved_hits = 0;
        }
        
        // ✅ 通过输出确认两个值之间的关系
        spdlog::debug("SimCache stats: sim_hits={}, actual_hits={}, saved={}",
              delta_sim_hits, delta_actual_hits, saved_hits);

        // 计算saved/op (单位pages/op)
        if (current_stats_.operations > 0 && saved_hits > 0 && current_stats_.total_reads > 0) {
            // 先计算总数
            double total_saved_per_op = static_cast<double>(saved_hits) / 
                                        current_stats_.operations;


            double query_ratio = static_cast<double>(current_stats_.total_query_reads) / 
                                current_stats_.total_reads;
            current_stats_.saved_query_reads = total_saved_per_op * query_ratio;
            current_stats_.saved_merge_reads = total_saved_per_op * (1.0 - query_ratio);

            }
    }

    // 5. 从Compactor中获取flush类型统计用于计算写成本系数
    if (compactor_) {
        current_stats_.memory_triggered_flushes = 
            compactor_->stats.epoch_memory_triggered_flush_count;
        current_stats_.log_triggered_flushes = 
            compactor_->stats.epoch_log_triggered_flush_count;
    }

    current_stats_.end_time = std::chrono::steady_clock::now();
    spdlog::info("Statistics collected: ops={}, merge_writes={}, merge_reads={}, "
                  "saved_q={:.4f}, saved_m={:.4f}",
                  current_stats_.operations,
                  current_stats_.total_merge_writes,
                  current_stats_.total_merge_reads,
                  current_stats_.saved_query_reads,
                  current_stats_.saved_merge_reads);
}

} // namespace tmpdb