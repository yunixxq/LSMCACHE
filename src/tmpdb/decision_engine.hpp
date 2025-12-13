#ifndef DECISION_ENGINE_H_
#define DECISION_ENGINE_H_

#include <deque>
#include <algorithm>
#include "spdlog/spdlog.h"
#include "tmpdb/performance_monitor.hpp"

namespace tmpdb
{

// ============================================================
// 调整动作枚举
// ============================================================
enum class AdjustmentAction
{
    NO_CHANGE,
    INCREASE_ALPHA,  // 增加Write Buffer-缓解写入瓶颈
    DECREASE_ALPHA   // 增加Block Cache-缓解读取瓶颈
};

// ============================================================
// 调整决策结构
// ============================================================
struct AdjustmentDecision
{
    AdjustmentAction action;
    double delta;           // 调整幅度
    std::string reason;     // 调整原因（用于日志）
};

// ============================================================
// 阈值配置
// ============================================================
struct ThresholdConfig
{
    // Cache 侧阈值
    double cache_hit_rate_low = 0.80;       // 低于此值认为 Cache 不足
    double cache_hit_rate_target = 0.90;    // 目标命中率（❗️未使用）
    
    // Write 侧阈值
    double write_stall_rate_high = 0.05;    // 高于 5% 认为 Write Buffer 不足
    double compaction_rate_high = 5.0;      // 每秒超过 5 次 compaction 认为过高
    double flush_rate_high = 10.0;          // 每秒超过 10 次 flush 认为过高
    
    // 调整参数
    double alpha_step = 0.05;               // 每次调整 5%
    double alpha_min = 0.1;                 // α 最小值
    double alpha_max = 0.9;                 // α 最大值
    
    // 稳定性参数
    int stability_window = 3;               // 连续 N 个 epoch 才触发调整
};

// ============================================================
// Decision Engine 类
// ============================================================
class DecisionEngine
{
private:
    double current_alpha_; // [0.1, 0.9]
    ThresholdConfig config_;
    
    // 稳定性检查：记录最近的决策
    // 双端队列，存储最近N次的调整动作
    std::deque<AdjustmentAction> recent_actions_;
    
public:
    DecisionEngine(double initial_alpha, ThresholdConfig config = {})
        : current_alpha_(initial_alpha), config_(config) {}
    
    // 根据性能指标做出决策
    AdjustmentDecision decide(const PerformanceMetrics& m)
    {
        spdlog::info("decide() entered");
        spdlog::info("metrics: cache_hit_rate={}, write_stall_rate={}, compaction_rate={}, flush_rate={}",
                 m.cache_hit_rate, m.write_stall_rate, m.compaction_rate, m.flush_rate);
    
        // 检查是否有无效值
        if (std::isnan(m.cache_hit_rate) || std::isnan(m.write_stall_rate) ||
            std::isnan(m.compaction_rate) || std::isnan(m.flush_rate)) {
            spdlog::error("Invalid NaN value detected in metrics!");
        }

        AdjustmentDecision decision{AdjustmentAction::NO_CHANGE, 0.0, ""};
        
        // ===== Step 1: 诊断瓶颈 =====
        bool write_pressure = is_write_bottleneck(m);
        bool read_pressure = is_read_bottleneck(m);
        
        spdlog::info("Diagnosis: write_pressure={}, read_pressure={}", 
                      write_pressure, read_pressure);
        
        // ===== Step 2: 决策逻辑 =====
        if (write_pressure && !read_pressure) 
        {
            // 写入瓶颈：需要更多 Write Buffer
            decision.action = AdjustmentAction::INCREASE_ALPHA;
            decision.delta = compute_delta(m, true);
            decision.reason = fmt::format(
                "Write bottleneck: stall_rate={:.2f}%, compaction_rate={:.1f}/s, flush_rate={:.1f}/s",
                m.write_stall_rate * 100.0, m.compaction_rate, m.flush_rate);
        }
        else if (read_pressure && !write_pressure)
        {
            // 读取瓶颈：需要更多 Cache
            decision.action = AdjustmentAction::DECREASE_ALPHA;
            decision.delta = compute_delta(m, false);

            decision.reason = fmt::format(
                "Read bottleneck: cache_hit_rate={:.2f}%",
                m.cache_hit_rate * 100.0);
        }
        else if (write_pressure && read_pressure)
        {
            // 两侧都有问题：选择更严重的一侧
            double write_severity = compute_write_severity(m);
            double read_severity = compute_read_severity(m);
            
            if (write_severity > read_severity)
            {
                decision.action = AdjustmentAction::INCREASE_ALPHA;
                decision.delta = compute_delta(m, true);
                decision.reason = "Both bottlenecks, write more severe";
            }
            else
            {
                decision.action = AdjustmentAction::DECREASE_ALPHA;
                decision.delta = compute_delta(m, false);
                decision.reason = "Both bottlenecks, read more severe";
            }
        }
        // else: 两侧都正常，不调整
        
        // ===== Step 3: 稳定性检查 =====
        if (!should_apply(decision.action))
        {
            decision.action = AdjustmentAction::NO_CHANGE;
            decision.reason = "Stability check: waiting for consistent signal";
        }
        
        return decision;
    }
    
    // 应用决策，返回新的 alpha 值
    double apply_decision(const AdjustmentDecision& d)
    {
        if (d.action == AdjustmentAction::INCREASE_ALPHA)
        {
            current_alpha_ = std::min(config_.alpha_max, current_alpha_ + d.delta);
            spdlog::info("Alpha adjusted: +{:.2f} -> {:.2f}, reason: {}", 
                         d.delta, current_alpha_, d.reason);
        }
        else if (d.action == AdjustmentAction::DECREASE_ALPHA)
        {
            current_alpha_ = std::max(config_.alpha_min, current_alpha_ - d.delta);
            spdlog::info("Alpha adjusted: -{:.2f} -> {:.2f}, reason: {}", 
                         d.delta, current_alpha_, d.reason);
        }
        return current_alpha_;
    }
    
    double get_current_alpha() const { return current_alpha_; }
    
    void set_config(const ThresholdConfig& config) { config_ = config; }
    
    // 判断是否有写入瓶颈
    bool is_write_bottleneck(const PerformanceMetrics& m) const
    {
        return (m.write_stall_rate > config_.write_stall_rate_high) ||
               (m.compaction_rate > config_.compaction_rate_high) ||
               (m.flush_rate > config_.flush_rate_high);
    }
    
    // 判断是否有读取瓶颈
    bool is_read_bottleneck(const PerformanceMetrics& m) const
    {
        return m.cache_hit_rate < config_.cache_hit_rate_low;
    }

private:
    // 计算写入瓶颈严重程度 实际值/阈值
    double compute_write_severity(const PerformanceMetrics& m) const
    {
        double stall_severity = m.write_stall_rate / config_.write_stall_rate_high;
        double compaction_severity = m.compaction_rate / config_.compaction_rate_high;
        double flush_severity = m.flush_rate / config_.flush_rate_high;
        return std::max({stall_severity, compaction_severity, flush_severity});
    }
    
    // 计算读取瓶颈严重程度
    double compute_read_severity(const PerformanceMetrics& m) const
    {
        if (m.cache_hit_rate >= config_.cache_hit_rate_low) return 0.0;
        return (config_.cache_hit_rate_low - m.cache_hit_rate) / config_.cache_hit_rate_low;
    }
    
    // 计算调整幅度
    double compute_delta(const PerformanceMetrics& m, bool is_write_issue) const
    {
        double severity;
        if (is_write_issue)
        {
            severity = compute_write_severity(m);
        }
        else
        {
            severity = compute_read_severity(m);
        }
        
        // 步长范围：[0.5 * base, 2.0 * base]
        // return config_.alpha_step * std::clamp(severity, 0.5, 2.0);
        return config_.alpha_step * std::max(0.5, std::min(severity, 2.0));
    }
    
    // 稳定性检查：连续N次相同方向才执行
    bool should_apply(AdjustmentAction action)
    {
        recent_actions_.push_back(action);
        if (recent_actions_.size() > static_cast<size_t>(config_.stability_window))
        {
            recent_actions_.pop_front();
        }
        
        // 窗口大小不足则不予执行
        if (recent_actions_.size() < static_cast<size_t>(config_.stability_window))
        {
            return false;
        }
        
        // 检查是否所有决策方向一致
        return std::all_of(recent_actions_.begin(), recent_actions_.end(),
            [action](AdjustmentAction a) { return a == action; });
    }
};

} /* namespace tmpdb */

#endif /* DECISION_ENGINE_H_ */