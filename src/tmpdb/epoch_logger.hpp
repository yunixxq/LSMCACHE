#ifndef EPOCH_LOGGER_H_
#define EPOCH_LOGGER_H_

#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>

#include "spdlog/spdlog.h"
#include "tmpdb/performance_monitor.hpp"
#include "tmpdb/decision_engine.hpp"

namespace tmpdb
{

// ============================================================
// Epoch Logger 类
// 负责将每个 epoch 的性能指标和决策记录到 CSV 文件
// ============================================================
class EpochLogger
{
private:
    std::ofstream file_;
    std::string filepath_;
    bool is_open_;
    
    // 实验开始时间（用于计算相对时间）
    std::chrono::steady_clock::time_point start_time_;

public:
    EpochLogger() : is_open_(false) {}
    
    ~EpochLogger()
    {
        close();
    }
    
    // 初始化并打开文件，写入 CSV 表头
    bool open(const std::string& filepath)
    {
        filepath_ = filepath;
        file_.open(filepath, std::ios::out | std::ios::trunc);
        
        if (!file_.is_open())
        {
            spdlog::error("Failed to open epoch log file: {}", filepath);
            return false;
        }
        
        is_open_ = true;
        start_time_ = std::chrono::steady_clock::now();
        
        // 写入 CSV 表头
        write_header();
        
        spdlog::info("Epoch logger initialized: {}", filepath);
        return true;
    }
    
    void close()
    {
        if (is_open_)
        {
            file_.close();
            is_open_ = false;
            spdlog::info("Epoch logger closed: {}", filepath_);
        }
    }
    
    // 记录一个 epoch 的完整信息
    void log_epoch(
        size_t epoch_num,                       // epoch 编号
        size_t step_num,                        // 当前步数
        const PerformanceMetrics& metrics,      // 性能指标
        bool write_bottleneck,                  // 是否存在写瓶颈
        bool read_bottleneck,                   // 是否存在读瓶颈
        AdjustmentAction action,                // 调整动作
        const std::string& reason,              // 调整原因
        double alpha_before,                    // 调整前 alpha
        double alpha_after,                     // 调整后 alpha
        bool adjustment_applied                 // 是否实际应用了调整 （存在稳定性窗口）
    )
    {
        if (!is_open_) return;
        
        // 计算相对时间（秒）
        auto now = std::chrono::steady_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(now - start_time_).count();
        
        // 将 action 转换为字符串
        std::string action_str = action_to_string(action);
        
        // 写入一行数据
        file_ << std::fixed << std::setprecision(6)
              // === 基本信息 ===
              << epoch_num << ","
              << step_num << ","
              << elapsed_seconds << ","
              
              // === 性能指标 ===
              << metrics.cache_hit_rate << ","
              << metrics.epoch_cache_hits << ","
              << metrics.epoch_cache_misses << ","
              << metrics.write_stall_rate << ","
              << metrics.epoch_write_stall_us << ","
              << metrics.epoch_flush_count << ","
              << metrics.flush_rate << ","
              << metrics.epoch_compaction_count << ","
              << metrics.compaction_rate << ","
            //   << metrics.epoch_compaction_read_bytes << ","
            //   << metrics.epoch_compaction_write_bytes << ","
              << metrics.epoch_duration_seconds << ","
              
              // === 瓶颈诊断 ===
              << (write_bottleneck ? 1 : 0) << ","
              << (read_bottleneck ? 1 : 0) << ","
              
              // === 决策信息 ===
              << action_str << ","
              << "\"" << escape_csv(reason) << "\"" << ","
              
              // === Alpha 变化 ===
              << alpha_before << ","
              << alpha_after << ","
              << (adjustment_applied ? 1 : 0)
              
              << "\n";
        
        // 立即刷新，确保数据写入
        file_.flush();
    }
    
    bool is_open() const { return is_open_; }

private:
    void write_header()
    {
        file_ << "epoch,"
              << "step,"
              << "elapsed_seconds,"
              
              // 性能指标
              << "cache_hit_rate,"
              << "cache_hits,"
              << "cache_misses,"
              << "write_stall_rate,"
              << "write_stall_us,"
              << "flush_count,"
              << "flush_rate,"
              << "compaction_count,"
              << "compaction_rate,"
              << "compaction_read_bytes,"
              << "compaction_write_bytes,"
              << "epoch_duration_seconds,"
              
              // 瓶颈诊断
              << "write_bottleneck,"
              << "read_bottleneck,"
              
              // 决策信息
              << "action,"
              << "reason,"
              
              // Alpha 变化
              << "alpha_before,"
              << "alpha_after,"
              << "adjustment_applied"
              
              << "\n";
        
        file_.flush();
    }
    
    std::string action_to_string(AdjustmentAction action) const
    {
        switch (action)
        {
            case AdjustmentAction::NO_CHANGE:
                return "NO_CHANGE";
            case AdjustmentAction::INCREASE_ALPHA:
                return "INCREASE_ALPHA";
            case AdjustmentAction::DECREASE_ALPHA:
                return "DECREASE_ALPHA";
            default:
                return "UNKNOWN";
        }
    }
    
    // 转义 CSV 中的特殊字符
    std::string escape_csv(const std::string& str) const
    {
        std::string result;
        for (char c : str)
        {
            if (c == '"')
                result += "\"\"";  // 双引号转义
            else
                result += c;
        }
        return result;
    }
};

} /* namespace tmpdb */

#endif /* EPOCH_LOGGER_H_ */