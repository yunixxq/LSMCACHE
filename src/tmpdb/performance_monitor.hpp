#ifndef PERFORMANCE_MONITOR_H_
#define PERFORMANCE_MONITOR_H_

#include <map>
#include <string>
#include <chrono>

#include "rocksdb/db.h"
#include "rocksdb/statistics.h"
#include "tmpdb/compactor.hpp"

namespace tmpdb
{

// ============================================================
// 性能指标结构 存储单个epoch内的所有性能指标
// ============================================================
struct PerformanceMetrics
{
    // ===== Cache 侧指标 =====
    double cache_hit_rate;              // 命中率(真实H_total)
    uint64_t epoch_cache_hits;          // epoch内命中次数
    uint64_t epoch_cache_misses;        // epoch内未命中次数
    
    // ===== Write 侧指标 =====
    double write_stall_rate;            // 写入停顿时间占epoch总时间的比例，范围[0.0, 1.0]
    uint64_t epoch_write_stall_us;      // epoch内stall微秒数
    
    // ===== Flush 指标 =====
    uint64_t epoch_flush_count;         // epoch内flush次数
    double flush_rate;                  // flush频率（次/秒）
    
    // ===== Compaction 指标 =====
    uint64_t epoch_compaction_count;    // epoch内compaction次数
    double compaction_rate;             // compaction频率(次/秒)
    uint64_t epoch_compaction_input_files;   // epoch内compaction输入文件数
    
    // ===== 读写字节数 ===== OnCompactionCompleted无法
    // uint64_t epoch_compaction_read_bytes;
    // uint64_t epoch_compaction_write_bytes;
    
    // ===== 综合指标 =====
    double epoch_duration_seconds;      // epoch持续时间
    
    // ===== 时间戳 =====
    std::chrono::steady_clock::time_point timestamp;
    
    // 打印指标
    void print() const {
        spdlog::info("=== Performance Metrics ===");
        spdlog::info("Cache: hit_rate={:.4f}, hits={}, misses={}", 
                     cache_hit_rate, epoch_cache_hits, epoch_cache_misses);
        spdlog::info("Write: stall_rate={:.4f}", write_stall_rate);
        spdlog::info("Flush: count={}, rate={:.2f}/s", 
                     epoch_flush_count, flush_rate);
        spdlog::info("Compaction: count={}, rate={:.2f}/s", 
                     epoch_compaction_count, compaction_rate);
    }
};

// ============================================================
// Performance Monitor 类
// ============================================================
class PerformanceMonitor
{
private:
    rocksdb::DB* db_;
    rocksdb::Statistics* statistics_;  // RocksDB在这个对象中累积各种操作计数
    Compactor* compactor_;  // 自定义Compactor对象
    
    // 上一次调用collect时的信息(用于计算增量)
    uint64_t prev_cache_hits_ = 0;
    uint64_t prev_cache_misses_ = 0;
    uint64_t prev_write_stall_us_ = 0;
    std::chrono::steady_clock::time_point prev_time_; //上一次调用collect的时间点
    
public:
    PerformanceMonitor(rocksdb::DB* db, 
                       rocksdb::Statistics* statistics,
                       Compactor* compactor)
        : db_(db), statistics_(statistics), compactor_(compactor)
    {
        prev_time_ = std::chrono::steady_clock::now(); //构造时的时间点作为第一个epoch的起始时间

        // 初始化快照
        if (statistics_) {
            spdlog::info("Initializing Performance Monitor snapshots..."); // 表明statistics存在
            std::map<std::string, uint64_t> stats;
            statistics_->getTickerMap(&stats);
            prev_cache_hits_ = stats["rocksdb.block.cache.hit"];
            prev_cache_misses_ = stats["rocksdb.block.cache.miss"];
            // STALL_MICROS: Writer has to wait for compaction or flush to finish.
            prev_write_stall_us_ = stats["rocksdb.stall.micros"];
        }
    }

    // 采集当前epoch的性能指标
    PerformanceMetrics collect()
    {
        PerformanceMetrics m; // 新结构体用于存储本轮epoch采集的指标
        m.timestamp = std::chrono::steady_clock::now();
        
        // 计算epoch持续时间
        m.epoch_duration_seconds = std::chrono::duration<double>(
            m.timestamp - prev_time_).count();
        
        // ===== 从RocksDB Statistics获取 =====
        if (statistics_) {
            std::map<std::string, uint64_t> stats;
            statistics_->getTickerMap(&stats);
            
            uint64_t curr_hits = stats["rocksdb.block.cache.hit"];
            uint64_t curr_misses = stats["rocksdb.block.cache.miss"];
            uint64_t curr_stall_us = stats["rocksdb.stall.micros"];
            
            // 计算epoch内的增量
            m.epoch_cache_hits = curr_hits - prev_cache_hits_;
            m.epoch_cache_misses = curr_misses - prev_cache_misses_;
            m.epoch_write_stall_us = curr_stall_us - prev_write_stall_us_;
            
            // (1)计算命中率 三元运算符避免除0，如果没有访问命中率为0
            uint64_t total_accesses = m.epoch_cache_hits + m.epoch_cache_misses;
            m.cache_hit_rate = total_accesses > 0 
                ? static_cast<double>(m.epoch_cache_hits) / total_accesses 
                : 0.0;
            
            // (2)计算Write Stall率 ❗️✅注意这里可能会>1，是否更改为使用stall次数而非时间
            uint64_t epoch_us = static_cast<uint64_t>(m.epoch_duration_seconds * 1e6);
            m.write_stall_rate = epoch_us > 0 
                ? static_cast<double>(m.epoch_write_stall_us) / epoch_us 
                : 0.0;
            
            // 更新快照
            prev_cache_hits_ = curr_hits;
            prev_cache_misses_ = curr_misses;
            prev_write_stall_us_ = curr_stall_us;
        }
        
        // ===== 从Compactor获取 =====
        if (compactor_) {
            m.epoch_flush_count = compactor_->stats.epoch_flush_count;
            m.epoch_compaction_count = compactor_->stats.epoch_compaction_count;
            m.epoch_compaction_input_files = compactor_->stats.epoch_input_files;
            // m.epoch_compaction_read_bytes = compactor_->stats.epoch_compaction_read_bytes;
            // m.epoch_compaction_write_bytes = compactor_->stats.epoch_compaction_write_bytes;
            
            // (3)计算flush/compaction频率
            m.flush_rate = m.epoch_duration_seconds > 0 
                ? m.epoch_flush_count / m.epoch_duration_seconds 
                : 0.0;
            m.compaction_rate = m.epoch_duration_seconds > 0 
                ? m.epoch_compaction_count / m.epoch_duration_seconds 
                : 0.0;
            
            // 重置epoch计数器
            compactor_->stats.reset_epoch();
        }
        
        // 更新时间戳
        prev_time_ = m.timestamp;
        
        return m;
    }
};

} /* namespace tmpdb */

#endif /* PERFORMANCE_MONITOR_H_ */