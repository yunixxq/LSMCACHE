#ifndef MEMORY_ALLOCATOR_H_
#define MEMORY_ALLOCATOR_H_

#include <cstdint>
#include <algorithm>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/cache.h"
#include "spdlog/spdlog.h"
#include "tmpdb/compactor.hpp"

namespace tmpdb
{

// ============================================================
// Memory Allocator 类
// 负责根据 alpha 值动态调整 Write Buffer 和 Block Cache 的分配
// ============================================================
class MemoryAllocator
{
private:
    rocksdb::DB* db_;
    std::shared_ptr<rocksdb::Cache> block_cache_;
    Compactor* compactor_;
    
    uint64_t total_memory_budget_;   // 总内存预算
    double current_alpha_;           // 当前 alpha 值
    
    // 配置参数
    int num_write_buffers_;          // Memtable数量
    uint64_t min_write_buffer_size_; // 最小 Memtable 大小
    uint64_t min_cache_size_;        // 最小 Cache 大小

public:
    MemoryAllocator(rocksdb::DB* db,
                    std::shared_ptr<rocksdb::Cache> block_cache,
                    Compactor* compactor,
                    uint64_t total_memory_budget,
                    double initial_alpha,
                    int num_write_buffers = 2)
        : db_(db),
          block_cache_(block_cache),
          compactor_(compactor),
          total_memory_budget_(total_memory_budget),
          current_alpha_(initial_alpha),
          num_write_buffers_(num_write_buffers),
          min_write_buffer_size_(1 << 20),   // 1 MB
          min_cache_size_(1 << 20)           // 1 MB
    {
        spdlog::info("MemoryAllocator initialized: total_budget={} MB, alpha={:.2f}",
                     total_memory_budget_ >> 20, current_alpha_);
    }
    
    // 根据新的alpha值调整内存分配
    bool adjust_allocation(double new_alpha)
    {
        // 限制 alpha 范围 确保两侧至少占有10%的内存
        // new_alpha = std::clamp(new_alpha, 0.1, 0.9); // C++11 不支持
        new_alpha = std::max(0.1, std::min(new_alpha, 0.9)); 
        
        if (std::abs(new_alpha - current_alpha_) < 0.01)
        {
            // 变化太小，不调整
            return false;
        }
        
        // 计算新的分配
        uint64_t new_write_buffer_total = static_cast<uint64_t>(total_memory_budget_ * new_alpha);
        uint64_t new_cache_size = total_memory_budget_ - new_write_buffer_total;
        
        // 确保最小值
        new_write_buffer_total = std::max(new_write_buffer_total, min_write_buffer_size_ * num_write_buffers_);
        new_cache_size = std::max(new_cache_size, min_cache_size_);
        
        // 重新计算实际 alpha
        new_alpha = static_cast<double>(new_write_buffer_total) / total_memory_budget_;
        
        // 计算单个 Write Buffer 大小
        uint64_t new_write_buffer_size = new_write_buffer_total / num_write_buffers_;
        
        spdlog::info("Adjusting memory allocation:");
        spdlog::info("  Alpha: {:.2f} -> {:.2f}", current_alpha_, new_alpha);
        spdlog::info("  Write Buffer: {} MB (per buffer: {} MB)", 
                     new_write_buffer_total >> 20, new_write_buffer_size >> 20);
        spdlog::info("  Block Cache: {} MB", new_cache_size >> 20);
        
        // ===== 调整 Write Buffer =====
        bool write_buffer_ok = adjust_write_buffer(new_write_buffer_size, new_write_buffer_total);
        
        // ===== 调整 Block Cache =====
        bool cache_ok = adjust_block_cache(new_cache_size);
        
        if (write_buffer_ok && cache_ok)
        {
            current_alpha_ = new_alpha;
            spdlog::info("Memory allocation adjusted successfully");
            return true;
        }
        else
        {
            spdlog::warn("Memory allocation adjustment partially failed");
            return false;
        }
    }
    
    // 获取当前状态
    double get_current_alpha() const { return current_alpha_; }
    uint64_t get_total_memory_budget() const { return total_memory_budget_; }
    
    uint64_t get_current_write_buffer_size() const
    {
        return static_cast<uint64_t>(total_memory_budget_ * current_alpha_);
    }
    
    uint64_t get_current_cache_size() const
    {
        return total_memory_budget_ - get_current_write_buffer_size();
    }
    
    // 打印当前分配状态
    void print_allocation() const
    {
        spdlog::info("=== Memory Allocation ===");
        spdlog::info("  Total Budget: {} MB", total_memory_budget_ >> 20);
        spdlog::info("  Alpha: {:.2f}", current_alpha_);
        spdlog::info("  Write Buffer: {} MB ({:.1f}%)", 
                     get_current_write_buffer_size() >> 20, 
                     current_alpha_ * 100);
        spdlog::info("  Block Cache: {} MB ({:.1f}%)", 
                     get_current_cache_size() >> 20,
                     (1 - current_alpha_) * 100);
    }

private:
    // 调整 Write Buffer 大小
    bool adjust_write_buffer(uint64_t new_buffer_size, uint64_t new_buffer_total)
    {
        if (!db_) return false;
        
        try
        {
            // 使用 SetOptions 动态调整 write_buffer_size
            rocksdb::Status status = db_->SetOptions({
                {"write_buffer_size", std::to_string(new_buffer_size)}
            });
            
            if (!status.ok())
            {
                spdlog::error("Failed to set write_buffer_size: {}", status.ToString());
                return false;
            }
            
            // 同时更新 Compactor 中的 buffer_size（用于容量阈值计算）
            if (compactor_)
            {
                compactor_->updateM(new_buffer_total);
            }
            
            return true;
        }
        catch (const std::exception& e)
        {
            spdlog::error("Exception in adjust_write_buffer: {}", e.what());
            return false;
        }
    }
    
    // 调整 Block Cache 大小
    bool adjust_block_cache(uint64_t new_cache_size)
    {
        if (!block_cache_) return false;
        
        try
        {
            // LRUCache 支持动态调整容量
            block_cache_->SetCapacity(new_cache_size);
            return true;
        }
        catch (const std::exception& e)
        {
            spdlog::error("Exception in adjust_block_cache: {}", e.what());
            return false;
        }
    }
};

} /* namespace tmpdb */

#endif /* MEMORY_ALLOCATOR_H_ */