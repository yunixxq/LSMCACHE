#ifndef ROCKSDB_FILE_CACHE_TRACKER_PUBLIC_H_
#define ROCKSDB_FILE_CACHE_TRACKER_PUBLIC_H_

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rocksdb/cache.h"
#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

// 缓存失效统计
struct CacheInvalidationStats {
    std::atomic<uint64_t> total_tracked_inserts{0};
    std::atomic<uint64_t> total_invalidated_entries{0};
    std::atomic<uint64_t> total_invalidated_bytes{0};
    std::atomic<uint64_t> total_invalidated_files{0};

    std::atomic<uint64_t> epoch_tracked_inserts{0};
    std::atomic<uint64_t> epoch_invalidated_entries{0};
    std::atomic<uint64_t> epoch_invalidated_bytes{0};
    std::atomic<uint64_t> epoch_invalidated_files{0};
    std::chrono::steady_clock::time_point epoch_start_time{std::chrono::steady_clock::now()};

    CacheInvalidationStats() {
        epoch_start_time = std::chrono::steady_clock::now();
    }

    void ResetEpoch() {
        epoch_tracked_inserts = 0;
        epoch_invalidated_entries = 0;
        epoch_invalidated_bytes = 0;
        epoch_invalidated_files = 0;
        epoch_start_time = std::chrono::steady_clock::now();
    }

    double GetEpochDurationSeconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - epoch_start_time).count();
    }
    
    // 用于五阶段耦合链路中的I_inv部分的计算
    // 失效率（条目/秒）
    double GetEpochInvalidationRate() const {
        double duration = GetEpochDurationSeconds();
        return duration > 0 ? 
            static_cast<double>(epoch_invalidated_entries.load()) / duration : 0.0;
    }
    
    // 失效字节率（字节/秒）
    double GetEpochInvalidationByteRate() const {
        double duration = GetEpochDurationSeconds();
        return duration > 0 ? 
            static_cast<double>(epoch_invalidated_bytes.load()) / duration : 0.0;
    }
};

// 单个文件失效事件记录
struct FileInvalidationEvent {
  uint64_t file_number;
  size_t tracked_entries;        // 该文件曾跟踪的 cache key 数
  size_t invalidated_entries;    // 实际仍在 cache 中、被失效的条目数
  size_t invalidated_bytes;      // 失效的字节数
  std::chrono::steady_clock::time_point time;
};

// 单个SST文件的缓存条目信息
struct FileCacheInfo {
    uint64_t file_number;  // SST文件编号
    std::unordered_set<std::string> cache_keys;  // 该文件的所有cache keys
    size_t total_charge;                         // 总缓存大小
    std::chrono::steady_clock::time_point first_insert_time; // 首次插入时间(该SST文件的第一个Block被插入Cache的时间)
    
    FileCacheInfo() : file_number(0), total_charge(0) {
        first_insert_time = std::chrono::steady_clock::now();
    }
};

// 文件-缓存条目跟踪器，
// 在Insert时显式传入file_number；维护file_number -> cache_keys的精确映射；在文件删除时主动失效相关缓存条目
class FileCacheTracker {
public:
    FileCacheTracker() : enabled_(true) {}
    
    // 启用/禁用跟踪
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_; }
    
    // ✅ 核心函数-跟踪插入，调用时机：Block Cache Insert成功后 
    void TrackEntry(uint64_t file_number, const Slice& cache_key, size_t charge) {
        if (!enabled_) return;
        
        std::string key_str = cache_key.ToString();
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // 添加到 file -> keys 映射
            auto& file_info = file_cache_map_[file_number];
            if (file_info.file_number == 0) {
                file_info.file_number = file_number;
                file_info.first_insert_time = std::chrono::steady_clock::now();
            }
            file_info.cache_keys.insert(key_str);
            file_info.total_charge += charge;
            
            // 添加到 key -> (file, charge) 映射（用于Untrack）
            key_info_map_[key_str] = {file_number, charge};
        }
        
        stats_.total_tracked_inserts++;
        stats_.epoch_tracked_inserts++;
    }

    // 当条目从Cache中移除时调用，即LRU淘汰时
    void UntrackEntry(const Slice& cache_key) {
        if (!enabled_) return;
        
        std::string key_str = cache_key.ToString();
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = key_info_map_.find(key_str);
        if (it == key_info_map_.end()) return;
        
        uint64_t file_number = it->second.first;
        auto file_it = file_cache_map_.find(file_number);
        if (file_it != file_cache_map_.end()) {
            file_it->second.cache_keys.erase(key_str);
        }
        
        key_info_map_.erase(it);
    }
    
    // 在Compaction删除SST文件时其对应的Cache文件失效
    size_t InvalidateFile(uint64_t file_number, Cache* cache) {
        if (!enabled_ || cache == nullptr) return 0;
        
        std::vector<std::string> keys_to_check;
        // Step1. 获取该文件的所有cache keys
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = file_cache_map_.find(file_number);
            if (it == file_cache_map_.end()) {
                return 0;
            }

            // ⚠️ 如果文件存在但没有cache keys，也应该提前返回并清理文件映射
            if (it->second.cache_keys.empty()) {
                file_cache_map_.erase(it);
                return 0;
            }

            keys_to_check.reserve(it->second.cache_keys.size());
            for (const auto& key : it->second.cache_keys) {
                keys_to_check.push_back(key);            
            }
        }

        // Step2. 检查失效的key哪些是实际仍然存在于cache中的(即未被淘汰)
        std::vector<std::string> actually_cached_keys;
        size_t total_charge = 0;
        for (const auto& key : keys_to_check) {
            Cache::Handle* handle = cache->Lookup(Slice(key)); // 会增加引用计数
            if (handle != nullptr) {
                // ✅ 条目仍在 cache 中，计入失效统计
                actually_cached_keys.push_back(key);
                total_charge += cache->GetCharge(handle);
                cache->Release(handle); // 减少引用计数
            }
            // 如果 handle == nullptr，说明已被 LRU 淘汰，不计入失效统计
        }

        // Step3. 清理内部映射
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // 从key_info_map_移除所有该文件的keys(无论是否仍在cache)
            for (const auto& key : keys_to_check) {
                key_info_map_.erase(key);
            }
            
            // 移除文件映射
            file_cache_map_.erase(file_number);
        }

        size_t invalidated_count = actually_cached_keys.size();

        // 更新统计
        stats_.total_invalidated_entries += invalidated_count;
        stats_.total_invalidated_bytes += total_charge;
        stats_.total_invalidated_files++;
        stats_.epoch_invalidated_entries += invalidated_count;
        stats_.epoch_invalidated_bytes += total_charge;
        stats_.epoch_invalidated_files++;
        
        if (invalidated_count > 0) {
            FileInvalidationEvent ev;
            ev.file_number = file_number;
            ev.tracked_entries = keys_to_check.size();
            ev.invalidated_entries = invalidated_count;
            ev.invalidated_bytes = total_charge;
            ev.time = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lock(mutex_);
                invalidation_events_.push_back(std::move(ev));
            }
        }

        return invalidated_count;
    }

    // 批量失效多个文件(一次Compaction可能删除多个SST文件)
    size_t InvalidateFiles(const std::vector<uint64_t>& file_numbers, Cache* cache) {
        size_t total = 0;
        for (uint64_t fn : file_numbers) {
            total += InvalidateFile(fn, cache);
        }
        return total;
    }
    
    // 获取某个文件当前缓存的条目数(Block数)
    size_t GetCachedEntryCount(uint64_t file_number) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = file_cache_map_.find(file_number);
        return (it != file_cache_map_.end()) ? it->second.cache_keys.size() : 0;
    }
    
    // 获取某个文件当前缓存的字节数
    size_t GetCachedBytes(uint64_t file_number) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t total = 0;
        auto it = file_cache_map_.find(file_number);
        if (it != file_cache_map_.end()) {
            for (const auto& key : it->second.cache_keys) {
                auto key_it = key_info_map_.find(key);
                if (key_it != key_info_map_.end()) {
                    total += key_it->second.second;
                }
            }
        }
        return total;
    }
    
    // 获取当前跟踪的文件数量 
    // 只要一个SST文件的其中一个Block进入Cache，该SST文件就会被跟踪
    size_t GetTrackedFileCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return file_cache_map_.size();
    }
    
    // 获取当前跟踪的条目数量
    size_t GetTrackedEntryCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return key_info_map_.size();
    }

    // 获取所有已记录的失效事件（只读快照）
    std::vector<FileInvalidationEvent> GetInvalidationEvents() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return invalidation_events_;
    }

    // 清空事件记录（例如在 epoch 切换时）
    void ClearInvalidationEvents() {
        std::lock_guard<std::mutex> lock(mutex_);
        invalidation_events_.clear();
    }

    
    // ✅ 获取统计信息 用于耦合理论验证
    const CacheInvalidationStats& GetStats() const { return stats_; }
    CacheInvalidationStats& GetMutableStats() { return stats_; }

    // 重置epoch统计
    void ResetEpochStats() { stats_.ResetEpoch(); }
    
    // 清空所有跟踪数据（用于测试）
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        file_cache_map_.clear();
        key_info_map_.clear();
    }
    
    // 打印调试信息
    void PrintDebugInfo() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        printf("=== FileCacheTracker Debug Info ===\n");
        printf("Enabled: %s\n", enabled_ ? "true" : "false");
        printf("Tracked files: %zu\n", file_cache_map_.size());
        printf("Tracked entries: %zu\n", key_info_map_.size());
        printf("Total tracked inserts: %lu\n", stats_.total_tracked_inserts.load());
        printf("Total invalidated entries: %lu\n", stats_.total_invalidated_entries.load());
        printf("Total invalidated bytes: %lu\n", stats_.total_invalidated_bytes.load());
        printf("Total invalidated files: %lu\n", stats_.total_invalidated_files.load());
        printf("Epoch invalidation rate: %.2f entries/sec\n", stats_.GetEpochInvalidationRate());
        printf("===================================\n");
    }

private:
    bool enabled_;
    mutable std::mutex mutex_;
    // file_number -> 该文件的缓存信息
    std::unordered_map<uint64_t, FileCacheInfo> file_cache_map_;
    // cache_key -> (file_number, charge)
    std::unordered_map<std::string, std::pair<uint64_t, size_t>> key_info_map_;
    CacheInvalidationStats stats_;
    // 最近的文件失效事件
    std::vector<FileInvalidationEvent> invalidation_events_;
};

}  // namespace ROCKSDB_NAMESPACE

#endif  // ROCKSDB_FILE_CACHE_TRACKER_PUBLIC_H_
