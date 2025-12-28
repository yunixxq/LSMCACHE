#include "tmpdb/compactor.hpp"
#include "memory_tuner.hpp"

using namespace tmpdb;
using namespace ROCKSDB_NAMESPACE;

// BaseCompactor 构造函数（保持不变）
BaseCompactor::BaseCompactor(const CompactorOptions compactor_opt, const rocksdb::Options rocksdb_opt)
    : compactor_opt(compactor_opt), rocksdb_opt(rocksdb_opt), rocksdb_compact_opt()
{
    this->rocksdb_compact_opt.compression = this->rocksdb_opt.compression;//压缩时使用和DB相同的压缩算法
    this->rocksdb_compact_opt.output_file_size_limit = this->rocksdb_opt.target_file_size_base;// 压缩生成文件的目标大小=RocksDB中配置的target_file_size_base
    this->level_being_compacted = std::vector<bool>(this->rocksdb_opt.num_levels, false);// 标记每一层是否正在被 compaction，用于避免同一层被多次并发压缩
}

// 辅助函数：寻找“最深的非空层”，用于选择Compaction的层
int Compactor::largest_occupied_level(rocksdb::DB *db) const
{
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);
    int largest_level_idx = 0;

    for (size_t level_idx = cf_meta.levels.size() - 1; level_idx > 0; level_idx--)
    {
        if (cf_meta.levels[level_idx].files.empty())
        {
            continue;
        }
        largest_level_idx = level_idx;
        break;
    }
    return largest_level_idx;
}

// 辅助函数：打印每一层有哪些文件（
void print_db_status1(rocksdb::DB *db)
{
    spdlog::debug("Files per level");
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::vector<std::string> file_names;
    int level_idx = 0;
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

// 🌟PickCompaction：核心逻辑(选哪些文件、往哪一层压)
CompactionTask *Compactor::PickCompaction(rocksdb::DB *db,
                                          const std::string &cf_name,
                                          const size_t level_idx)
{
    /*读取当前 level 的文件情况*/
    this->meta_data_mutex.lock(); // 访问元数据时加锁（多线程安全
    size_t T = this->compactor_opt.size_ratio; // ✅

    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::vector<std::string> input_file_names;
    size_t level_size = 0;
    for (auto &file : cf_meta.levels[level_idx].files)
    {
        if (file.being_compacted) // ✅正在压缩的文件跳过
        {
            continue;
        }
        input_file_names.push_back(file.name);
        level_size += file.size;
    }

    // ✅该层没有可压缩的文件，释放锁直接返回
    if (input_file_names.empty())
    {
        this->meta_data_mutex.unlock();
        return nullptr;
    }

    if (level_idx == 0) // L0处理逻辑：固定Compact至L1且实现容量阈值触发
    {
        // L0阈值： (T-1) * Mbuf
        if(level_size <= (T - 1) * this->compactor_opt.buffer_size)
        {
            this->meta_data_mutex.unlock();
            return nullptr;
        }

        // L0压缩至的目标层固定为L1
        int target_lvl = 1;
        spdlog::debug("PickCompaction: L0 -> L{}, files_num={}", 
                      target_lvl, input_file_names.size());
        
        this->meta_data_mutex.unlock();  // ✅ 添加 unlock
        return new CompactionTask(db, this, cf_name, input_file_names, target_lvl,
                                      this->rocksdb_compact_opt, level_idx, false,
                                      false);    
    }
    else //L1+层
    {
        uint64_t level_capacity = static_cast<uint64_t>(
            std::pow(T, level_idx) * (T - 1) * this->compactor_opt.buffer_size
        );

        // 未超过容量阈值，无需触发Compaction
        if (level_size <= level_capacity)
        {
            this->meta_data_mutex.unlock();
            return nullptr;
        }

        // 选择要Compaction的文件直至剩余容量低于阈值
        std::vector<std::string> compact_files;
        size_t compaction_size = 0;

        for (auto &file : cf_meta.levels[level_idx].files)
        {
            if (file.being_compacted)
            {
                continue;
            }
            compact_files.push_back(file.name);
            compaction_size += file.size;
            
            // 如果 compact 这些文件后，剩余容量低于阈值，停止选择
            if ((level_size - compaction_size) <= level_capacity)
            {
                break;
            }
        }

        if (compact_files.empty())
        {
            this->meta_data_mutex.unlock();
            return nullptr;
        }
        
        int target_lvl = level_idx + 1;

        // 检查目标层是否有效
        if (target_lvl >= static_cast<int>(cf_meta.levels.size()))
        {
            this->meta_data_mutex.unlock();
            return nullptr;
        }

        spdlog::debug("PickCompaction: L{} -> L{}, files={}, size={} bytes",
                      level_idx, target_lvl, compact_files.size(), compaction_size);
        this->meta_data_mutex.unlock();
        return new CompactionTask(db, this, cf_name, compact_files, target_lvl,
                                  this->rocksdb_compact_opt, level_idx, false, false);
    }
}

// 🌟OnFlushCompleted：每次flush后尝试对所有层触发compaction 修改：添加flush数量统计信息
// → Compactor::OnFlushCompleted → PickCompaction → ScheduleCompaction
void Compactor::OnFlushCompleted(rocksdb::DB *db, const ROCKSDB_NAMESPACE::FlushJobInfo &info)
{
    // ===== 记录 Flush 统计 ===== 
    stats.total_flush_count++;
    stats.epoch_flush_count++;

    // ✅ 区分flush类型
    // 根据FlushReason判断是memory-triggered还是log-triggered
    switch (info.flush_reason) {
        // 1️⃣ 内存压力触发
        case ROCKSDB_NAMESPACE::FlushReason::kWriteBufferFull:
        case ROCKSDB_NAMESPACE::FlushReason::kWriteBufferManager:
            // 由于 write buffer 满了触发
            stats.memory_triggered_flush_count++;
            stats.epoch_memory_triggered_flush_count++; 
            break;
        
        // 2️⃣ 日志压力触发
        case ROCKSDB_NAMESPACE::FlushReason::kWalFull:
            stats.log_triggered_flush_count++;
            stats.epoch_log_triggered_flush_count++;
            if (memory_tuner_) {
                memory_tuner_->notify_log_triggered_flush();
            }
            break;
        
        // 其他flush原因不计入论文模型中
        default:
            break;
    }

    // 检查每一层是否需要Compaction，从L0开始向下遍历所有非空层
    int largest_level_idx = this->largest_occupied_level(db);

    // int count = 0;
    for (int level_idx = 0; level_idx <= largest_level_idx; level_idx++)
    {
        CompactionTask *task = nullptr;
        task = PickCompaction(db, info.cf_name, level_idx); //尝试为当前 level 选一个 compaction 任务
        if (task != nullptr)
        {
            if (info.triggered_writes_stop)
            {
                task->retry_on_fail = true;
            }
            // Schedule compaction in a different thread.
            ScheduleCompaction(task);
            // count++;
        }
    }

    // // 检查一次flush是否会引发级联的多次Compaction
    // if(count != 0)
    // {
    //     printf("OnFlushCompleted: triggered %d compactions after flush\n", count);
    // }
}

// ✅新增：OnCompactionCompleted 似乎不会自动回调 因为我们使用的是自定义的Compactor(完全禁用原生自动Compaction)
// void Compactor::OnCompactionCompleted(rocksdb::DB *db, const ROCKSDB_NAMESPACE::CompactionJobInfo &info)
// {
//     // ===== 记录 Compaction 统计 =====
//     stats.total_compaction_count++;
//     stats.epoch_compaction_count++;
    
//     stats.total_input_files += info.input_files.size();
//     stats.epoch_input_files += info.input_files.size();
    
//     stats.total_output_files += info.output_files.size();
//     stats.epoch_output_files += info.output_files.size();
    
//     // 记录读写字节数
//     stats.total_compaction_read_bytes += info.stats.total_input_bytes;
//     stats.epoch_compaction_read_bytes += info.stats.total_input_bytes;
    
//     stats.total_compaction_write_bytes += info.stats.total_output_bytes;
//     stats.epoch_compaction_write_bytes += info.stats.total_output_bytes;
    
//     // 记录时间
//     stats.total_compaction_time_us += info.stats.elapsed_micros;
//     stats.epoch_compaction_time_us += info.stats.elapsed_micros;
    
//     // 记录每层统计
//     if (info.output_level < CompactionStats::MAX_LEVELS) {
//         stats.compaction_count_per_level[info.output_level]++;
//     }
    
//     spdlog::debug("Compaction completed: L{} -> L{}, "
//                   "input_files={}, output_files={}, "
//                   "read_bytes={}, write_bytes={}, time={}us",
//                   info.base_input_level, info.output_level,
//                   info.input_files.size(), info.output_files.size(),
//                   info.stats.total_input_bytes, info.stats.total_output_bytes,
//                   info.stats.elapsed_micros);
    
//     // ===== 检查级联 Compaction =====
//     // Compaction 完成后，目标层可能超过容量阈值
//     // 需要检查是否触发新的 Compaction
    
//     // 只检查从output_level开始的层（因为只有这些层可能受影响）
//     int largest_level_idx = this->largest_occupied_level(db);
    
//     for (int level_idx = info.output_level; level_idx <= largest_level_idx; level_idx++)
//     {
//         CompactionTask *task = PickCompaction(db, info.cf_name, level_idx);
//         if (task != nullptr)
//         {
//             spdlog::debug("Cascade compaction triggered: L{} -> L{}",
//                           level_idx, task->output_level);
//             ScheduleCompaction(task);
//         }
//     }
// }

bool Compactor::requires_compaction(rocksdb::DB *db)
{
    int largest_level_idx = this->largest_occupied_level(db);
    bool task_scheduled = false;

    for (int level_idx = 0; level_idx <= largest_level_idx; level_idx++)
    {
        CompactionTask *task = nullptr;
        task = PickCompaction(db, "default", level_idx);
        if (!task)
        {
            continue;
        }
        ScheduleCompaction(task);
        task_scheduled = true;
    }

    return task_scheduled;
}

void Compactor::CompactFiles(void *arg)
{
    std::unique_ptr<CompactionTask> task(reinterpret_cast<CompactionTask *>(arg));
    assert(task);
    assert(task->db);
    assert(task->output_level > (int)task->origin_level_id);

    spdlog::info("CompactFiles starting: L{} -> L{}, files={}",
                  task->origin_level_id, task->output_level,
                  task->input_file_names.size());
    
    // auto start_time = std::chrono::steady_clock::now();
    // 实际执行Compaction：RocksDB内部接口，合并多个SST文件到目标层
    rocksdb::Status s = task->db->CompactFiles(
        task->compact_options,
        task->input_file_names,
        task->output_level);
    
    // auto end_time = std::chrono::steady_clock::now();
    // auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
    //     end_time - start_time).count();
    
    // ✅ 通过task->compactor访问实例成员
    Compactor* compactor = static_cast<Compactor*>(task->compactor);
    
    if(s.ok())
    {
        // ✅ 更新统计信息
        compactor->stats.total_compaction_count++;
        compactor->stats.epoch_compaction_count++;

        compactor->stats.total_input_files += task->input_file_names.size();
        compactor->stats.epoch_input_files += task->input_file_names.size();

        // compactor->stats.total_compaction_time_us += elapsed_us;
        // compactor->stats.epoch_compaction_time_us += elapsed_us;

        // 记录每层统计❓
        if (task->output_level < CompactionStats::MAX_LEVELS) {
            compactor->stats.compaction_count_per_level[task->output_level]++;
        }

        // 级联 Compaction 检查
        int largest_level_idx = compactor->largest_occupied_level(task->db);
        for (int level_idx = task->output_level; level_idx <= largest_level_idx; level_idx++)
        {
            CompactionTask *cascade_task = compactor->PickCompaction(
                task->db, 
                task->column_family_name, 
                level_idx);
                
            if (cascade_task != nullptr)
            {
                spdlog::info("Cascade compaction triggered: L{} -> L{}",
                              level_idx, cascade_task->output_level);
                compactor->ScheduleCompaction(cascade_task);
            }
        }

    }
    else if (!s.ok() && !s.IsIOError() && task->retry_on_fail && !s.IsInvalidArgument())
    {
        // If a compaction task with its retry_on_fail=true failed,
        // try to schedule another compaction in case the reason
        // is not an IO error.

        spdlog::warn("CompactFile L{} -> L{} with {} files did not finish: {}",
                     task->origin_level_id,
                     task->output_level,
                     task->input_file_names.size(),
                     s.ToString());
        CompactionTask *new_task = nullptr;
        new_task = task->compactor->PickCompaction(
            task->db,
            task->column_family_name,
            task->origin_level_id);
        
        if (new_task) 
        {
            new_task->is_a_retry = true;
            compactor->ScheduleCompaction(new_task);
        }
        // new_task->is_a_retry = true;
        // task->compactor->ScheduleCompaction(new_task);
        return;
    }
    else if (!s.ok())
    {
        spdlog::error("CompactFiles failed: L{} -> L{}, status: {}",
                      task->origin_level_id, task->output_level, s.ToString());
    }

    spdlog::trace("CompactFiles L{} -> L{} finished | Status: {}",
                  task->origin_level_id, task->output_level, s.ToString());
    ((Compactor *)task->compactor)->compactions_left_count--;
    return;
}

void Compactor::ScheduleCompaction(CompactionTask *task)
{
    if (!task->is_a_retry)
    {
        this->compactions_left_count++; // 增加"待完成 Compaction"计数
    }
    this->rocksdb_opt.env->Schedule(&Compactor::CompactFiles, task);//使用 RocksDB的Env调度到后台线程池
    return;
}

// 🌟此处的B是指Memtable大小
size_t Compactor::estimate_levels(size_t N, double T, size_t E, size_t B)
{
    if ((N * E) < B)
    {
        spdlog::warn("Number of entries (N = {}) fits in the in-memory buffer, defaulting to 1 level", N);
        return 1;
    }

    size_t estimated_levels = std::ceil(std::log((N * E / B) + 1) / std::log(T));

    return estimated_levels;
}

// (未使用)
size_t Compactor::calculate_full_tree(double T, size_t E, size_t B, size_t L)
{
    int full_tree_size = 0;
    size_t entries_in_buffer = B / E;

    for (size_t level = 1; level < L + 1; level++)
    {
        full_tree_size += entries_in_buffer * (T - 1) * (std::pow(T, level - 1));
    }

    return full_tree_size;
}

void Compactor::updateT(int T)
{
    this->meta_data_mutex.lock();
    this->compactor_opt.size_ratio = T;
    this->meta_data_mutex.unlock();
    return;
}

// buffer size表示的是Memtable的大小
void Compactor::updateM(size_t M)
{
    this->meta_data_mutex.lock();
    this->compactor_opt.buffer_size = M;
    this->meta_data_mutex.unlock();
    return;
}