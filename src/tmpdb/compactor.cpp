#include "tmpdb/compactor.hpp"

using namespace tmpdb;
using namespace ROCKSDB_NAMESPACE;
BaseCompactor::BaseCompactor(const CompactorOptions compactor_opt, const rocksdb::Options rocksdb_opt)
    : compactor_opt(compactor_opt), rocksdb_opt(rocksdb_opt), rocksdb_compact_opt()
{
    this->rocksdb_compact_opt.compression = this->rocksdb_opt.compression;//压缩时使用和DB相同的压缩算法
    this->rocksdb_compact_opt.output_file_size_limit = this->rocksdb_opt.target_file_size_base;// 压缩生成文件的目标大小=RocksDB中配置的target_file_size_base
    this->level_being_compacted = std::vector<bool>(this->rocksdb_opt.num_levels, false);// 标记每一层是否正在被 compaction，用于避免同一层被多次并发压缩
}
// 寻找“最深的非空层”
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
// debug：打印每一层有哪些文件（辅助函数）
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
    size_t T = this->compactor_opt.size_ratio;
    size_t K = this->compactor_opt.K;
    // int largest_level_idx = this->largest_occupied_level(db);
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::vector<std::string> input_file_names;
    size_t level_size = 0;
    for (auto &file : cf_meta.levels[level_idx].files)
    {
        if (file.being_compacted) // 正在压缩的文件跳过
        {
            continue;
        }
        input_file_names.push_back(file.name);
        level_size += file.size;
    }
    // 这个 level 没有可压缩的文件，直接返回
    if (input_file_names.size() < 1)
    {
        this->meta_data_mutex.unlock();
        return nullptr;
    }

    // Level 0 的处理逻辑：任意合并 + 选择最“空”的目标层
    if (level_idx == 0)
    {
        if (input_file_names.size() >= 1)
        {
            // pick targer output level
            int target_lvl = 1;
            size_t min_size = UINT64_MAX;
            std::vector<size_t> empty_levels;
            for (size_t i = 1; i < compactor_opt.levels - 1; i++)
            {
                size_t lvl_size = 0;
                for (auto &file : cf_meta.levels[i].files)
                {
                    if (file.being_compacted)
                    {
                        lvl_size += this->rocksdb_opt.target_file_size_base;
                        continue;
                    }
                    lvl_size += file.size;
                }
                if (lvl_size < min_size)
                {
                    min_size = lvl_size;
                    target_lvl = i;
                }
                if (lvl_size < min_size + this->rocksdb_opt.target_file_size_base && min_size != 0)
                {
                    empty_levels.push_back(i);
                }
            }
            if (!empty_levels.empty())
            {
                size_t random_index = std::rand() % empty_levels.size();
                target_lvl = empty_levels[random_index];
            }
            // pick input file
            std::vector<std::string> compact_files;
            for (auto &file : cf_meta.levels[0].files)
            {
                if (file.being_compacted)
                {
                    continue;
                }
                compact_files.push_back(file.name);
            }
            this->meta_data_mutex.unlock();
            if (compact_files.empty())
                return nullptr;
            return new CompactionTask(db, this, cf_name, compact_files, target_lvl,
                                      this->rocksdb_compact_opt, level_idx, false,
                                      false);
        }
    }
    else //L1+层
    {
        for (size_t i = 0; i < (this->compactor_opt.levels / K); i++)
        {
            if (i * K < level_idx && level_idx <= (i + 1) * K)
            {
                // get input level size
                size_t lvl_size = 0;
                for (auto &file : cf_meta.levels[level_idx].files)
                {
                    if (file.being_compacted)
                    {
                        continue;
                    }
                    lvl_size += file.size;
                }
                if (lvl_size < pow(T, i) * this->compactor_opt.buffer_size / K)
                    continue;
                std::vector<std::string> compact_files;
                std::vector<rocksdb::SstFileMetaData> compact_files_meta;
                size_t compaction_size = 0;
                // bool flag = false;
                for (auto &file : cf_meta.levels[level_idx].files)
                {
                    if (file.being_compacted)
                    {
                        continue;
                    }
                    compact_files.push_back(file.name);
                    compact_files_meta.push_back(file);
                    compaction_size += file.size;
                    if ((lvl_size - compaction_size) <
                        pow(T, i) * this->compactor_opt.buffer_size / K)
                        break;
                }
                // pick target output level
                int target_lvl = (i + 1) * K + 1;
                std::vector<size_t> empty_levels;
                size_t min_size = UINT64_MAX;
                for (size_t j = (i + 1) * K + 1; j < compactor_opt.levels - 1;
                     j++)
                {
                    size_t lvl_size = 0;
                    for (auto &file : cf_meta.levels[j].files)
                    {
                        if (file.being_compacted)
                        {
                            lvl_size += this->rocksdb_opt.target_file_size_base;
                            continue;
                        }
                        lvl_size += file.size;
                    }
                    if (lvl_size < min_size)
                    {
                        min_size = lvl_size;
                        target_lvl = j;
                    }
                    if (lvl_size < min_size + this->rocksdb_opt.target_file_size_base && min_size != 0)
                    {
                        empty_levels.push_back(j);
                    }
                }
                if (!empty_levels.empty())
                {
                    size_t random_index = std::rand() % empty_levels.size();
                    target_lvl = empty_levels[random_index];
                }
                this->meta_data_mutex.unlock();
                if (compact_files.empty())
                    return nullptr;
                return new CompactionTask(db, this, cf_name, compact_files, target_lvl,
                                          this->rocksdb_compact_opt, level_idx, false,
                                          false);
            }
        }
    }
    this->meta_data_mutex.unlock();
    return nullptr;
}

// 每次flush后尝试对所有层触发compaction
// → Compactor::OnFlushCompleted → PickCompaction → ScheduleCompaction
void Compactor::OnFlushCompleted(rocksdb::DB *db, const ROCKSDB_NAMESPACE::FlushJobInfo &info)
{
    int largest_level_idx = this->largest_occupied_level(db);
    //选择从 L0 向下遍历所有非空层
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
        }
    }
    // this->requires_compaction(db);
    return;
}

//手动触发一轮全面扫描
bool Compactor::requires_compaction(rocksdb::DB *db)
{
    // this->meta_data_mutex.lock();
    int largest_level_idx = this->largest_occupied_level(db);
    // this->meta_data_mutex.unlock();
    bool task_scheduled = false;

    // for (int level_idx = largest_level_idx; level_idx > -1; level_idx--)
    for (int level_idx = 0; level_idx <= largest_level_idx; level_idx++)
    {
        CompactionTask *task = nullptr;
        task = PickCompaction(db, "default", level_idx);
        if (!task)
        {
            continue;
        }
        // spdlog::info("req compaction from {} to {}", task->origin_level_id, task->output_level);
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

    // 调 RocksDB 内部接口，合并多个 SST 文件到目标层
    rocksdb::Status s = task->db->CompactFiles(
        task->compact_options,
        task->input_file_names,
        task->output_level);

    if (!s.ok() && !s.IsIOError() && task->retry_on_fail && !s.IsInvalidArgument())
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

        new_task->is_a_retry = true;
        task->compactor->ScheduleCompaction(new_task);
        return;
    }

    spdlog::trace("CompactFiles L{} -> L{} finished | Status: {}",
                  task->origin_level_id, task->output_level, s.ToString());
    ((Compactor *)task->compactor)
        ->compactions_left_count--;
    return;
}

void Compactor::ScheduleCompaction(CompactionTask *task)
{
    if (!task->is_a_retry)
    {
        this->compactions_left_count++;
    }
    this->rocksdb_opt.env->Schedule(&Compactor::CompactFiles, task);//使用 RocksDB 的 Env 调度到后台线程池
    return;
}

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

//运行时更新压缩阈值参数
void Compactor::updateM(size_t M)
{
    this->meta_data_mutex.lock();
    this->compactor_opt.buffer_size = M;
    this->meta_data_mutex.unlock();
    return;
}