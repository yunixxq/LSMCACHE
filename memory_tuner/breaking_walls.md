# Memory Tuner 复现说明文档

## 一、论文核心思想回顾

### 1.1 问题背景

论文《Breaking Down Memory Walls》(VLDB 2021) 解决的核心问题是：
- **在固定总内存预算 M 下，如何动态调整 Write Memory（Mwrite）和 Buffer Cache（Mcache）的分配比例，以最小化总 I/O 成本？**

传统静态分配方案的问题：
- RocksDB 默认为每个 MemTable 分配固定 64MB
- AsterixDB 静态划分 write memory 和 buffer cache
- 不能适应工作负载变化

### 1.2 Memory Tuner 的设计思路

论文采用**白盒分析方法**，通过数学建模计算成本函数的导数 `cost'(x)`：

```
cost(x) = ω · write(x) + γ · read(x)

其中:
- x = write memory 大小
- ω = 写成本权重（SSD上通常设为2，因为写比读贵）
- γ = 读成本权重（通常设为1）
```

**关键洞察**：
- 如果 `cost'(x) > 0`：增加 write memory 会增加成本 → 应减少 write memory
- 如果 `cost'(x) < 0`：增加 write memory 会减少成本 → 应增加 write memory
- 如果 `cost'(x) = 0`：达到最优分配

---

## 二、核心公式详解

### 2.1 写成本导数 write'(x)

**论文公式 (4)**：
```
write'_i(x) = -merge_i(x) / (x · ln(|L_N_i| / (a_i · x)))
```

**论文公式 (5)**：
```
write'(x) = Σ write'_i(x) · scale_factor

scale_factor = flush_mem / (flush_mem + flush_log)
```

**参数说明**：
- `merge_i(x)`: 每操作的 merge 写入页数 (pages/op)
- `x`: 当前 write memory 大小
- `|L_N_i|`: 第 i 个 LSM-tree 最后一层的大小
- `a_i`: 第 i 个 LSM-tree 占总 write memory 的比例
- `scale_factor`: 用于修正 log-triggered flush 的影响

**直观理解**：
- 分母中的 `ln(|L_N|/x)` 反映了 LSM-tree 的层数
- write memory 越大，层数越少，merge 成本越低
- 所以 `write'(x)` 通常为负值

### 2.2 读成本导数 read'(x)

**论文公式 (6)**：
```
read'(x) = (saved_q + saved_m) / sim + write'(x) · read_m(x) / merge(x)
```

**参数说明**：
- `saved_q`: SimCache 报告的可节省的查询读取 (pages/op)
- `saved_m`: SimCache 报告的可节省的 merge 读取 (pages/op)
- `sim`: SimCache 大小
- `read_m(x)`: 每操作的 merge 读取页数
- `merge(x)`: 每操作的 merge 写入页数

**直观理解**：
- 第一项 `(saved_q + saved_m) / sim` 反映了增加 buffer cache 能节省多少读取
- 第二项反映了减少 write memory 对 merge 读取的影响

### 2.3 SimCache 的作用

**SimCache**（模拟缓存）是论文的关键技术：
- 它只存储页面 ID，不存储实际数据
- 用于模拟"如果我们有更大的 cache，能节省多少 I/O"
- RocksDB 原生支持 SimCache: `rocksdb::NewSimCache()`

工作原理：
1. 当页面被从实际 cache 中驱逐时，其 ID 被添加到 SimCache
2. 当需要从磁盘读取页面时，检查 SimCache
3. 如果 SimCache 命中，说明"如果 cache 更大，这次读取可以避免"

---

## 三、实现架构

### 3.1 类图

```
┌─────────────────────────────────────────────────────────────┐
│                      MemoryTuner                            │
├─────────────────────────────────────────────────────────────┤
│ - db_: DB*                                                  │
│ - statistics_: Statistics*                                  │
│ - block_cache_: shared_ptr<Cache>                          │
│ - total_memory_: size_t                                     │
│ - current_write_memory_: atomic<size_t>                    │
│ - sim_cache_collector_: unique_ptr<SimCacheCollector>      │
│ - cost_estimator_: unique_ptr<CostDerivativeEstimator>     │
│ - tuning_history_: deque<TuningHistoryEntry>               │
├─────────────────────────────────────────────────────────────┤
│ + performTuning(op_count): bool                            │
│ + notifyOperation()                                         │
│ + notifyWrite(entry_count)                                 │
│ + notifyFlush(is_log_triggered, bytes)                     │
│ + getCurrentWriteMemory(): size_t                          │
│ + getCurrentBufferCache(): size_t                          │
└─────────────────────────────────────────────────────────────┘
              │                    │
              ▼                    ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│  SimCacheCollector   │  │  CostDerivativeEstimator     │
├──────────────────────┤  ├──────────────────────────────┤
│ - sim_cache_         │  │ - config_                    │
│ - capacity_          │  ├──────────────────────────────┤
├──────────────────────┤  │ + estimateWriteCostDerivative│
│ + getSimCache()      │  │ + estimateReadCostDerivative │
│ + computeSavedReads()│  │ + computeTotalCostDerivative │
└──────────────────────┘  └──────────────────────────────┘
```

### 3.2 调优流程

```
┌─────────────────────────────────────────────────────────────┐
│                    performTuning() 流程                     │
└─────────────────────────────────────────────────────────────┘

Step 1: 收集统计信息
         │
         ▼
    ┌─────────────┐
    │ collectStats │ → merge_writes_per_op, saved_q, flush_mem/log
    └─────────────┘
         │
         ▼
Step 2: 计算成本导数
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│write'(x)│ │read'(x)│
└────────┘ └────────┘
    │         │
    └────┬────┘
         ▼
    ┌─────────────┐
    │  cost'(x)   │ = ω·write'(x) + γ·read'(x)
    └─────────────┘
         │
         ▼
Step 3: Newton-Raphson 优化
         │
    ┌─────────────────────┐
    │ 线性拟合 cost'(x)=Ax+B │
    └─────────────────────┘
         │
         ▼
    ┌─────────────────────┐
    │ x_new = x - cost'(x)/A │
    └─────────────────────┘
         │
         ▼
Step 4: 应用约束
         │
    ┌─────────────────────┐
    │ - 最大步长限制        │
    │ - 内存边界限制        │
    │ - 停止条件检查        │
    └─────────────────────┘
         │
         ▼
Step 5: 应用新分配
         │
    ┌─────────────────────┐
    │ - 更新 write_memory  │
    │ - 调整 cache capacity │
    └─────────────────────┘
```

---

## 四、与您现有代码的集成

### 4.1 需要修改的文件

1. **db_runner.cpp**
2. **compactor.hpp/cpp** (可选：添加 flush 通知)
3. **CMakeLists.txt** (添加新文件)

### 4.2 关键修改点

#### 4.2.1 Step 4: 配置 Block Cache

**原代码**：
```cpp
std::shared_ptr<Cache> block_cache = nullptr;
if (env.cache_cap == 0)
    table_options.no_block_cache = true;
else {
    block_cache = rocksdb::NewLRUCache(env.cache_cap);
    table_options.block_cache = block_cache;
}
```

**新代码**：
```cpp
std::shared_ptr<Cache> block_cache = nullptr;
std::shared_ptr<WriteBufferManager> write_buffer_manager = nullptr;

if (env.enable_memory_tuner && env.M > 0) {
    // 计算初始分配
    size_t initial_write_memory = static_cast<size_t>(env.M * env.initial_alpha);
    size_t initial_cache_size = env.M - initial_write_memory;
    
    // 创建 Block Cache
    block_cache = rocksdb::NewLRUCache(initial_cache_size);
    table_options.block_cache = block_cache;
    
    // 创建 WriteBufferManager（用于控制总 MemTable 内存）
    write_buffer_manager = std::make_shared<WriteBufferManager>(
        initial_write_memory, block_cache);
    rocksdb_opt.write_buffer_manager = write_buffer_manager;
}
```

#### 4.2.2 Step 6: 创建 Memory Tuner

**替换您原有的组件**：
```cpp
// 替换原有的:
// tmpdb::PerformanceMonitor* perf_monitor = nullptr;
// tmpdb::DecisionEngine* decision_engine = nullptr;
// tmpdb::MemoryAllocator* memory_allocator = nullptr;

// 使用新的 Memory Tuner:
tmpdb::MemoryTuner* memory_tuner = nullptr;

if (env.enable_memory_tuner) {
    tmpdb::MemoryTunerConfig tuner_config;
    tuner_config.omega = 2.0;  // SSD 写成本权重
    tuner_config.gamma = 1.0;  // 读成本权重
    tuner_config.sim_cache_size = 128 * 1024 * 1024;  // 128MB SimCache
    tuner_config.size_ratio = env.T;
    
    size_t initial_write_memory = static_cast<size_t>(env.M * env.initial_alpha);
    
    memory_tuner = new tmpdb::MemoryTuner(
        db,
        rocksdb_opt.statistics.get(),
        block_cache,
        env.M,
        initial_write_memory,
        tuner_config
    );
    
    memory_tuner->printStatus();
}
```

#### 4.2.3 主循环中的修改

```cpp
for (size_t i = 0; i < env.steps; i++) {
    // ... 原有操作代码 ...
    
    // 通知 Memory Tuner
    if (memory_tuner) {
        memory_tuner->notifyOperation();
        
        if (outcome == 3) {  // Write 操作
            memory_tuner->notifyWrite(1);
        }
    }
    
    // Epoch 结束时调优
    if (memory_tuner && ((i + 1) % env.epoch_size == 0)) {
        epoch_count++;
        
        spdlog::info("--- Epoch {} ---", epoch_count);
        
        bool adjusted = memory_tuner->performTuning(env.epoch_size);
        
        if (adjusted) {
            total_adjustments++;
            
            // 更新 WriteBufferManager
            size_t new_write_memory = memory_tuner->getCurrentWriteMemory();
            if (write_buffer_manager) {
                write_buffer_manager->SetBufferSize(new_write_memory);
            }
            
            // 更新 Compactor 的 buffer_size
            compactor->updateM(new_write_memory);
        }
        
        memory_tuner->printStatus();
    }
}
```

---

## 五、与您原有方案的对比

| 方面 | 您的原有方案 (DecisionEngine) | 论文方案 (Memory Tuner) |
|------|------------------------------|------------------------|
| **决策依据** | 基于阈值的启发式规则 | 基于数学模型的成本导数 |
| **优化目标** | cache hit rate, write stall rate | 最小化总 I/O 成本 |
| **调整策略** | 固定步长 (alpha_step) | Newton-Raphson 自适应步长 |
| **缓存分析** | 无 | SimCache 模拟不同大小的命中率 |
| **收敛速度** | 较慢，可能震荡 | 快速收敛（使用导数信息） |
| **理论基础** | 经验规则 | LSM-tree 成本模型 |

---

## 六、RocksDB 相关 API 说明

### 6.1 SimCache

```cpp
// 创建 SimCache
auto base_cache = rocksdb::NewLRUCache(capacity);
auto sim_cache = rocksdb::NewSimCache(base_cache, sim_capacity, 0);

// 获取统计
uint64_t hit_count = sim_cache->get_hit_counter();
uint64_t miss_count = sim_cache->get_miss_counter();

// 重置统计
sim_cache->reset_counter();
```

### 6.2 Statistics

```cpp
// 创建统计对象
rocksdb_opt.statistics = rocksdb::CreateDBStatistics();

// 获取统计值
uint64_t compact_read = statistics->getTickerCount(rocksdb::COMPACT_READ_BYTES);
uint64_t compact_write = statistics->getTickerCount(rocksdb::COMPACT_WRITE_BYTES);
uint64_t flush_write = statistics->getTickerCount(rocksdb::FLUSH_WRITE_BYTES);
uint64_t cache_hit = statistics->getTickerCount(rocksdb::BLOCK_CACHE_HIT);
uint64_t cache_miss = statistics->getTickerCount(rocksdb::BLOCK_CACHE_MISS);
```

### 6.3 WriteBufferManager

```cpp
// 创建 WriteBufferManager
auto wbm = std::make_shared<WriteBufferManager>(buffer_size, cache);
rocksdb_opt.write_buffer_manager = wbm;

// 动态调整大小
wbm->SetBufferSize(new_size);
```

### 6.4 Cache

```cpp
// 创建 LRU Cache
auto cache = rocksdb::NewLRUCache(capacity);

// 动态调整容量
cache->SetCapacity(new_capacity);

// 获取当前状态
size_t capacity = cache->GetCapacity();
size_t usage = cache->GetUsage();
```

---

## 七、注意事项

1. **SimCache 的开销**：SimCache 虽然不存储数据，但仍有一定内存开销，建议设置为总内存的 5-10%

2. **调优频率**：论文建议在日志截断时触发调优，或使用定时器（默认10分钟）

3. **稳定性保护**：
   - 限制单次调整的最大步长（默认10%）
   - 设置最小步长阈值（默认32MB）
   - 成本变化阈值检查

4. **log-triggered flush**：如果 flush 主要由日志截断触发，增加 write memory 不会减少写成本

5. **多 LSM-tree 场景**：论文中的公式支持多个 LSM-tree，需要分别统计每个树的参数

---

## 八、实验建议

1. **基线对比**：
   - 静态 50-50 分配
   - 静态 64MB write memory
   - 您原有的 DecisionEngine

2. **评估指标**：
   - 总 I/O 成本 (写字节 + 读字节)
   - 吞吐量 (ops/s)
   - 调优收敛时间

3. **工作负载**：
   - Write-heavy (50% writes)
   - Read-heavy (5% writes)
   - 混合负载
   - 负载变化场景
