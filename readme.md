### exp1. 实验
本实验使用的是小参数配置，此时总内存预算较小，属于一种边界条件，五种读写比的情况下均表现出似均是在alpha最小的时候表现出的最高的命中率。
```yaml
lsm_tree_config:
    M: 33554432 # 总内存预算 32MB 32 * 2^20(1M)
    N: 100000  # entry 数量 10 万条记录 ≈ 100MB 数据
    Q: 1500000 # 查询数量 150 万
    s: 4 # range query selectivity
    E: 1024    # bits/entry  1 KB
    T: 5                 # Size ratio
    h: 5                 # Bloom filter bits per element
```
### exp2. 实验
本实验使用大参数配置。
```yaml
lsm_tree_config:
    E: 1024              # Entry size: 1KB bytes/entry
    M: 1073741824        # Total memory: 1GB bytes (可调整为 4GB = 4294967296)
    N: 10000000          # 初始化 entry 数量: 1000万条记录 (~10GB 数据)
    Q: 10000000           # 查询数量: 1000万 (每个 alpha 值)
    s: 4                 # Range query selectivity
    T: 5                 # Size ratio
    h: 5                 # Bloom filter bits per element
```
### offline目录下
* sampling
* train


### online 部分实现
* Warm Start
* RL Fine-tune
* Drift Detection

### 对于CAMAL & CALM: 为确保公平性我们为二者使用相同的采样集
* skew = [0.7, 0.8, 0.9, 0.99]; workload = 5种; alpha = [0.1, 0.9];
* (4 * 5 * 9 = 180) M = 1GB; N = 1000w; Q = 1000w; T = 5;   sampling_1
* (4 * 5 * 9 = 180) M = 512MB; N = 700w; Q = 700w; T = 5;   sampling_2 
* (4 * 5 * 9 = 324) M = 512MB; N = 500w; Q = 500w; T = 6    sampling_3
* (4 * 9 * 9 = 324)workload = 9种 M = 32MB; N = 10w; Q = 150w; T = 5   sampling_4