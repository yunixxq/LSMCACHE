import numpy as np
import pandas as pd
import sys
import logging
import os
import yaml
import copy
import random
import pickle as pkl

sys.path.append("./lsmcache")
from lsmcache_runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import CostFunction
from utils import model_xgb

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("lsmcache/config/config_lsm_cache.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 共5个工作负载(参考Breaking walls)
workloads = [
    # (0.00, 0.90, 0.00, 0.10), # writes = 10%
    (0.00, 0.80, 0.00, 0.20), # writes = 20%
    (0.00, 0.70, 0.00, 0.30), # writes = 30%
    # (0.00, 0.60, 0.00, 0.40), # writes = 40%
    # (0.00, 0.50, 0.00, 0.50), # writes = 50%
]

# ratio 的候选值 总实验次数：5 * 17 = 85
# ratios = np.arange(0.1, 0.91, 0.05)
ratios = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
          0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

class Optimizer(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
    
    def compaction_aware_initial_ratio(z0, z1, q, w, M, N, E):
        memory_data_ratio = M / N * E

        read_ratio = z0 + z1 + q
        write_ratio = w

        # 基础ratio：根据读写比例线性插值
        # 纯读(w=0) → ratio=0.1 (90%给cache)
        # 纯写(w=1) → ratio=0.7 (70%给buffer)
        base_ratio = 0.1 + 0.6 * write_ratio

        if memory_data_ratio < 0.15:
            # 内存紧张时，写密集负载更需要buffer
            adjustment = 0.1 * write_ratio
            base_ratio += adjustment

        return np.clip(base_ratio, 0.1, 0.9)

    def run(self):
        i = -1
        df = [] # 记录所有运行结果的row
        xgb_t = [] # 延迟结果
        xgb_h = [] # 命中率结果

        for workload in workloads:
            z0, z1, q, w = workload

            for ratio in ratios:
                i += 1 # 第一个workload对应i=0
                dist = "zipf"
                skew = 0.99

                row = self.config["lsm_tree_config"].copy()
                row["db_name"] = "level_test"
                row["path_db"] = self.config["app"]["DATABASE_PATH"]
        
                row["dist"] = dist
                row["skew"] = skew
                row["z0"] = z0
                row["z1"] = z1
                row["q"] = q
                row["w"] = w

                key_path = "key_log_xgb_optimizer"
                if not os.path.exists(key_path):
                    os.makedirs(key_path)
                key_log = key_path + "/{}.dat".format(i)
                row["key_log"] = key_log
                self.logger.info(f'Building DB at size: {row["N"]}')

                best_ratio = ratio
                # best_ratio = self.compaction_aware_initial_ratio(z0, z1, q, w, row["M"] / 8, row["N"], row["E"] / 8, T=10)

                print(f"启发式方法静态最优参数: best_ratio: {best_ratio}")

                row["T"] = 10
                row["h"] = 5 # bpe
                row["ratio"] = best_ratio
                row["Mbuf"] = best_ratio * row["M"] / 8 # bytes
                row["Mcache"] = (1 - best_ratio) * row["M"] / 8 # bytes

                db = RocksDB(self.config)
                results = db.run(
                    row["db_name"],
                    row["path_db"],
                    row["h"], # 最佳h
                    row["T"], # 最佳T
                    row["N"], 
                    row["E"],
                    row["M"],
                    row["Mbuf"], # 根据最佳h和ratio计算得到的写内存大小
                    row["Mcache"],
                    z0,
                    z1,
                    q,
                    w,
                    dist,
                    skew,
                    row["Q"],
                    row["s"],
                    key_log=key_log,
                    # 动态调参相关参数
                    enable_dynamic_tuning=False,
                    initial_alpha=row["ratio"],
                    enable_epoch_log=True,
                )

                for key, val in results.items():
                    self.logger.info(f"{key} : {val}")
                    row[f"{key}"] = val

                df.append(row)
                pd.DataFrame(df).to_csv(self.config["optimizer_path"]["lsmcache_ckpt"])
                xgb_t.append(row["total_latency"])
                xgb_h.append(row["cache_hit_rate"])

                # 平均latency
                print("xgb_t: ", np.mean(xgb_t))
                
                # 平均命中率
                print("xgb_h: ", np.mean(xgb_h))

        pd.DataFrame(df).to_csv(self.config["optimizer_path"]["lsmcache_final"])
        self.logger.info("Finished optimizer\n")


if __name__ == "__main__":
    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("lsmcache/config/config_lsm_cache.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Runner(config)
    driver.run(Optimizer(config))
