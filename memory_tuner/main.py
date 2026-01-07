import numpy as np
import pandas as pd
import sys
import logging
import os
import yaml
import copy
import random
import pickle as pkl

sys.path.append("./memory_tuner")
from memory_tuner_runner import Runner
from lsm_tree.PyRocksDB import RocksDB

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("memory_tuner/config/config_memory_tuner.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 共5个工作负载
workloads = [
    (0.00, 0.90, 0.00, 0.10), # writes = 10%
    (0.00, 0.80, 0.00, 0.20), # writes = 20%
    (0.00, 0.70, 0.00, 0.30), # writes = 30%
    (0.00, 0.60, 0.00, 0.40), # writes = 40%
    (0.00, 0.50, 0.00, 0.50), # writes = 50%
]

class Optimizer(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
    

    def run(self):
        i = -1
        df = [] # 记录所有运行结果的row

        for workload in workloads:
            z0, z1, q, w = workload

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
            row["initial_write_memory"] = 67108864 # 64MB 单位:bytes
            self.logger.info(f'Building DB at size: {row["N"]}')

            db = RocksDB(self.config)
            results = db.run(
                row["db_name"],
                row["path_db"],
                row["h"], # 5
                row["T"], # 10
                row["N"], 
                row["E"], # 1024 = 1K 单位:bytes
                row["M"], # 总内存 4GB / 20GB 单位:bytes
                row["initial_write_memory"],
                z0,
                z1,
                q,
                w,
                dist,
                skew,
                row["Q"],
                row["s"],
            )

            for key, val in results.items():
                self.logger.info(f"{key} : {val}")
                row[f"{key}"] = val

            df.append(row)
            pd.DataFrame(df).to_csv(self.config["optimizer_path"]["memory_tuner_ckpt"])

        pd.DataFrame(df).to_csv(self.config["optimizer_path"]["memory_tuner_final"])


if __name__ == "__main__":
    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("memory_tuner/config/config_memory_tuner.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Runner(config)
    driver.run(Optimizer(config))
