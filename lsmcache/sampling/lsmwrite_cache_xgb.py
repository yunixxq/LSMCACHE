import logging
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import copy
import sys
import os
import yaml

sys.path.append("./lsmcache")
from lsmcache_runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from utils.model_xgb import traverse_for_T, traverse_for_h, iter_model
from utils.lsm import *

# 共15个
workloads = [
    (0.25, 0.25, 0.25, 0.25),
    (0.97, 0.01, 0.01, 0.01),
    (0.01, 0.97, 0.01, 0.01),
    (0.01, 0.01, 0.97, 0.01),
    (0.01, 0.01, 0.01, 0.97),
    (0.49, 0.49, 0.01, 0.01),
    (0.49, 0.01, 0.49, 0.01),
    (0.49, 0.01, 0.01, 0.49),
    (0.01, 0.49, 0.49, 0.01),
    (0.01, 0.49, 0.01, 0.49),
    (0.01, 0.01, 0.49, 0.49),
    (0.33, 0.33, 0.33, 0.01),
    (0.33, 0.33, 0.01, 0.33),
    (0.33, 0.01, 0.33, 0.33),
    (0.01, 0.33, 0.33, 0.33),
]

config_yaml_path = os.path.join("lsmcache/config/config_lsm_cache.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

E = config["lsm_tree_config"]["E"] / 8 # bits/entry -> Bytes/entry
Q = int(config["lsm_tree_config"]["Q"])
B = int(4096 / E) # entries per page
M = config["lsm_tree_config"]["M"] # total memory in bits
N = config["lsm_tree_config"]["N"] # total entries
sel = config["lsm_tree_config"]["s"]

class LevelCost(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

    def single_run(
        self,
        workload,
        ratio,
        n,
        Mbuf,
        dist,
        skew,
        Mcache,
        queries,
        key_log,
    ):
        z0, z1, q, w = workload
        self.logger.info(f"Workload : {z0},{z1},{q},{w}")
        self.logger.info(f"Building DB at size : {n}")
        row = self.config["lsm_tree_config"].copy()

        row["db_name"] = "level_cost"
        row["path_db"] = self.config["app"]["DATABASE_PATH"]
        row["T"] = 10 # 默认值
        row["N"] = n
        row["h"] = 5 # 默认值
        row["dist"] = dist
        row["skew"] = skew
        row["Mcache"] = Mcache
        row["Mbuf"] = Mbuf
        row["queries"] = queries
        row["z0"] = z0
        row["z1"] = z1
        row["q"] = q
        row["w"] = w
        db = RocksDB(self.config)

        self.logger.info("Running workload")
        row["key_log"] = key_log
        results = db.run(
            row["db_name"],
            row["path_db"],
            row["h"], # bpe
            row["T"], # size ratio
            row["N"], # n
            row["E"], # bits/entry
            row["M"], # bits/entry
            row["Mbuf"],
            z0,
            z1,
            q,
            w,
            dist,
            skew,
            queries,
            sel,
            cache_cap=row["Mcache"],
            key_log=key_log,
            initial_alpha=ratio, # ✅新增
        )

        for key, val in results.items():
            self.logger.info(f"{key} : {val}")
            row[f"{key}"] = val

        row["L"] = estimate_level(N, row["Mbuf"] / 2, row["T"], E) # E(bytes)
        row["z0"] = z0
        row["z1"] = z1
        row["q"] = q
        row["w"] = w
        row["ratio"] = ratio

        return row


    def run(self):
        start_time = time.time()
        df = []
        key_path = "key_log_al_level_cost"
        if not os.path.exists(key_path):
            os.makedirs(key_path)
        step = 0

        # ============ 确定最佳的ratio(Mb与Mc之间的比例) =============
        # M 专指分配给 buffer + cache 的内存预算
        # Bloom Filter 内存不计入此分配
        for workload in workloads:

            min_err = 1e9
            for ratio in np.arange(0.1, 0.9, 0.05):  # [0.1,0.9) 8个值
                Mbuf = ratio * M / 8 # bytes
                Mcache = (1 - ratio) * M / 8 # bytes
                dist = "zipf"
                skew = 0.8
                key_log = key_path + "/{}.dat".format(step)
                row = self.single_run(
                    workload,
                    ratio,
                    N,
                    Mbuf,
                    dist,
                    skew,
                    Mcache,
                    Q,
                    key_log,
                )
                df.append(row)
                pd.DataFrame(df).to_csv(self.config["samples_path"]["lsmcache_xgb_ckpt"])
                step += 1
                self.logger.info(f"Used {time.time()-start_time}s\n")

        self.logger.info("Exporting data from xgb level")
        pd.DataFrame(df).to_csv(self.config["samples_path"]["lsmcache_xgb_final"])
        self.logger.info(f"Finished xgb level, use {time.time()-start_time}s\n")

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
    driver.run(LevelCost(config))
