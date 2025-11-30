import numpy as np
import pandas as pd
import sys
import logging
import os
import yaml
import copy
import random
import pickle as pkl

sys.path.append("./lrkv")
from runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import CostFunction
from lsm_tree.tunner import NominalWorkloadTuning
from utils import model_lr
from utils import model_xgb
from utils.distribution import dist_regression

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("lrkv/config/config.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
scaling = config["lsm_tree_config"]["scaling"]
E = config["lsm_tree_config"]["E"] / 8 # bits/entry -> Bytes/entry
Q = int(config["lsm_tree_config"]["Q"] * scaling) 
B = int(4096 / E) # entries per page
M = config["lsm_tree_config"]["M"] * scaling # total memory in bits
N = config["lsm_tree_config"]["N"] * scaling # total entries
sel = config["lsm_tree_config"]["s"]
workloads = [
    (0.25, 0.25, 0.25, 0.25),
    (0.97, 0.01, 0.01, 0.01),
    (0.01, 0.97, 0.01, 0.01),
    (0.01, 0.01, 0.97, 0.01),
    #(0.01, 0.01, 0.01, 0.97),
    (0.49, 0.49, 0.01, 0.01),
    (0.49, 0.01, 0.49, 0.01),
    #(0.49, 0.01, 0.01, 0.49),
    (0.01, 0.49, 0.49, 0.01),
    (0.01, 0.49, 0.01, 0.49),
    (0.01, 0.01, 0.49, 0.49),
    (0.33, 0.33, 0.33, 0.01),
    (0.33, 0.33, 0.01, 0.33),
    (0.33, 0.01, 0.33, 0.33),
    (0.01, 0.33, 0.33, 0.33),
    (0.91, 0.03, 0.03, 0.03),
    (0.75, 0.15, 0.05, 0.05),
    (0.60, 0.30, 0.05, 0.05),
    (0.45, 0.45, 0.05, 0.05),
    (0.30, 0.60, 0.05, 0.05),
    (0.15, 0.75, 0.05, 0.05),
    (0.03, 0.91, 0.03, 0.03),
    (0.05, 0.75, 0.15, 0.05),
    (0.05, 0.60, 0.30, 0.05),
    (0.05, 0.45, 0.45, 0.05),
    (0.05, 0.30, 0.60, 0.05),
    (0.05, 0.15, 0.75, 0.05),
    (0.03, 0.03, 0.91, 0.03),
    (0.05, 0.05, 0.75, 0.15),
    (0.05, 0.05, 0.60, 0.30),
    (0.05, 0.05, 0.45, 0.45),
    (0.05, 0.05, 0.30, 0.60),
    (0.05, 0.05, 0.15, 0.75),
    (0.03, 0.03, 0.03, 0.91),
    (0.15, 0.05, 0.05, 0.75),
    (0.30, 0.05, 0.05, 0.60),
    (0.45, 0.05, 0.05, 0.45),
    (0.60, 0.05, 0.05, 0.30),
    (0.75, 0.05, 0.05, 0.15),
]


class Optimizer(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

    # ================ 执行训练得到的模型 ================
    def run(self):
        i = -1
        df = [] # 记录所有运行结果的row
        xgb_t = [] # 延迟结果
        xgb_h = [] # 命中率结果

        for workload in workloads:
            i += 1 # 第一个workload对应i=0
            dist = "zipf"
            skew = 0.8

            row = self.config["lsm_tree_config"].copy()
            row["optimizer"] = "xgb"
            row["db_name"] = "level_optimizer"
            row["path_db"] = self.config["app"]["DATABASE_PATH"]
            z0, z1, q, w = workload
            row["N"] = N
            row["M"] = M
            row["queries"] = Q
            row["dist"] = dist
            row["skew"] = skew
            row["is_leveling_policy"] = True
            row["z0"] = z0
            row["z1"] = z1
            row["q"] = q
            row["w"] = w

            key_path = "key_log_xgb_optimizer"
            if not os.path.exists(key_path):
                os.makedirs(key_path)
            key_log = key_path + "/{}.dat".format(i)
            row["key_log"] = key_log
            self.logger.info(f"Building DB at size : {N}")

            level_cost_models = pkl.load(
                open(self.config["xgb_model"]["level_xgb_cost_model"], "rb")
            )            
            (
                best_T,
                best_h,
                best_ratio,
                best_var,
                best_cost,
            ) = model_xgb.traverse_var_optimizer_uniform(
                level_cost_models,
                1,
                z0,
                z1,
                q,
                w,
                E, # Bytes
                M,
                N,
            )
            row["is_leveling_policy"] = True
            print(
                f"level_optimizer: best_T: {best_T}, best_h: {best_h}, best_ratio: {best_ratio}, best_var: {best_var}, best_cost:{best_cost*Q}"
            )
            row["T"] = int(best_T)
            row["h"] = best_h
            row["mbuf"] = best_ratio * (M - best_h * N) / 8 # bytes
            row["cache_cap"] = (1 - best_ratio) * (M - best_h * N) / 8 # bytes
            self.logger.info(f"Building DB at size : {N}")
            db = RocksDB(self.config)
            results = db.run(
                row["db_name"],
                row["path_db"],
                row["h"], # 最佳h
                row["T"], # 最佳T
                row["N"], 
                row["E"],
                row["M"],
                row["mbuf"], # 根据最佳h和ratio计算得到的写内存大小
                z0,
                z1,
                q,
                w,
                dist,
                skew,
                Q,
                sel,
                is_leveling_policy=row["is_leveling_policy"],
                cache_cap=row["cache_cap"], # 根据最佳h和ratio计算得到的块缓存大小
                key_log=key_log,
                scaling=scaling,
            )
            for key, val in results.items():
                self.logger.info(f"{key} : {val}")
                row[f"{key}"] = val
            row["write_io"] = (
                row["bytes_written"]
                + row["compact_read"]
                + row["compact_write"]
                + row["flush_written"]
            ) / 4096
            self.logger.info("write_io: {}".format(row["write_io"]))
            self.logger.info("mbuf: {}".format(row["mbuf"]))
            # print(row)
            df.append(row)
            pd.DataFrame(df).to_csv(self.config["optimizer_path"]["ckpt"])
            xgb_t.append(row["total_latency"])
            xgb_h.append(row["cache_hit_rate"])

            # 平均latency
            print("xgb_t: ", np.mean(xgb_t))
            
            # 平均命中率
            print("xgb_h: ", np.mean(xgb_h))

        self.logger.info("Exporting data from lr optimizer")
        pd.DataFrame(df).to_csv(self.config["optimizer_path"]["final"])
        self.logger.info("Finished optimizer\n")


if __name__ == "__main__":
    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("lrkv/config/config.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Runner(config)
    driver.run(Optimizer(config))
