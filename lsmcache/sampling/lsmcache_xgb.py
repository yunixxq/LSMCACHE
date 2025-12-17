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
        self.samples = self.config["lsm_tree_config"]["samples"]

    def single_run(
        self,
        workload,
        size_ratio,
        ratio,
        n,
        buffer,
        bpe,
        dist,
        skew,
        cache_cap,
        queries,
        key_log,
    ):
        z0, z1, q, w = workload
        self.logger.info(f"Workload : {z0},{z1},{q},{w}")
        self.logger.info(f"Building DB at size : {n}")
        row = self.config["lsm_tree_config"].copy()

        row["db_name"] = "level_cost"
        row["path_db"] = self.config["app"]["DATABASE_PATH"]
        row["T"] = size_ratio
        row["N"] = n
        row["h"] = bpe
        row["dist"] = dist
        row["skew"] = skew
        row["cache_cap"] = cache_cap
        row["is_leveling_policy"] = True
        row["queries"] = queries
        row["mbuf"] = buffer # bytes
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
            row["mbuf"],
            z0,
            z1,
            q,
            w,
            dist,
            skew,
            queries,
            sel,
            is_leveling_policy=row["is_leveling_policy"],
            cache_cap=cache_cap,
            key_log=key_log,
            initial_alpha=ratio, # ✅新增
        )

        for key, val in results.items():
            self.logger.info(f"{key} : {val}")
            row[f"{key}"] = val

        row["L"] = estimate_level(N, row["mbuf"], row["T"], E) # E(bytes)
        row["z0"] = z0
        row["z1"] = z1
        row["q"] = q
        row["w"] = w
        row["ratio"] = ratio

        return row

    def sample_around_x0(self, x0, h, lower_bound, upper_bound):
        lower_bound = lower_bound
        upper_bound = upper_bound
        if x0 < lower_bound:
            x0 = lower_bound
        if x0 > upper_bound:
            x0 = upper_bound
        samples = []
        left_offset = min(h // 2, x0 - lower_bound)
        right_offset = h - left_offset - 1
        for i in range(-left_offset, right_offset + 1):
            value = x0 + i
            if lower_bound <= value < upper_bound:
                samples.append(value)
        while len(samples) < h and (upper_bound - lower_bound + 1) >= h:
            right_offset += 1
            value = x0 + right_offset
            if lower_bound <= value < upper_bound:
                samples.append(value)
            left_offset += 1
            value = x0 - left_offset
            if lower_bound <= value < upper_bound:
                samples.append(value)
        return samples

    def run(self):
        start_time = time.time()
        df = []
        df2 = []
        key_path = "key_log_al_level_cost"
        if not os.path.exists(key_path):
            os.makedirs(key_path)
        step = 0
        for workload in workloads:
            z0, z1, q, w = workload

            # ============ 确定最佳T =============
            min_err = 1e9
            for T in range(2, 100):
                err = T_level_equation(T, q, w)
                if err < min_err:
                    min_err = err
                    temp = T # 理论最佳T
            if df == []:
                T_list = self.sample_around_x0(temp, self.samples, 2, 200)
            else:
                regr = iter_model(df, "level", E, M, N)
                t = traverse_for_T([regr], z0, z1, q, w, E, M, N, h0=5, n=-1)
                T_list = [temp]
                T_list = weight_sampling(t, 0, self.samples, T_list)
            print(T_list)

            z0, z1, q, w = workload
            ratio = 1.0 # cache=0
            dist = "zipf"
            skew = 0.8
            bpe = 8
            Mbuf = ratio * M / 8 # bytes
            Mcache = 0
            # buffer = ratio * (M - bpe * N) / 8 # bytes
            # cache_cap = 0
            for T in T_list:
                key_log = key_path + "/{}.dat".format(step)
                row = self.single_run(
                    workload,
                    T,
                    ratio,
                    N,
                    Mbuf,
                    bpe,
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

            regr = iter_model(df, "level", E, M, N)
            candidates = traverse_for_T([regr], z0, z1, q, w, E, M, N, n=1)
            # T0 = int((candidates[0][0] + T_list[0]) / 2) # 平均理论最佳T与模型预测最佳T
            T0 = candidates[0][0] # 模型训练得到的最佳T

            # ============ 确定最佳h =============
            min_err = 1e9
            for h in range(2, 11):
                err = h_mbuf_level_equation(h, z0, z1, q, w, T0, E, M, N)
                if err < min_err:
                    min_err = err
                    temp = h # 理论最佳h
            h_list = []
            if False:
                h_list = self.sample_around_x0(temp, self.samples, 2, 11)
            else:
                regr = iter_model(df, "level", E, M, N)
                h = traverse_for_h([regr], z0, z1, q, w, E, M, N, T0=T0, n=-1)
                h_list = [temp]
                h_list = weight_sampling(h, 1, self.samples, h_list)
            print(h_list)
            
            for h in h_list:
                T = T0
                key_log = key_path + "/{}.dat".format(step)
                row = self.single_run(
                    workload,
                    T,
                    ratio,
                    N,
                    Mbuf, # 不变，因为只有ratio会影响Mbuf
                    h,
                    dist,
                    skew,
                    Mcache, # 0
                    Q,
                    key_log,
                )
                df.append(row) # 实际运行得到的结果放入采样集df
                pd.DataFrame(df).to_csv(self.config["samples_path"]["lsmcache_xgb_ckpt"])
                step += 1
                self.logger.info(f"Used {time.time()-start_time}s\n")
                
            # iter model
            regr = iter_model(df, "level", E, M, N)
            candidates = traverse_for_h([regr], z0, z1, q, w, E, M, N, T0=T0, n=1)
            # h0 = int((candidates[0][1] + h_list[0]) / 2) # 平均理论最佳h与模型预测最佳h
            h0 = candidates[0][1] # 模型训练得到的最佳h

            # ============ 确定最佳的ratio(Mb与Mc之间的比例) =============
            # 注：此阶段 h 已固定为 h0，M 专指分配给 buffer+cache 的内存预算
            # Bloom Filter 内存不计入此分配
            min_err = 1e9
            for ratio in np.arange(0.1, 0.9, 0.1):  # [0.1,0.9) 8个值
                # buffer = ratio * (M - h0 * N) / 8 # bytes
                # cache_cap = (1 - ratio) * (M - h0 * N) / 8  # bytes
                Mbuf = ratio * M / 8 # bytes
                Mcache = (1 - ratio) * M / 8 # bytes
                T = T0
                key_log = key_path + "/{}.dat".format(step)
                row = self.single_run(
                    workload,
                    T,
                    ratio,
                    N,
                    Mbuf,
                    h0,
                    dist,
                    skew,
                    Mcache,
                    Q,
                    key_log,
                )
                df.append(row)
                df2.append(row)
                pd.DataFrame(df).to_csv(self.config["samples_path"]["lsmcache_xgb_ckpt"])
                pd.DataFrame(df2).to_csv(self.config["samples_path"]["lsmcache_hit_xgb_ckpt"])
                step += 1
                self.logger.info(f"Used {time.time()-start_time}s\n")

        self.logger.info("Exporting data from xgb level")
        pd.DataFrame(df).to_csv(self.config["samples_path"]["lsmcache_xgb_final"])
        pd.DataFrame(df2).to_csv(self.config["samples_path"]["lsmcache_hit_xgb_final"])
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
