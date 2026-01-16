import numpy as np
import pandas as pd
import sys
import logging
import os
import yaml
import copy
import random
import pickle as pkl

sys.path.append("./motivating_exp")
from calm_runner import Runner
from lsm_tree.PyRocksDB import RocksDB

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("motivating_exp/config/config_sampling_exp.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

skewness_values = [0.8, 0.9]
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

workloads = [
    # (0.00, 0.90, 0.00, 0.10), # writes = 10%
    # (0.00, 0.80, 0.00, 0.20), # writes = 20%
    # (0.00, 0.70, 0.00, 0.30), # writes = 30%
    (0.00, 0.60, 0.00, 0.40), # writes = 40%
    (0.00, 0.50, 0.00, 0.50), # writes = 50%
]

class SamplingExperiment(object):
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
    
    def run(self):
        output_file = self.config["output_path"]["sampling_exp_output"]  # "/data/sampling_results.csv"
        
        # 写入CSV表头(只写一次)
        csv_header = "M_MB,N,Q,T,skewness,read_ratio,write_ratio,alpha,Mbuf_MB,Mcache_MB," \
                 "flush_count,flush_rate,compaction_count,compaction_rate," \
                 "sst_inv_count,sst_inv_rate,cache_inv_count,cache_inv_rate," \
                 "H_cache,write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op,latency"

        with open(output_file, 'w') as f:
            f.write(csv_header + "\n")
        
        # 三层循环
        total_experiments = len(skewness_values) * len(workloads) * len(alpha_values)
        current_exp = 0

        for skew in skewness_values:
            for workload in workloads:
                z0, z1, q, w = workload
                for alpha in alpha_values:
                    current_exp += 1
                    self.logger.info(f"[{current_exp}/{total_experiments}] "
                                f"skew={skew}, R:W={z1}:{w}, alpha={alpha}")
                    
                    db = RocksDB(self.config)
                    db.run(
                        db_name = "sampling_exp",
                        path_db = self.config["app"]["DATABASE_PATH"],
                        h = self.config["lsm_tree_config"]["h"],
                        T = self.config["lsm_tree_config"]["T"],
                        N = self.config["lsm_tree_config"]["N"],
                        E = self.config["lsm_tree_config"]["E"],
                        M = self.config["lsm_tree_config"]["M"],
                        num_z0 = z0,
                        num_z1 = z1,
                        num_q = q,
                        num_w = w,
                        dist = "zipfian",
                        skew = skew,
                        queries = self.config["lsm_tree_config"]["Q"],
                        sel = self.config["lsm_tree_config"].get("s", 4),
                        alpha = alpha,
                        output_file = output_file,
                    )


        self.logger.info(f"All {total_experiments} sampling experiments completed!")
        self.logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # 加载配置
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("motivating_exp/config/config_sampling_exp.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 启动实验
    driver = Runner(config)
    driver.run(SamplingExperiment(config))