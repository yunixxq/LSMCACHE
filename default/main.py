import numpy as np
import pandas as pd
import sys
import logging
import os
import yaml
import copy
import random
import pickle as pkl
import threading
import time

sys.path.append("./default")
from default_runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.lsm import predict_best_alpha

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("default/config/config.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# skewness_values = [0.7, 0.8, 0.9, 0.99]
skewness_values = [0.8, 0.9, 0.99]

workloads = [
    # (0.90, 0.10), 
    # (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
]

class MainExp(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
    

    def run(self):
        exp_output_file = self.config["output_path"]["main_exp_output"]

        # 写入CSV表头(只写一次)
        csv_header = "M_MB,N,Q,T,skewness,read_ratio_1,write_ratio_1," \
                     "read_ratio_2,write_ratio_2,H_cache," \
                     "write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op,latency"
        
        with open(exp_output_file, 'w') as f:
            f.write(csv_header + "\n")

        h = self.config["lsm_tree_config"]["h"]
        T = self.config["lsm_tree_config"]["T"]
        N = self.config["lsm_tree_config"]["N"]
        E = self.config["lsm_tree_config"]["E"]
        M = self.config["lsm_tree_config"]["M"]  # Bytes
        Q = self.config["lsm_tree_config"]["Q"]
        
        total_experiments = len(skewness_values) * len(workloads)
        current_exp = 0
        
        for skew in skewness_values:
            for (read_ratio, write_ratio) in workloads:
                current_exp += 1
                self.logger.info(f"[{current_exp}/{total_experiments}] "
                                f"skew={skew}, R:W={read_ratio}:{write_ratio}")
                
                db = RocksDB(self.config)
                db.run(
                    db_name="main_exp",
                    path_db=self.config["app"]["DATABASE_PATH"],
                    h=h,
                    T=T,
                    N=N,
                    E=E,
                    M=M,
                    Q=Q,
                    read_num_1=read_ratio,
                    write_num_1=write_ratio,
                    read_num_2=read_ratio,
                    write_num_2=write_ratio,
                    dist="zipfian",
                    skew=skew,
                    exp_output_file=exp_output_file,
                )
                
        self.logger.info(f"All  experiments completed!")
        self.logger.info(f"Results saved to: {exp_output_file}")


if __name__ == "__main__":
    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("default/config/config.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Runner(config)
    driver.run(MainExp(config))
