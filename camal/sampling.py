import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import os
import yaml

sys.path.append("./camal")
from camal_runner import Runner
from lsm_tree.PyRocksDB import RocksDB

skewness_values = [0.7, 0.8, 0.9, 0.99]
alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 共9个
workloads = [
    (0.00, 0.90, 0.00, 0.10), 
    (0.00, 0.85, 0.00, 0.15),
    (0.00, 0.80, 0.00, 0.20),
    (0.00, 0.75, 0.00, 0.25),
    (0.00, 0.70, 0.00, 0.30),
    (0.00, 0.65, 0.00, 0.35),
    (0.00, 0.60, 0.00, 0.40),
    (0.00, 0.55, 0.00, 0.45),
    (0.00, 0.50, 0.00, 0.50),
]

config_yaml_path = os.path.join("camal/config/config.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class LevelCost(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

    def run(self):
        output_file = self.config["output_path"]["sampling_output"] # "data/camal/sampling.csv"

        # 写入CSV表头(只写一次)
        csv_header = "M_MB,N,Q,T,skewness,read_ratio_1,write_ratio_1," \
                 "read_ratio_2,write_ratio_2,alpha1,alpha2,H_cache," \
                 "write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op,latency"

        with open(output_file, 'w') as f:
            f.write(csv_header + "\n")

        # ============ 确定最佳的ratio(Mb与Mc之间的比例) =============
        total_experiments = len(skewness_values) * len(workloads) * len(alpha_values)
        current_exp = 0
        for skew in skewness_values:
            for workload in workloads:
                z0, z1, q, w = workload # z1 = read_ratio; w = write_ratio
                for alpha in alpha_values:
                    current_exp += 1
                    self.logger.info(f"[{current_exp}/{total_experiments}] "
                                f"skew={skew}, R:W={z1}:{w}, alpha={alpha}")

                    h = self.config["lsm_tree_config"]["h"]
                    T = self.config["lsm_tree_config"]["T"]
                    N = self.config["lsm_tree_config"]["N"]
                    E = self.config["lsm_tree_config"]["E"]
                    M = self.config["lsm_tree_config"]["M"]
                    Q = self.config["lsm_tree_config"]["Q"]
                    db = RocksDB(self.config)
                    db.run(
                        db_name = "sampling_exp",
                        path_db = self.config["app"]["DATABASE_PATH"],
                        h = h,
                        T = T,
                        N = N,
                        E = E,
                        M = M,
                        Q = Q,
                        alpha_1 = alpha,
                        alpha_2 = alpha,
                        read_num_1 = z1,
                        write_num_1 = w,
                        read_num_2 = z1,
                        write_num_2 = w,
                        dist = "zipfian",
                        skew = skew,
                        exp_output_file = output_file,
                    )
                    


        self.logger.info(f"All {total_experiments} sampling experiments completed!")
        self.logger.info(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("camal/config/config.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Runner(config)
    driver.run(LevelCost(config))
