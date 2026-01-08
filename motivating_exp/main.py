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
from motivating_runner import Runner
from lsm_tree.PyRocksDB import RocksDB

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("motivating_exp/config/config_motivating_exp.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Motivating Experiment 工作负载配置
workloads = [
    # (0.00, 0.90, 0.00, 0.10), # writes = 10%
    # (0.00, 0.80, 0.00, 0.20), # writes = 20%
    # (0.00, 0.70, 0.00, 0.30), # writes = 30%
    # (0.00, 0.60, 0.00, 0.40), # writes = 40%
    (0.00, 0.50, 0.00, 0.50), # writes = 50%
]

# 通过扫描不同的alpha值，测量H_cap, H_val, H_cache
# 验证五阶段耦合模型
class MotivatingExperiment(object):
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
    
    def run(self):
        df = []  # 记录所有运行结果
        
        for workload_idx, workload in enumerate(workloads):
            z0, z1, q, w = workload
            
            self.logger.info(f"=" * 60)
            self.logger.info(f"Workload {workload_idx + 1}: empty={z0}, non_empty={z1}, range={q}, writes={w}")
            self.logger.info(f"=" * 60)
            
            # 获取 alpha 扫描配置
            alpha_config = self.config.get("alpha_sweep", {})
            alpha_start = alpha_config.get("start", 0.05)
            alpha_end = alpha_config.get("end", 0.95)
            alpha_step = alpha_config.get("step", 0.05)
            
            # 构建基础配置
            base_row = self.config["lsm_tree_config"].copy()
            base_row["db_name"] = "motivating_exp"
            base_row["path_db"] = self.config["app"]["DATABASE_PATH"]
            base_row["dist"] = self.config.get("workload", {}).get("dist", "zipfian")
            base_row["skew"] = self.config.get("workload", {}).get("skew", 0.99)
            base_row["z0"] = z0
            base_row["z1"] = z1
            base_row["q"] = q
            base_row["w"] = w
            
            self.logger.info(f'Running Motivating Experiment with N={base_row["N"]} entries')
            
            # 创建 RocksDB 实例
            db = RocksDB(self.config)
            
            # 运行实验
            results = db.run(
                db_name=base_row["db_name"],
                path_db=base_row["path_db"],
                h=base_row["h"],
                T=base_row["T"],
                N=base_row["N"],
                E=base_row["E"],
                M=base_row["M"],
                num_z0=z0,
                num_z1=z1,
                num_q=q,
                num_w=w,
                dist=base_row["dist"],
                skew=base_row["skew"],
                queries=base_row["Q"],
                sel=base_row.get("s", 4),
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                alpha_step=alpha_step,
                output_file=self.config["optimizer_path"]["motivating_exp_output"],
            )
            
            # 记录结果
            for key, val in results.items():
                self.logger.info(f"{key} : {val}")
                base_row[f"{key}"] = val
            
            df.append(base_row)
            
            # 保存检查点
            pd.DataFrame(df).to_csv(
                self.config["optimizer_path"]["motivating_exp_ckpt"], 
                index=False
            )
        
        # 保存最终结果
        pd.DataFrame(df).to_csv(
            self.config["optimizer_path"]["motivating_exp_final"], 
            index=False
        )
        
        self.logger.info("=" * 60)
        self.logger.info("Motivating Experiment Completed!")
        self.logger.info(f"Results saved to: {self.config['optimizer_path']['motivating_exp_final']}")
        self.logger.info("=" * 60)


if __name__ == "__main__":
    # 加载配置
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("motivating_exp/config/config_motivating_exp.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 启动实验
    driver = Runner(config)
    driver.run(MotivatingExperiment(config))