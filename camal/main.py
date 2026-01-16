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

sys.path.append("./camal")
from camal_runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.lsm import predict_best_alpha

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("camal/config/config.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

skewness_values = [0.99]
# skewness_values = [0.99]

# read_ratio1, write_ratio1 = 0.5, 0.5
# read_ratio2, write_ratio2 = 0.8, 0.2

# workloads = [
#     (0.00, 0.90, 0.00, 0.10), 
#     (0.00, 0.85, 0.00, 0.15),
#     (0.00, 0.80, 0.00, 0.20),
#     (0.00, 0.75, 0.00, 0.25),
#     (0.00, 0.70, 0.00, 0.30),
#     (0.00, 0.65, 0.00, 0.35),
#     (0.00, 0.60, 0.00, 0.40),
#     (0.00, 0.55, 0.00, 0.45),
#     (0.00, 0.50, 0.00, 0.50),
# ]

workloads = [
    (0.90, 0.10), 
    (0.85, 0.15),
    (0.80, 0.20),
    (0.75, 0.25),
    (0.70, 0.30),
    (0.65, 0.35),
    (0.60, 0.40),
    (0.55, 0.45),
    (0.50, 0.50),
]

class MainExp(object):
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

        # 加载XGBoost代价模型
        model_path = config["xgb_model"]["camal_xgb_cost_model"]
        with open(model_path, "rb") as f:
            self.cost_models = pkl.load(f)
        self.logger.info(f"Loaded cost models from {model_path}")

    def model_serving_loop(self, stop_event):
        """监听workload.in，预测后写入optimal_alpha.in"""
        while not stop_event.is_set():
            if not os.path.exists("workload.in"):
                time.sleep(0.01)
                continue
            
            try:
                with open("workload.in", "r") as f:
                    params = f.readline().strip().split()
                
                read_ratio = float(params[0])
                skewness = float(params[1])
                T = float(params[2])
                M = float(params[3])
                N = int(params[4])
                
                best_alpha = predict_best_alpha(
                    self.cost_models, N, M, T, read_ratio, skewness
                )
                
                with open("optimal_alpha.in", "w") as f:
                    f.write(f"{best_alpha}\n")
                
                os.remove("workload.in")
            except:
                if os.path.exists("workload.in"):
                    os.remove("workload.in")
    
    def run(self):
        exp_output_file = self.config["output_path"]["main_exp_output"]
        
        # 写入CSV表头(只写一次)
        csv_header = "M_MB,N,Q,T,skewness,read_ratio_1,write_ratio_1," \
                     "read_ratio_2,write_ratio_2,alpha1,alpha2,H_cache," \
                     "write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op,latency"

        with open(exp_output_file, 'w') as f:
            f.write(csv_header + "\n")

        # 启动模型服务线程
        stop_event = threading.Event()
        server_thread = threading.Thread(
            target=self.model_serving_loop,
            args=(stop_event,)
        )
        server_thread.start()
        self.logger.info("Model serving thread started")
        
        # 三层循环：模型4 * skewness4 * 消融配置4 (后面可能还会有不同的工作负载)
        try:
            # 从配置加载 LSM-tree 参数
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
                    
                    best_alpha = predict_best_alpha(
                        self.cost_models, N, M, T, read_ratio, skew
                    )

                    self.logger.info(f"Predicted optimal alpha = {best_alpha:.2f}")

                    # 当前实验两个工作负载相同，alpha1 == alpha2
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
                        alpha_1=best_alpha,
                        alpha_2=best_alpha,
                        read_num_1=read_ratio,
                        write_num_1=write_ratio,
                        read_num_2=read_ratio,
                        write_num_2=write_ratio,
                        dist="zipfian",
                        skew=skew,
                        exp_output_file=exp_output_file,
                    )

        finally:
            stop_event.set()
            server_thread.join()
                
        self.logger.info(f"All  experiments completed!")
        self.logger.info(f"Results saved to: {exp_output_file}")


if __name__ == "__main__":
    # 加载配置
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("camal/config/config.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 启动实验
    driver = Runner(config)
    driver.run(MainExp(config))