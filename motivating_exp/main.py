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

sys.path.append("./motivating_exp")
from calm_runner import Runner
from lsm_tree.PyRocksDB_main import RocksDB
from lsm_tree.lsm import predict_best_alpha

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("motivating_exp/config/config_main_exp.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# skewness_values = [0.7, 0.8, 0.9, 0.99]
skewness_values = [0.99]

read_ratio1, write_ratio1 = 0.5, 0.5
read_ratio2, write_ratio2 = 0.8, 0.2

# 消融实验 - 四种选择
ablations = [
        {"tag": "RL_on_JS_off", "enable_rl_tuning": True,  "enable_jump_start": False},
        {"tag": "RL_off_JS_on", "enable_rl_tuning": False, "enable_jump_start": True},
        {"tag": "RL_off_JS_off","enable_rl_tuning": False, "enable_jump_start": False},
        {"tag": "RL_on_JS_on","enable_rl_tuning": True, "enable_jump_start": True},
    ]

# 模型选择 - 四种模型配置
model_configs = [
    {"name": "lgb_base", "path": "data/calm/models/calm_lgb_base_model.pkl", "feature_type": "base"},
    {"name": "lgb_full", "path": "data/calm/models/calm_lgb_full_model.pkl", "feature_type": "full"},
    {"name": "xgb_base", "path": "data/calm/models/calm_xgb_base_model.pkl", "feature_type": "base"},
    {"name": "xgb_full", "path": "data/calm/models/calm_xgb_full_model.pkl", "feature_type": "full"},
]

# with open("models/calm_lgb_full_model.pkl", "rb") as f:
#     offline_model = pkl.load(f)

class MainExp(object):
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        self.current_model = None
        self.current_feature_type = None

    def model_serving_loop(self, stop_event):
        """监听workload.in，预测后写入optimal_alpha.in"""
        while not stop_event.is_set():
            if not os.path.exists("workload.in"):
                time.sleep(0.01)
                continue
            
            try:
                with open("workload.in", "r") as f:
                    params = f.readline().strip().split()
                
                r, w, s, T, M_MB, N = float(params[0]), float(params[1]), float(params[2]), \
                                    float(params[3]), float(params[4]), int(params[5])
                
                best_alpha = predict_best_alpha(self.current_model, r, w, s, T, M_MB, N, self.current_feature_type)
                
                with open("optimal_alpha.in", "w") as f:
                    f.write(f"{best_alpha}\n")
                
                os.remove("workload.in")
            except:
                if os.path.exists("workload.in"):
                    os.remove("workload.in")
    
    def run(self):
        exp_output_file = self.config["output_path"]["full_exp_output"]
        epoch_output_file = self.config["output_path"]["epoch_exp_output"]
        
        # 写入CSV表头(只写一次)
        csv_header_1 = "M_MB,N,Q,T,skewness,read_ratio_1,write_ratio_1,read_ratio_2,write_ratio_2," \
                 "alpha_initial,alpha_final,Mbuf_MB,Mcache_MB," \
                 "H_cache,write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op," \
                 "latency,rl_epochs_count,drift_count,converged,rl_agent_enabled,jump_start_enabled,model_type"

        csv_header_2 = "epoch_id,alpha,queries,H_cache,latency_ms," \
                "write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op," \
                "performance_score"

        with open(exp_output_file, 'w') as f:
            f.write(csv_header_1 + "\n")
        
        with open(epoch_output_file, 'w') as f:
            f.write(csv_header_2 + "\n")

        # 启动模型服务线程
        stop_event = threading.Event()
        server_thread = threading.Thread(
            target=self.model_serving_loop,
            args=(stop_event,)
        )
        server_thread.start()
        
        # 三层循环：模型4 * skewness4 * 消融配置4 (后面可能还会有不同的工作负载)
        try:
            for model_cfg in model_configs:
                self.logger.info(f"Loading model: {model_cfg['name']}")
                with open(model_cfg["path"], "rb") as f:
                    self.current_model = pkl.load(f)
                self.current_feature_type = model_cfg["feature_type"]
                
                for skew in skewness_values:
                    for ab in ablations:

                        feature_type = "full"

                        h = self.config["lsm_tree_config"]["h"]
                        T = self.config["lsm_tree_config"]["T"]
                        N = self.config["lsm_tree_config"]["N"]
                        E = self.config["lsm_tree_config"]["E"]
                        M = self.config["lsm_tree_config"]["M"]
                        Q = self.config["lsm_tree_config"]["Q"]

                        M_MB = M / (1024 * 1024)
                        best_alpha_1 = predict_best_alpha(self.current_model, read_ratio1, write_ratio1, skew, T, M_MB, N, self.current_feature_type)
                        best_alpha_2 = predict_best_alpha(self.current_model, read_ratio2, write_ratio2, skew, T, M_MB, N, self.current_feature_type)

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
                            alpha_1 = best_alpha_1,
                            alpha_2 = best_alpha_2,
                            read_num_1 = read_ratio1,
                            write_num_1 = write_ratio1,
                            read_num_2 = read_ratio2,
                            write_num_2 = write_ratio2,
                            dist = "zipfian",
                            skew = skew,
                            # RL参数（新版UCB+Q-Learning）
                            rl_step = 0.01,
                            rl_learning_rate = 0.1,
                            rl_discount = 0.9,
                            rl_epsilon_start = 0.3,
                            rl_epsilon_decay = 0.95,
                            rl_epsilon_min = 0.05,
                            rl_ucb_c = 1.414,
                            # 消融实验
                            enable_rl_tuning=ab["enable_rl_tuning"],
                            enable_jump_start=ab["enable_jump_start"],
                            epoch_ops = 100000,
                            exp_output_file = exp_output_file,
                            epoch_output_file = epoch_output_file,
                            model_type = model_cfg['name']
                        )
        
        finally:
            stop_event.set()
            server_thread.join()
                
        self.logger.info(f"All  experiments completed!")
        self.logger.info(f"Results saved to: {exp_output_file} and {epoch_output_file}")


if __name__ == "__main__":
    # 加载配置
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join("motivating_exp/config/config_main_exp.yaml")

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 启动实验
    driver = Runner(config)
    driver.run(MainExp(config))