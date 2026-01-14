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
read_ratio2, write_ratio2 = 0.6, 0.4

ablations = [
        {"tag": "RL_on_JS_off", "enable_rl_tuning": True,  "enable_jump_start": False},
        {"tag": "RL_off_JS_on", "enable_rl_tuning": False, "enable_jump_start": True},
        {"tag": "RL_off_JS_off","enable_rl_tuning": False, "enable_jump_start": False},
        {"tag": "RL_on_JS_on","enable_rl_tuning": False, "enable_jump_start": False},
    ]
''' 一共四种模型选择:
    calm_lgb_base_model.pkl  calm_lgb_full_model.pkl
    calm_xgb_base_model.pkl  calm_xgb_full_model.pkl
'''
with open("models/calm_lgb_full_model.pkl", "rb") as f:
    offline_model = pkl.load(f)

class MainExp(object):
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
    
    def run(self):
        exp_output_file = self.config["output_path"]["full_exp_output"]
        epoch_output_file = self.config["output_path"]["epoch_exp_output"]
        
        # 写入CSV表头(只写一次)
        csv_header_1 = "M_MB,N,Q,T,skewness,read_ratio_1,write_ratio_1,read_ratio_2,write_ratio_2," \
                 "alpha_initial,alpha_final,Mbuf_MB,Mcache_MB," \
                 "H_cache,write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op," \
                 "latency,rl_epochs_count,drift_count,converged,rl_agent_enabled,jump_start_enabled"

        csv_header_2 = "epoch_id,alpha,queries,H_cache,latency_ms," \
                "write_io_kb_per_op,read_io_kb_per_op,total_io_kb_per_op," \
                "performance_score"

        with open(exp_output_file, 'w') as f:
            f.write(csv_header_1 + "\n")
        
        with open(epoch_output_file, 'w') as f:
            f.write(csv_header_2 + "\n")

        for skew in skewness_values:
            for ab in ablations:
                # offline_model = pkl.load(
                #     open("models/calm_lgb_base_model.pkl", "rb")
                # )

                ''' 一共四种特征参数选择
                full base
                '''
                feature_type = "full"

                h = self.config["lsm_tree_config"]["h"]
                T = self.config["lsm_tree_config"]["T"]
                N = self.config["lsm_tree_config"]["N"]
                E = self.config["lsm_tree_config"]["E"]
                M = self.config["lsm_tree_config"]["M"]
                Q = self.config["lsm_tree_config"]["Q"]

                M_MB = M / (1024 * 1024)
                best_alpha_1 = predict_best_alpha(offline_model, read_ratio1, write_ratio1, skew, T, M_MB, N, feature_type)
                best_alpha_2 = predict_best_alpha(offline_model, read_ratio2, write_ratio2, skew, T, M_MB, N, feature_type)

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
                    epoch_ops = 1000000,
                    exp_output_file = exp_output_file,
                    epoch_output_file = epoch_output_file,
                )

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