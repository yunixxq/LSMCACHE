import numpy as np
import pandas as pd
import subprocess
import sys
import logging
import os
import re
import yaml
import pickle as pkl
from multiprocessing import Process

sys.path.append("./lsmcache")
from lsmcache_runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import CostFunction
from utils import model_lr
from utils import model_xgb

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("lsmcache/config/config_lsm_cache.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
scaling = config["lsm_tree_config"]["scaling"]
E = config["lsm_tree_config"]["E"] / 8 # bits/entry -> Bytes/entry
Q = int(config["lsm_tree_config"]["Q"] * scaling)
B = int(4096 / E) # entries per page
M = int(config["lsm_tree_config"]["M"] * scaling) # total memory in bits
N = int(config["lsm_tree_config"]["N"] * scaling) # total entries
sel = config["lsm_tree_config"]["s"] # 4

workloads = [
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

test_workloads_filename = "test_workloads.in"
if os.path.exists(test_workloads_filename):
    os.remove(test_workloads_filename)

dists = ["uniform"]

# ===================== 模型服务：与 db_runner_dynamic.cpp 通信 =====================
def model_server(model="xgb"):
    if model == "xgb":
        xgb_cost_models = pkl.load(
            open(config["xgb_model"]["lsmcache_xgb_cost_model"], "rb")
        )

        xgb_hit_models = pkl.load(
            open(config["xgb_model"]["lsmcache_xgb_hit_model"], "rb")
        )


        # 死循环：不停地监控当前目录下是否存在workloads.in，不存在则continue相当于忙等
        while True:
            if not os.path.exists('workloads.in'):
                continue

            # 一旦发现workloads.in存在，打开文件，读取一行    
            f_in = open('workloads.in', "r")
            workload = f_in.readline().strip().split(' ')
            z0, z1, q, w = [float(x) for x in workload]
            (
                best_T,
                best_h,
                best_ratio,
                best_var,
                best_cost,
            ) = model_xgb.traverse_var_optimizer_uniform2(
                xgb_cost_models,
                xgb_hit_models,
                1,
                z0,
                z1,
                q,
                w,
                E,
                M,
                N,
            )
            # 将预测到的最优参数写入optimal_params.in
            optimal_params_file = open("optimal_params.in", "w")
            optimal_params_file.write(f"{best_T} {best_h} {best_ratio}\n")
            optimal_params_file.close()
            os.remove('workloads.in') # 删除，避免误读旧值
    

# 组装实验，实际执行db_runner_dynamic
class Optimizer(object):
    def __init__(self, config):
        self.db_id = 0
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        # 正则表达式，用来从db_runner_dynamic的日志里提取出
        # [12:34:56.789][info] latency_per_workload : [123, 456, 789, ...]
        self.latency_per_workload_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] latency_per_workload : " r"(\[[0-9,\s]+\])"
        )

    # 将workloads列表中的每个(z0, z1, q, w)写一行到test_workloads.in
    # Python文件负责把workload序列写给C++，C++再按顺序模拟每一段workload的运行
    def generate_workloads(self, workload):
        test_workloads_file = open(test_workloads_filename, "a")
        z0, z1, q, w = workload
        test_workloads_file.write(f"{z0} {z1} {q} {w}\n")
        return [z0, z1, q, w]


    def start_db_runner(self, tuning_T, tuning_h, default_config, w=10000, r=0.05):
        # 通过multiprocessing.Process启动一个子进程
        server = Process(target=model_server, args=("xgb",))
        server.start()

        cmd = [
            "build/db_runner_dynamic",
            f"/tmp/level_test_{self.db_id}", #数据库路径
            f"-N {N}",
            f"-M {M}",
            f"-E {E}"
            f"-s {Q}",
            f"-w {w}",
            f"-r {r}",
            f"--sel {sel}",
            f"--scaling {scaling}",
            f"--parallelism 1",
            f"--dist uniform",
            f"--skew 0.0",
            f"--cache 0.0",
            f"--key-log-file optimizer_data/{self.db_id}.dat",
        ]
        if tuning_T:
            cmd.append("--tuning-T")
        if tuning_h:
            cmd.append("--tuning-h")
        if default_config:
            cmd.append("--default-config")
        cmd = " ".join(cmd)
        self.db_id += 1
        self.logger.debug(f"{cmd}")
        self.logger.info(f"{cmd}")
        # 启动子进程并捕获输出
        proc = subprocess.Popen(
            cmd,
            # stdin=None,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        )
        results = []

        try:
            timeout = 10 * 60 * 60 # 最长等待10h
            proc_results, _ = proc.communicate(timeout=timeout) # 成功结束后，proc_results包含了整个C++程序的输出日志
        except subprocess.TimeoutExpired:
            self.logger.warn("Timeout limit reached. Aborting")
            proc.kill()
        try:
            latency_per_workload = self.latency_per_workload_prog.findall(proc_results)[
                0
            ]
            latency_per_workload = latency_per_workload.strip()
            results = latency_per_workload.strip("][").split(", ")
            results = [int(r) for r in results]
        except:
            self.logger.warn("Log errors")
            proc.kill()

        server.terminate() # 实验结束后，把model_server子进程关掉
        return results # 每个results[i]对应一个workload段的总latancy

    # 注意这里是调用一次db_runner_dynamic执行的是全部工作负载
    def run(self):
        cases = [] # 用于后续构建结果表
        for workload in workloads:
            case = self.generate_workloads(workload)
            cases.append(case)

        # 对比动态调整T、h与默认不调整
        latency_ht = self.start_db_runner(
            tuning_T=True, tuning_h=True, default_config=False, 
            w=10000, r=0.05
        )

        latency_default = self.start_db_runner(
            tuning_T=False, tuning_h=False, default_config=True
        )

        print("latency_ht: ", latency_ht)
        print("latency_default: ", latency_default)

        # 构造CSV结果 工作负载都是动态变化的(source-target)
        df = []
        for i in range(len(cases) - 1):
            row = {}
            (
                row["source_z0"],
                row["source_z1"],
                row["source_q"],
                row["source_w"],
            ) = cases[i]
            (
                row["target_z0"],
                row["target_z1"],
                row["target_q"],
                row["target_w"],
            ) = cases[i + 1]
            row["N"] = N
            row["M"] = M
            row["s"] = Q
            row["latency_tuning_ht"] = latency_ht[i]
            row["latency_rocksdb"] = latency_default[i]
            df.append(row)

        pd.DataFrame(df).to_csv("optimizer_data/dynamic_tuning_results.csv")


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
