import logging
import os
import re
import shutil
import subprocess
import numpy as np
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

THREADS = 8

class RocksDB(object):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("rlt_logger")
        
    def run(
        self,
        db_name: str,
        path_db: str,
        h: int,
        T: int,
        N: int,
        E: int,
        M: int,
        Q: int,
        # 预测的两个静态alpha*
        alpha_1: float,
        alpha_2: float,
        # 工作负载
        read_num_1: float,
        write_num_1: float,
        read_num_2: float,
        write_num_2: float,
        dist: str,
        skew: float,
        # RL 相关指标
        rl_step: float,
        rl_max_epochs: int,
        rl_conv_win: int,
        rl_imp_thre: float,
        epoch_ops: int,
        # 输出文件目录
        exp_output_file: str = "/data/main_results.csv",
        epoch_output_file: str = "/data/epoch_results.csv",
        is_leveling_policy: bool = True,
    ) -> Dict:
        """运行 Motivating Experiment"""
        
        self.path_db = path_db
        self.db_name = db_name
        self.compaction_style = "level" if is_leveling_policy else "tier"
        
        # 创建数据库目录
        os.makedirs(os.path.join(self.path_db, self.db_name), exist_ok=True)
        db_dir = os.path.join(self.path_db, self.db_name)
        
        # 保存参数

        # 构建执行命令
        cmd = [
            self.config["app"]["EXECUTION_PATH"],
            db_dir,
            f"-N {N}",
            f"-T {T}",
            f"-M {M}",
            f"-E {E}",
            f"-b {h}",
            f"-s {Q}",
            f"-a1 {alpha_1}",
            f"-a2 {alpha_2}",            
            f"-r1 {read_num_1}",
            f"-w1 {write_num_1}",
            f"-r2 {read_num_2}",
            f"-w2 {write_num_2}",
            f"--dist {dist}",
            f"--skew {skew}",
            f"-c {self.compaction_style}",
            f"--parallelism {THREADS}",
            f"-o1 {exp_output_file}",
            f"-o2 {epoch_output_file}",
            f"--rl-step {rl_step}",
            f"--rl-max-epochs {rl_max_epochs}",
            f"--rl-conv-window {rl_conv_win}",
            f"--rl-threshold {rl_imp_thre}",
            f"--epoch-ops {epoch_ops}",
            "--append", 
        ]
        
        cmd_str = " ".join(cmd)
        self.logger.info(f"Executing command: {cmd_str}")
        
        # 执行命令
        proc = subprocess.Popen(
            cmd_str,
            stdout=None,
            stderr=None,
            universal_newlines=True,
            shell=True,
        )
        
        try:
            timeout = 24 * 60 * 60  # 24小时超时（实验可能很长）
            proc_results, _ = proc.communicate(timeout=timeout)
            
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout limit reached. Aborting process.")
            proc.kill()
        