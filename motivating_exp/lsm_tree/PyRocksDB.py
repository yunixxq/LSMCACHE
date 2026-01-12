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
    """
    Python API for RocksDB Motivating Experiment
    用于验证五阶段耦合模型: α → Flush → Compaction → SST失效 → Cache失效 → H_cache
    """

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
        num_z0: float,
        num_z1: float,
        num_q: float,
        num_w: float,
        dist: str,
        skew: float,
        queries: int,
        sel: int = 0,
        alpha: float = 0.05,
        output_file: str = "motivating_exp_results.csv",
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
        self.T = int(T)
        self.N = int(N)
        self.E = int(E)
        self.M = int(M)
        self.h = int(h)
        

        # 构建执行命令
        cmd = [
            self.config["app"]["EXECUTION_PATH"],
            db_dir,
            f"-N {self.N}",
            f"-T {self.T}",
            f"-M {self.M}",
            f"-E {self.E}",
            f"-b {self.h}",
            f"-e {num_z0}",
            f"-r {num_z1}",
            f"-q {num_q}",
            f"-w {num_w}",
            f"-s {queries}",
            f"-c {self.compaction_style}",
            f"--sel {sel}",
            f"--parallelism {THREADS}",
            f"--dist {dist}",
            f"--skew {skew}",
            f"-a {alpha}",
            f"-o {output_file}",
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
        