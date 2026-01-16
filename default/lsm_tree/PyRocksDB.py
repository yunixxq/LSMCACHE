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
        # 工作负载
        read_num_1: float,
        write_num_1: float,
        read_num_2: float,
        write_num_2: float,
        dist: str,
        skew: float,
        exp_output_file: str = "/data/main_results.csv",
        is_leveling_policy: bool = True,
    ) -> Dict:
        
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
            f"-r1 {read_num_1}",
            f"-w1 {write_num_1}",
            f"-r2 {read_num_2}",
            f"-w2 {write_num_2}",
            f"--dist {dist}",
            f"--skew {skew}",
            f"-c {self.compaction_style}",
            f"--parallelism {THREADS}",
            f"-o {exp_output_file}",
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
        
        return {}