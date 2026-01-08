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
        
        # ==================== 编译正则表达式 ====================
        # 基础统计
        self.init_time_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(init_time\) : \((-?\d+)\)"
        )
        self.read_only_time_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(read_only_time\) : \((-?\d+)\)"
        )
        self.mix_time_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(mix_time\) : \((-?\d+)\)"
        )
        
        # Phase 1: H_cap
        self.h_cap_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \[Phase 1\] H_cap = ([0-9.]+) \(hits=(\d+), misses=(\d+)\)"
        )
        
        # Phase 2: H_cache, H_val
        self.h_cache_prog = re.compile(
            r"\[[0-9:.]+\]\[info\]\s+\[Phase 2\] H_cache = ([0-9.]+), H_val = H_cache/H_cap = ([0-9.]+)"
        )
        
        # 五阶段链条指标
        self.stage0_mbuf_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] Stage 0:\s+α=([0-9.]+) → Mbuf=(\d+) MB"
        )
        self.stage0_mcache_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] Stage 0':\s+α=([0-9.]+) → Mcache=(\d+) MB"
        )
        self.stage1_flush_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] Stage 1:\s+Flush count=(\d+), bytes=(\d+) KB"
        )
        self.stage2_compaction_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] Stage 2:\s+Compaction count=(\d+), read=(\d+) KB, write=(\d+) KB"
        )
        self.stage3_sst_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] Stage 3:\s+SST files invalidated=(\d+)"
        )
        self.stage4_cache_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] Stage 4:\s+Cache entries invalidated=(\d+)"
        )
        self.stage5_metrics_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] Stage 5:\s+H_cap=([0-9.]+), H_val=([0-9.]+), H_cache=([0-9.]+)"
        )
        
        # 非单调性验证
        self.optimal_alpha_prog = re.compile(
            r"\[[0-9:.]+\]\[info\]\s+Optimal alpha\* = ([0-9.]+)"
        )
        self.improvement_prog = re.compile(
            r"\[[0-9:.]+\]\[info\]\s+Improvement over boundary: ([0-9.]+) \(([0-9.]+)%\)"
        )
        
        # 通用统计
        self.level_hit_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(l0, l1, l2plus\) : \((-?\d+), (-?\d+), (-?\d+)\)"
        )
        self.files_per_level_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] files_per_level : (\[[0-9,\s]+\])"
        )

    def _log_run_params(self, db_dir, num_z0, num_z1, num_q, num_w, queries,
                        sel, dist, skew, alpha_start, alpha_end, alpha_step):
        """格式化输出运行参数"""
        self.logger.info("=" * 60)
        self.logger.info("Motivating Experiment Parameters")
        self.logger.info("=" * 60)
        
        self.logger.info("[LSM-Tree Config]")
        self.logger.info(f"  db_dir            : {db_dir}")
        self.logger.info(f"  N (entries)       : {self.N}")
        self.logger.info(f"  T (size ratio)    : {self.T}")
        self.logger.info(f"  E (entry size)    : {self.E} bytes")
        self.logger.info(f"  h (BF bits)       : {self.h}")
        
        self.logger.info("[Memory Config]")
        self.logger.info(f"  M (total memory)  : {self.M} bytes ({self.M / 1024 / 1024:.2f} MB)")
        
        self.logger.info("[Workload Config]")
        self.logger.info(f"  empty_reads (z0)  : {num_z0}")
        self.logger.info(f"  non_empty_reads   : {num_z1}")
        self.logger.info(f"  range_reads (q)   : {num_q}")
        self.logger.info(f"  writes (w)        : {num_w}")
        self.logger.info(f"  total queries     : {queries}")
        self.logger.info(f"  distribution      : {dist}")
        self.logger.info(f"  skew              : {skew}")
        
        self.logger.info("[Alpha Sweep Config]")
        self.logger.info(f"  alpha_start       : {alpha_start}")
        self.logger.info(f"  alpha_end         : {alpha_end}")
        self.logger.info(f"  alpha_step        : {alpha_step}")
        
        self.logger.info("=" * 60)

    def _get_default_results(self) -> Dict:
        """返回默认的结果字典"""
        return {
            "optimal_alpha": 0.0,
            "max_H_cache": 0.0,
            "improvement": 0.0,
            "improvement_pct": 0.0,
            "non_monotonic_verified": False,
            "init_time": 0,
            "total_experiments": 0,
        }

    def _parse_results(self, proc_results: str) -> Dict:
        """解析实验输出"""
        results = self._get_default_results()
        parse_errors = []
        
        # 解析最优 alpha
        match = self.optimal_alpha_prog.search(proc_results)
        if match:
            results["optimal_alpha"] = float(match.group(1))
            results["non_monotonic_verified"] = True
        
        # 解析改进幅度
        match = self.improvement_prog.search(proc_results)
        if match:
            results["improvement"] = float(match.group(1))
            results["improvement_pct"] = float(match.group(2))
        
        # 解析所有 Stage 5 的 H_cache 值，找最大值
        h_cache_values = []
        for match in self.stage5_metrics_prog.finditer(proc_results):
            h_cache = float(match.group(3))
            h_cache_values.append(h_cache)
        
        if h_cache_values:
            results["max_H_cache"] = max(h_cache_values)
            results["total_experiments"] = len(h_cache_values)
        
        # 解析初始化时间（取第一个）
        match = self.init_time_prog.search(proc_results)
        if match:
            results["init_time"] = int(match.group(1))
        
        # 输出解析汇总
        self._log_parse_summary(results, proc_results)
        
        return results

    def _log_parse_summary(self, results: Dict, proc_results: str):
        """输出解析结果汇总"""
        self.logger.info("=" * 60)
        self.logger.info("Motivating Experiment Results Summary")
        self.logger.info("=" * 60)
        
        if results["non_monotonic_verified"]:
            self.logger.info("✓ NON-MONOTONICITY VERIFIED!")
            self.logger.info(f"  Optimal alpha*    : {results['optimal_alpha']:.4f}")
            self.logger.info(f"  Max H_cache       : {results['max_H_cache']:.4f}")
            self.logger.info(f"  Improvement       : {results['improvement']:.4f} ({results['improvement_pct']:.1f}%)")
        else:
            self.logger.warning("Non-monotonicity not clearly observed")
        
        self.logger.info(f"  Total experiments : {results['total_experiments']}")
        self.logger.info("=" * 60)

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
        alpha_start: float = 0.05,
        alpha_end: float = 0.95,
        alpha_step: float = 0.05,
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
        
        # # 输出参数日志
        # self._log_run_params(
        #     db_dir, num_z0, num_z1, num_q, num_w, queries,
        #     sel, dist, skew, alpha_start, alpha_end, alpha_step
        # )
        
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
            # Motivating Experiment 特有参数
            f"--motivating-exp",
            f"--alpha-start {alpha_start}",
            f"--alpha-end {alpha_end}",
            f"--alpha-step {alpha_step}",
            f"-o {output_file}",
        ]
        
        cmd_str = " ".join(cmd)
        self.logger.info(f"Executing command: {cmd_str}")
        
        # 执行命令
        proc = subprocess.Popen(
            cmd_str,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
            stdout=None,  # None 表示继承父进程的 stdout
            stderr=None,  # None 表示继承父进程的 stderr
            universal_newlines=True,
            shell=True,
        )
        
        try:
            timeout = 24 * 60 * 60  # 24小时超时（实验可能很长）
            proc_results, _ = proc.communicate(timeout=timeout)
            
            print("\n" + "=" * 60)
            print("RAW OUTPUT FROM MOTIVATING EXPERIMENT")
            print("=" * 60 + "\n")
            print(proc_results)
            print("\n" + "=" * 60 + "\n")
            
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout limit reached. Aborting process.")
            proc.kill()
            return self._get_default_results()
        
        # 解析结果
        # results = self._parse_results(proc_results)
        
        # return results