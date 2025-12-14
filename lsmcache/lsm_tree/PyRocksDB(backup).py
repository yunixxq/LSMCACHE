"""
Python API for RocksDB
"""

import logging
import os
import re
import shutil
import subprocess
import numpy as np

THREADS = 32


class RocksDB(object):
    """
    Python API for RocksDB
    """

    def __init__(self, config):
        """
        Constructor

        :param config:
        """
        self.config = config
        self.logger = logging.getLogger("rlt_logger")

        # 
        self.level_hit_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(l0, l1, l2plus\) : "
            r"\((-?\d+), (-?\d+), (-?\d+)\)"
        )
        self.bf_count_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(bf_true_neg, bf_pos, bf_true_pos\) : "
            r"\((-?\d+), (-?\d+), (-?\d+)\)"
        )
        # Bytes
        self.compaction_bytes_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(bytes_written, compact_read, compact_write, flush_write\) : "
            r"\((-?\d+), (-?\d+), (-?\d+), (-?\d+)\)"
        )

        self.read_bytes_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(total_read, estimate_read\) : "
            r"\((-?\d+), (-?\d+)\)"
        )


        self.files_per_level_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] files_per_level : " r"(\[[0-9,\s]+\])"
        )
        self.size_per_level_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] size_per_level : " r"(\[[0-9,\s]+\])"
        )
        self.total_latency_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(total_latency\) : " r"\((-?\d+)\)"
        )
        self.cache_hit_rate_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(cache_hit_rate\) : " r"\((\d+(\.\d+)?)\)"
        )
        self.cache_hit_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(cache_hit\) : " r"\((-?\d+)\)"
        )
        self.cache_miss_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(cache_miss\) : " r"\((-?\d+)\)"
        )
        self.init_time_prog = re.compile(
            r"\[[0-9:.]+\]\[info\] \(init_time\) : " r"\((-?\d+)\)"
        )

    def options_from_config(self):
        db_settings = {}
        db_settings["path_db"] = self.config["app"]["DATABASE_PATH"]
        db_settings["N"] = self.config["lsm_tree_config"]["N"]
        db_settings["B"] = self.config["lsm_tree_config"]["B"]
        db_settings["E"] = self.config["lsm_tree_config"]["E"]
        db_settings["M"] = self.config["lsm_tree_config"]["M"]
        db_settings["P"] = self.config["lsm_tree_config"]["P"]
        db_settings["is_leveling_policy"] = self.config["lsm_tree_config"][
            "is_leveling_policy"
        ]

        # Defaults
        db_settings["db_name"] = "default"
        db_settings["h"] = 5
        db_settings["T"] = 10

        return db_settings


    def run(
        self,
        db_name,
        path_db,
        h,
        T,
        N,
        E,
        M,
        mbuff,
        num_z0,
        num_z1,
        num_q,
        num_w,
        dist,
        skew,
        steps,
        sel=0,
        is_leveling_policy=True,
        auto_compaction=False,
        cache_cap=0,
        K="",
        f="",
        key_log="",
        scaling=1,
        # 新增：动态调参相关参数
        enable_dynamic_tuning=False,
        epoch_size=1000,
        initial_alpha=0.5,
        # 新增：epoch相关参数
        enable_epoch_log=False,
    ):
        """
        Runs a set of queries on the database

        :param num_z0: empty reads
        :param num_z1: non-empty reads
        :param num_w: writes
        :param enable_dynamic_tuning: 是否启用动态调参
        :param initial_alpha: 初始alpha值 (内存分配给write buffer/block cache的比例)
        :param epoch_size: 每个epoch的操作数
        """
        self.path_db = path_db
        self.db_name = db_name
        self.h, self.T = h, int(np.ceil(T))
        self.K = K
        self.N, self.M = int(N), int(M)
        self.E = E >> 3  # bytes
        self.M = M >> 3  # bytes
        
        if enable_epoch_log:
            self.epoch_log_file = self.config["app"]["LOG_FILE_PATH"]
            # print(f"[RocksDB] Epoch log file: {self.epoch_log_file}")
        if is_leveling_policy:
            self.compaction_style = "level"
        else:
            self.compaction_style = "tier"
        if auto_compaction:
            self.compaction_style = "auto"
        os.makedirs(os.path.join(self.path_db, self.db_name), exist_ok=True)
        self.mbuff = int(mbuff) # bytes
        db_dir = os.path.join(self.path_db, self.db_name)

        # 调用日志函数进行格式化输出
        self._log_run_params(
            db_dir=db_dir,
            num_z0=num_z0,
            num_z1=num_z1,
            num_q=num_q,
            num_w=num_w,
            steps=steps,
            sel=sel,
            dist=dist,
            skew=skew,
            cache_cap=cache_cap,
            key_log=key_log,
            scaling=scaling,
            enable_dynamic_tuning=enable_dynamic_tuning,
            epoch_size=epoch_size,
            initial_alpha=initial_alpha,
            enable_epoch_log=enable_epoch_log,
            f=f,
            K=K,
        )

        cmd = [
            self.config["app"]["EXECUTION_PATH"], # db_runner / db_runner_dynamic
            db_dir,
            f"-N {self.N}",
            f"-T {self.T}",
            f"-K {K}" if K != "" else "",
            f"-B {self.mbuff}",
            f"-M {self.M}", # ✅增加总内存预算(bytes)
            f"-E {self.E}", # bytes
            f"-b {self.h}",
            f"-f {f}" if f != "" else "",
            f"-e {num_z0}",
            f"-r {num_z1}",
            f"-q {num_q}",
            f"-w {num_w}",
            f"-s {steps}",
            f"-c {self.compaction_style}",
            f"--sel {sel}",
            f"--scaling {scaling}",
            f"--parallelism {THREADS}",
            f"--dist {dist}",
            f"--skew {skew}",
            f"--cache {cache_cap}",
            f"--key-log-file {key_log}",
            # ===== 新增：动态调参参数 =====
            f"--enable-tuning" if enable_dynamic_tuning else "",
            f"--epoch-size {epoch_size}",
            f"--initial-alpha {initial_alpha}",
            # ===== 新增：epoch日志参数 =====
            f"--enable-epoch-log" if enable_epoch_log else "",
            f"--epoch-log-file {self.epoch_log_file}",
        ]

        cmd = " ".join(cmd)
        # self.logger.debug(f"{cmd}")
        # self.logger.info(f"{cmd}")

        proc = subprocess.Popen(
            cmd,
            # stdin=None,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        )

        results = {}

        try:
            timeout = 10 * 60 * 60
            proc_results, _ = proc.communicate(timeout=timeout)
            print("\n======= RAW OUTPUT FROM ROCKSDB EXECUTION =======\n")
            print(proc_results)  # 先只打印前 1000 字符
            print("\n================================================\n")
        except subprocess.TimeoutExpired:
            self.logger.warn("Timeout limit reached. Aborting")
            proc.kill()
            results["l0_hit"] = 0
            results["l1_hit"] = 0
            results["l2_plus_hit"] = 0
            results["filter_neg"] = 0
            results["filter_pos"] = 0
            results["filter_pos_true"] = 0
            results["bytes_written"] = 0
            results["compact_read"] = 0
            results["compact_write"] = 0
            results["flush_written"] = 0
            # results["read_io"] = 0
            results["files_per_level"] = 0
            results["size_per_level"] = 0
            results["total_latency"] = 0
            results["cache_hit_rate"] = 0
            results["cache_hit"] = 0
            results["cache_miss"] = 0
            results["init_time"] = 0
            return results
        try:
            level_hit_results = [int(result) for result in self.level_hit_prog.search(proc_results).groups()]  # type: ignore
            bf_count_results = [int(result) for result in self.bf_count_prog.search(proc_results).groups()]  # type: ignore
            compaction_results = [int(result) for result in self.compaction_bytes_prog.search(proc_results).groups()]  # type: ignore
            read_results = [int(result) for result in self.read_bytes_prog.search(proc_results).groups()]

            files_per_level = self.files_per_level_prog.findall(proc_results)[0]
            size_per_level = self.size_per_level_prog.findall(proc_results)[0]
            total_latency_result = [
                int(result)
                for result in self.total_latency_prog.search(proc_results).groups()
            ]
            cache_hit_rate_result = [
                float(result)
                for result in self.cache_hit_rate_prog.search(proc_results).groups()
            ]
            cache_hit_result = [
                int(result)
                for result in self.cache_hit_prog.search(proc_results).groups()
            ]
            cache_miss_result = [
                int(result)
                for result in self.cache_miss_prog.search(proc_results).groups()
            ]
            init_time_result = [
                int(result)
                for result in self.init_time_prog.search(proc_results).groups()
            ]
            results["l0_hit"] = level_hit_results[0]
            results["l1_hit"] = level_hit_results[1]
            results["l2_plus_hit"] = level_hit_results[2]

            results["filter_neg"] = bf_count_results[0]
            results["filter_pos"] = bf_count_results[1]
            results["filter_pos_true"] = bf_count_results[2]

            results["bytes_written"] = compaction_results[0]
            results["compact_read"] = compaction_results[1]
            results["compact_write"] = compaction_results[2]
            results["flush_written"] = compaction_results[3]

            results["total_read"] = read_results[0]
            results["estimate_read"] = read_results[1]

            results["write_cost"] = results["bytes_written"] + results["compact_write"] + results["flush_written"]
            results["read_cost"] = results["total_read"] + results["compact_read"]

            # 不要直接计算，因为可能会存在分母为0的情况
            # results["WA"] = results["bytes_written"] / results["write_cost"]
            # results["RA"] = results["total_read"] / results["estimate_read"]

            results["files_per_level"] = files_per_level.strip()
            results["size_per_level"] = size_per_level.strip()

            results["total_latency"] = total_latency_result[0]
            results["cache_hit_rate"] = cache_hit_rate_result[0]
            results["cache_hit"] = cache_hit_result[0]
            results["cache_miss"] = cache_miss_result[0]

            
            results["init_time"] = init_time_result[0]
            return results
        except:
            self.logger.warn("Log errors")
            proc.kill()
            results["l0_hit"] = 0
            results["l1_hit"] = 0
            results["l2_plus_hit"] = 0
            results["z0_ms"] = 0
            results["z1_ms"] = 0
            results["q_ms"] = 0
            results["w_ms"] = 0
            results["filter_neg"] = 0
            results["filter_pos"] = 0
            results["filter_pos_true"] = 0
            results["bytes_written"] = 0
            results["compact_read"] = 0
            results["compact_write"] = 0
            results["flush_written"] = 0
            # results["read_io"] = 0
            results["files_per_level"] = 0
            results["size_per_level"] = 0
            results["total_latency"] = 0
            results["cache_hit_rate"] = 0
            results["cache_hit"] = 0
            results["cache_miss"] = 0
            results["init_time"] = 0
            return results


def _log_run_params(self, db_dir, num_z0, num_z1, num_q, num_w, steps, 
                     sel, dist, skew, cache_cap, key_log, scaling,
                     enable_dynamic_tuning, epoch_size, initial_alpha,
                     enable_epoch_log, f, K):
    """格式化输出运行参数"""
    self.logger.info("=" * 60)
    self.logger.info("RocksDB Run Parameters")
    self.logger.info("=" * 60)
    
    # LSM-Tree 配置
    self.logger.info("[LSM-Tree Config]")
    self.logger.info(f"  db_dir            : {db_dir}")
    self.logger.info(f"  N (entries)       : {self.N}")
    self.logger.info(f"  T (size ratio)    : {self.T}")
    self.logger.info(f"  E (entry size)    : {self.E} bytes")
    self.logger.info(f"  h (BF bits)       : {self.h}")
    self.logger.info(f"  K                 : {K if K else 'N/A'}")
    self.logger.info(f"  f                 : {f if f else 'N/A'}")
    self.logger.info(f"  compaction_style  : {self.compaction_style}")
    
    # 内存配置
    self.logger.info("[Memory Config]")
    self.logger.info(f"  M (total memory)  : {self.M} bytes ({self.M / 1024 / 1024:.2f} MB)")
    self.logger.info(f"  B (write buffer)  : {self.mbuff} bytes ({self.mbuff / 1024 / 1024:.2f} MB)")
    self.logger.info(f"  cache_cap         : {cache_cap} bytes ({cache_cap / 1024 / 1024:.2f} MB)")
    
    # 工作负载配置
    self.logger.info("[Workload Config]")
    self.logger.info(f"  num_z0 (empty)    : {num_z0}")
    self.logger.info(f"  num_z1 (non-empty): {num_z1}")
    self.logger.info(f"  num_q (queries)   : {num_q}")
    self.logger.info(f"  num_w (writes)    : {num_w}")
    self.logger.info(f"  steps             : {steps}")
    self.logger.info(f"  sel               : {sel}")
    self.logger.info(f"  dist              : {dist}")
    self.logger.info(f"  skew              : {skew}")
    self.logger.info(f"  scaling           : {scaling}")
    self.logger.info(f"  key_log           : {key_log if key_log else 'N/A'}")
    
    # 动态调参配置
    self.logger.info("[Dynamic Tuning]")
    self.logger.info(f"  enabled           : {enable_dynamic_tuning}")
    self.logger.info(f"  epoch_size        : {epoch_size}")
    self.logger.info(f"  initial_alpha     : {initial_alpha:.4f}")
    self.logger.info(f"  epoch_log         : {enable_epoch_log}")
    if enable_epoch_log:
        self.logger.info(f"  epoch_log_file    : {self.epoch_log_file}")
    
    self.logger.info("=" * 60)