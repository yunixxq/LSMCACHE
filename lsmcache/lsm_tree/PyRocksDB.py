"""
Python API for RocksDB
"""

import logging
import os
import re
import shutil
import subprocess
import numpy as np
import time

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

        
        self.epoch_log_file = ""

        # 编译正则表达式
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

    def _log_run_params(self, db_dir, num_z0, num_z1, num_q, num_w, steps, 
                        sel, dist, skew,
                        enable_dynamic_tuning, epoch_size, initial_alpha,
                        enable_epoch_log):
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
        self.logger.info(f"  compaction_style  : {self.compaction_style}")
        
        # 内存配置
        self.logger.info("[Memory Config]")
        self.logger.info(f"  M (total memory)     : {self.M} bytes ({self.M / 1024 / 1024:.2f} MB)")
        self.logger.info(f"  Mbuf (write buffer)  : {self.mbuf} bytes ({self.mbuf / 1024 / 1024:.2f} MB)")
        self.logger.info(f"  Mcache (block cache) : {self.mcache} bytes ({self.mcache / 1024 / 1024:.2f} MB)")
        
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
        
        # 动态调参配置
        self.logger.info("[Dynamic Tuning]")
        self.logger.info(f"  enabled           : {enable_dynamic_tuning}")
        self.logger.info(f"  epoch_size        : {epoch_size}")
        self.logger.info(f"  initial_alpha     : {initial_alpha:.4f}")
        self.logger.info(f"  epoch_log         : {enable_epoch_log}")
        if enable_epoch_log:
            self.logger.info(f"  epoch_log_file    : {self.epoch_log_file}")
        
        self.logger.info("=" * 60)

    def _get_default_results(self):
            """返回默认的结果字典"""
            return {
                "l0_hit": 0, 
                "l1_hit": 0, 
                "l2_plus_hit": 0,
                "filter_neg": 0, 
                "filter_pos": 0, 
                "filter_pos_true": 0,
                "bytes_written": 0, 
                "compact_read": 0, 
                "compact_write": 0, 
                "flush_written": 0,
                "total_read": 0, 
                "estimate_read": 0,
                "write_cost": 0, 
                "read_cost": 0,
                "files_per_level": "[]", 
                "size_per_level": "[]",
                "total_latency": 0, 
                "cache_hit_rate": 0.0,
                "cache_hit": 0, 
                "cache_miss": 0, 
                "init_time": 0,
            }

    def _parse_results(self, proc_results):
        """
        使用配置驱动方式统一解析所有结果字段
        
        :param proc_results: RocksDB 执行的原始输出
        :return: 解析后的结果字典
        """
        # 定义解析配置: (正则模式, 字段名列表, 数据类型, 描述)
        parse_configs = [
            (self.level_hit_prog, 
             ["l0_hit", "l1_hit", "l2_plus_hit"], 
             int, 
             "level_hit (l0, l1, l2plus)"),
            
            (self.bf_count_prog, 
             ["filter_neg", "filter_pos", "filter_pos_true"], 
             int, 
             "bf_count (bf_true_neg, bf_pos, bf_true_pos)"),
            
            (self.compaction_bytes_prog, 
             ["bytes_written", "compact_read", "compact_write", "flush_written"], 
             int, 
             "compaction_bytes (bytes_written, compact_read, compact_write, flush_write)"),
            
            (self.read_bytes_prog, 
             ["total_read", "estimate_read"], 
             int, 
             "read_bytes (total_read, estimate_read)"),
            
            (self.total_latency_prog, 
             ["total_latency"], 
             int, 
             "total_latency"),
            
            (self.cache_hit_rate_prog, 
             ["cache_hit_rate"], 
             float, 
             "cache_hit_rate"),
            
            (self.cache_hit_prog, 
             ["cache_hit"], 
             int, 
             "cache_hit"),
            
            (self.cache_miss_prog, 
             ["cache_miss"], 
             int, 
             "cache_miss"),
            
            (self.init_time_prog, 
             ["init_time"], 
             int, 
             "init_time"),
        ]

        # 定义 findall 类型的解析配置
        findall_configs = [
            (self.files_per_level_prog, "files_per_level", "files_per_level"),
            (self.size_per_level_prog, "size_per_level", "size_per_level"),
        ]

        results = {}
        parse_errors = []

        # 解析 search 类型的字段
        for pattern, field_names, dtype, description in parse_configs:
            try:
                match = pattern.search(proc_results)
                if match is None:
                    raise ValueError("Pattern not found in output")

                groups = match.groups()
                for i, field_name in enumerate(field_names):
                    # 处理 cache_hit_rate 的特殊情况（正则有额外的捕获组）
                    if field_name == "cache_hit_rate":
                        results[field_name] = dtype(groups[0])
                    else:
                        results[field_name] = dtype(groups[i])

            except Exception as e:
                error_info = {
                    "fields": field_names,
                    "description": description,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "pattern": pattern.pattern,
                }
                parse_errors.append(error_info)
                
                self.logger.warning(
                    f"[Parse Failed] {description}\n"
                    f"    Fields: {field_names}\n"
                    f"    Error: {type(e).__name__} - {e}"
                )
                self.logger.debug(f"    Pattern: {pattern.pattern}")

                # 设置默认值
                default_value = dtype(0) if dtype in [int, float] else "0"
                for field_name in field_names:
                    results[field_name] = default_value

        # 解析 findall 类型的字段
        for pattern, field_name, description in findall_configs:
            try:
                matches = pattern.findall(proc_results)
                if not matches:
                    raise ValueError("Pattern not found in output")
                results[field_name] = matches[0].strip()
                
            except Exception as e:
                error_info = {
                    "fields": [field_name],
                    "description": description,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "pattern": pattern.pattern,
                }
                parse_errors.append(error_info)
                
                self.logger.warning(
                    f"[Parse Failed] {description}\n"
                    f"    Field: {field_name}\n"
                    f"    Error: {type(e).__name__} - {e}"
                )
                results[field_name] = "[]"

        # 计算派生字段
        results["write_cost"] = (
            results.get("bytes_written", 0) + 
            results.get("compact_write", 0) + 
            results.get("flush_written", 0)
        )
        results["read_cost"] = (
            results.get("total_read", 0) + 
            results.get("compact_read", 0)
        )

        # 输出解析汇总
        self._log_parse_summary(parse_errors, proc_results)

        return results

    def _log_parse_summary(self, parse_errors, proc_results):
        """
        输出解析结果汇总，并在有错误时保存调试信息
        
        :param parse_errors: 解析错误列表
        :param proc_results: 原始输出
        """
        self.logger.info("=" * 60)
        self.logger.info("Parse Results Summary")
        self.logger.info("=" * 60)
        
        if parse_errors:
            self.logger.warning(f"[Parse Summary] {len(parse_errors)} field group(s) failed to parse:")
            for i, err in enumerate(parse_errors, 1):
                self.logger.warning(
                    f"  {i}. {err['description']}\n"
                    f"       Fields: {err['fields']}\n"
                    f"       Error: {err['error_type']} - {err['error_msg']}"
                )
            
            # 保存调试输出
            self._save_debug_output(proc_results, parse_errors)
        else:
            self.logger.info("[Parse Summary] All fields parsed successfully ✓")
        
        self.logger.info("=" * 60)

    def _save_debug_output(self, proc_results, parse_errors):
        """
        解析失败时保存原始输出用于调试
        
        :param proc_results: 原始输出
        :param parse_errors: 解析错误列表
        """
        try:
            timestamp = int(time.time())
            debug_file = f"debug_output_{timestamp}.txt"
            debug_path = os.path.join(self.path_db, debug_file)
            
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("DEBUG OUTPUT - RocksDB Parse Errors\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("=== PARSE ERRORS ===\n\n")
                for i, err in enumerate(parse_errors, 1):
                    f.write(f"Error {i}:\n")
                    f.write(f"  Description: {err['description']}\n")
                    f.write(f"  Fields: {err['fields']}\n")
                    f.write(f"  Error Type: {err['error_type']}\n")
                    f.write(f"  Error Message: {err['error_msg']}\n")
                    f.write(f"  Pattern: {err['pattern']}\n")
                    f.write("\n")
                
                f.write("\n=== RAW OUTPUT ===\n\n")
                f.write(proc_results)
            
            self.logger.info(f"[Debug] Raw output saved to: {debug_path}")
            
        except Exception as e:
            self.logger.error(f"[Debug] Failed to save debug output: {e}")
    
    def run(
        self,
        db_name,
        path_db,
        h,
        T,
        N,
        E,
        M,
        mbuf,
        mcache,
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
        key_log="",
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
        self.N, self.M = int(N), int(M)
        self.E = E >> 3  # bytes
        self.M = M >> 3  # bytes
        
        if is_leveling_policy:
            self.compaction_style = "level"
        else:
            self.compaction_style = "tier"
        if auto_compaction:
            self.compaction_style = "auto"
        os.makedirs(os.path.join(self.path_db, self.db_name), exist_ok=True)
        self.mbuf = int(mbuf) # bytes
        self.mcache = int(mcache) # bytes
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
            enable_dynamic_tuning=enable_dynamic_tuning,
            epoch_size=epoch_size,
            initial_alpha=initial_alpha,
            enable_epoch_log=enable_epoch_log,
        )

        # 构建执行命令
        cmd = [
            self.config["app"]["EXECUTION_PATH"], # db_runner / db_runner_dynamic
            db_dir,
            f"-N {self.N}",
            f"-T {self.T}",
            f"-B {self.mbuf}",
            f"-M {self.M}", # ✅增加总内存预算(bytes)
            f"-E {self.E}", # bytes
            f"-b {self.h}",
            f"-e {num_z0}",
            f"-r {num_z1}",
            f"-q {num_q}",
            f"-w {num_w}",
            f"-s {steps}",
            f"-c {self.compaction_style}",
            f"--sel {sel}",
            f"--parallelism {THREADS}",
            f"--dist {dist}",
            f"--skew {skew}",
            f"--cache {self.mcache}",
            f"--key-log-file {key_log}",
            # ===== 新增：动态调参参数 =====
            f"--enable-tuning" if enable_dynamic_tuning else "",
            f"--epoch-size {epoch_size}",
            f"--initial-alpha {initial_alpha}",
            # ===== 新增：epoch日志参数 =====
            f"--enable-epoch-log" if enable_epoch_log else "",
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
            print(proc_results)
            print("\n================================================\n")
        except subprocess.TimeoutExpired:
            self.logger.warning("Timeout limit reached. Aborting process.")
            proc.kill()
            return self._get_default_results()
        
        # 使用配置驱动方式解析结果
        results = self._parse_results(proc_results)

        return results