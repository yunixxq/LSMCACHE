import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import sys
import random
import time
import os
import yaml
from datetime import datetime

sys.path.append("./lsmcache")
from utils.model_xgb import get_cost_uniform_only_ratio
from utils.lsm import *
from sklearn.model_selection import KFold

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("lsmcache/config/config_lsm_cache.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
LAT_PATH = config["samples_path"]["lsmcache_xgb_final"]
HIT_PATH = config["samples_path"]["lsmcache_xgb_final"]

FOLD = 15

# 分别训练两个模型，一个用于延迟/另一个用于命中率
for num_sample in [18]:  # 每个工作负载的采样数量，因为工作负载数=FOLD数
    start_time = time.time() # s
    print("Start level training")

    # 针对延迟采样集合
    all_samples = pd.read_csv(LAT_PATH) # 加载数据
    timestamp = os.path.getctime(LAT_PATH)
    creation_time = datetime.fromtimestamp(timestamp)
    print(creation_time)

    all_samples = all_samples.sample(frac=1) # 随机打乱样本顺序
    all_samples = all_samples[: num_sample * FOLD]
    print(len(all_samples))

    # 针对命中率采样集合
    all_hit_samples = pd.read_csv(HIT_PATH) # 加载数据
    hit_timestamp = os.path.getctime(HIT_PATH)
    hit_creation_time = datetime.fromtimestamp(hit_timestamp)
    print(hit_creation_time)

    all_hit_samples = all_hit_samples.sample(frac=1) # 随机打乱样本顺序
    all_hit_samples = all_hit_samples[: num_sample * FOLD]
    print(len(all_hit_samples))

    # ------------- 构造训练特征X和标签Y -------------
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []

    # 针对延迟采样集合
    for _, sample in all_samples.iterrows():
        if sample["queries"] == 0:
            continue
        # if sample["read_io"] + sample["write_io"] == 0:
        #     continue

        X1.append(
            get_cost_uniform_only_ratio(
                sample["ratio"],
                sample["z0"],
                sample["z1"],
                sample["q"],
                sample["w"],
                sample["E"] / 8, # bytes
                sample["M"] / 8, # bytes
                sample["N"],
            )
        )
        y1 = sample["total_latency"] / sample["queries"] # 训练标签是查询的平均延迟
        Y1.append(y1)

    # 针对命中率采样集合
    for _, sample in all_hit_samples.iterrows():
        if sample["queries"] == 0:
            continue
        # if sample["read_io"] + sample["write_io"] == 0:
        #     continue

        X2.append(
            get_cost_uniform_only_ratio(
                sample["ratio"],
                sample["z0"],
                sample["z1"],
                sample["q"],
                sample["w"],
                sample["E"] / 8,
                sample["M"] / 8,
                sample["N"],
            )
        )
        y2 = sample["cache_hit_rate"] # 训练标签是查询的平均缓存命中率
        Y2.append(y2)

    eps = 1e-8
    regrs1 = []
    regrs2 = []
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)

    # ---------------------- 训练 XGBoost 模型 ----------------------
    kf = KFold(n_splits=FOLD)
    errors1 = []
    errors2 = []

    for train_index, test_index in kf.split(X1):
        X1_train = X1[train_index]
        X1_test = X1[test_index]

        Y1_train = Y1[train_index]        
        regr1 = xgb.XGBRegressor(learning_rate=0.1, n_estimators=500)
        # Train the XGBoost cache model
        regr1.fit(X1_train, Y1_train)

        Y1_test = Y1[test_index]
        y_hat1 = regr1.predict(X1_test)
        error1 = abs(y_hat1 - Y1_test) # 绝对误差
        # rerror1 = abs(y_hat1 - Y1_test) / Y1_test # 相对误差
        for _y_hat1, _y1, _error1 in zip(y_hat1, Y1_test, error1):
            errors1.append(_error1)
            #rerrors1.append(_rerror1)
        regrs1.append(regr1) # 保存每一折的模型


    for train_index, test_index in kf.split(X2):
        X2_train = X2[train_index]
        X2_test = X2[test_index]

        Y2_train = Y2[train_index]        
        regr2 = xgb.XGBRegressor(learning_rate=0.1, n_estimators=500)
        # Train the XGBoost cache model
        regr2.fit(X2_train, Y2_train)

        Y2_test = Y2[test_index]
        y_hat2 = regr2.predict(X2_test)
        error2 = abs(y_hat2 - Y2_test) # 绝对误差
        # rerror2 = abs(y_hat2 - Y2_test) / Y2_test # 相对误差 可能会出现分母为0的情况
        for _y_hat2, _y2, _error2 in zip(y_hat2, Y2_test, error2):
            errors2.append(_error2)
            #rerrors2.append(_rerror2)
        regrs2.append(regr2) # 保存每一折的模型

    print(np.mean(errors1))
    pickle.dump(regrs1, open(config["xgb_model"]["lsmcache_xgb_cost_model"], "wb"))

    print(np.mean(errors2))
    pickle.dump(regrs2, open(config["xgb_model"]["lsmcache_xgb_hit_model"], "wb"))

    print(time.time() - start_time)

