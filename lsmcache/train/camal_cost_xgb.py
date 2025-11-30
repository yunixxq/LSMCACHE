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
from utils.model_xgb import get_cost_uniform
from utils.lsm import *
from sklearn.model_selection import KFold

np.set_printoptions(suppress=True)

# ============ 加载全局配置文件 ============
config_yaml_path = os.path.join("lsmcache/config/config_lsm_cache.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
level_data = config["samples_path"]["camal_xgb_final"]
FOLD = 15

for num_sample in [12]:  # 原始为100，修改为12(与每个工作负载的采样数一致)
    start_time = time.time()
    print("Start level training")
    all_samples = pd.read_csv(level_data) # 加载数据
    timestamp = os.path.getctime(level_data)
    creation_time = datetime.fromtimestamp(timestamp)
    print(creation_time)

    all_samples = all_samples.sample(frac=1) # 随机打乱样本顺序
    all_samples = all_samples[: num_sample * FOLD]
    print(len(all_samples))

    # ------------- 构造训练特征X和标签Y -------------
    X = []
    Y = []
    for _, sample in all_samples.iterrows():
        if sample["read_io"] + sample["write_io"] == 0:
            continue

        X.append(
            get_cost_uniform(
                sample["T"],
                sample["h"],
                sample["ratio"],
                sample["z0"],
                sample["z1"],
                sample["q"],
                sample["w"],
                sample["E"] / 8,
                sample["M"],
                sample["N"],
            )
        )
        y = sample["total_latency"] / sample["queries"]
        Y.append(y)

    eps = 1e-8
    regrs = []
    X = np.array(X)
    Y = np.array(Y)

    # ---------------------- 训练 XGBoost 模型 ----------------------
    kf = KFold(n_splits=FOLD)
    errors = []
    rerrors = []
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        Y_train = Y[train_index]
        weights = 1 / Y_train
        regr = xgb.XGBRegressor(learning_rate=0.5, n_estimators=10)
        # Train the XGBoost cache model
        regr.fit(X_train, Y_train)
        X_test = X[test_index]
        Y_test = Y[test_index]
        # print(X_train.shape, X_test.shape)
        y_hat = regr.predict(X_test)
        error = abs(y_hat - Y_test) # 绝对误差
        rerror = abs(y_hat - Y_test) / Y_test # 相对误差
        for _y_hat, _y, _error, _rerror in zip(y_hat, Y_test, error, rerror):
            errors.append(_error)
            rerrors.append(_rerror)
        regrs.append(regr) # 保存每一折的模型
    print(np.mean(errors), np.mean(rerrors))
    pickle.dump(regrs, open(config["xgb_model"]["camal_xgb_cost_model"], "wb"))
    print(time.time() - start_time)
