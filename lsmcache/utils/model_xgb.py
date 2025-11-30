import numpy as np
import sys
import random
import xgboost as xgb
import pickle
import pandas as pd
import time
import yaml
import os

sys.path.append("./lrkv")
from utils.lsm import estimate_level, estimate_fpr

eps = 1e-5


def iter_model(df, policy, E, M, N):
    X = []
    Y = []
    for sample in df:
        X.append(
            get_cost_uniform(
                sample["T"],
                sample["h"],
                sample["ratio"],
                sample["z0"],
                sample["z1"],
                sample["q"],
                sample["w"],
                E,
                M,
                N,
            )
        )
        Y.append(sample["total_latency"] / sample["queries"])
    _X = np.array(X)
    _Y = np.array(Y)
    regr = xgb.XGBRegressor()
    regr.fit(_X, _Y)
    return regr


def prepare_df(samples, save_path):
    df = []
    for _, sample in samples.iterrows():
        row = {}
        if sample["read_io"] + sample["write_io"] == 0:
            continue
        l = estimate_level(sample["N"], sample["mbuf"], sample["T"], get_ceiling=False)
        fpr = np.exp(-1 * sample["h"] * (np.log(2) ** 2))
        data = np.zeros([int(sample["N"])])
        with open(sample["key_log"], "r") as f:
            for line in f.readlines():
                last = ord(line.strip("\n")[-1])
                if last >= ord("A"):
                    data[int(line.strip("\n")[:-1] + str(last - 65))] += 1
                else:
                    data[int(line.strip("\n"))] += 1
        data = np.sort(np.squeeze(data[np.argwhere(data)]))[::-1]
        zipf_X = []
        zipf_Y = []
        for k, d in enumerate(data):
            x0 = np.log(k + 1)
            x1 = 1
            zipf_X.append([x0, x1])
            zipf_Y.append(np.log(d))
        zipf_X = np.array(zipf_X)
        zipf_Y = np.array(zipf_Y)
        alpha, c = np.linalg.lstsq(zipf_X, zipf_Y, rcond=-1)[0]
        alpha = -alpha
        row["alpha"] = alpha
        row["c"] = c
        row["z0"] = sample["z0"]
        row["z1"] = sample["z1"]
        row["q"] = sample["q"]
        row["w"] = sample["w"]
        row["T"] = sample["T"]
        row["l"] = l
        row["fpr"] = fpr
        row["cache_cap"] = sample["cache_cap"]
        row["mbuf"] = sample["mbuf"]
        row["cache_hit_rate"] = sample["cache_hit_rate"]
        row["total_latency"] = sample["total_latency"]
        df.append(row)
    pd.DataFrame(df).to_csv(save_path, index=False)


def load_models(model_path, folds):
    models = []
    for fold in range(folds):
        model = pickle.load(open(model_path.replace("holder", str(fold)), "rb"))
        models.append(model)
    return models


def get_cache(current_T, current_h, current_ratio, alpha, c, z0, z1, q, w, M, N):
    fpr = estimate_fpr(current_h)
    buffer = current_ratio * (M - current_h * N)
    cache_cap = (1 - current_ratio) * (M - current_h * N) / 8
    l = estimate_level(N, buffer, current_T)
    return [alpha, c, z0, z1, q, w, current_T, l, fpr, cache_cap, buffer]


def get_cost_uniform(
    # is_leveling_policy,
    current_T,
    current_h,
    current_ratio,
    z0,
    z1,
    q,
    w,
    E,
    M,
    N,
):
    fpr = estimate_fpr(current_h)
    buffer = current_ratio * (M - current_h * N) / 8 # write buffer in bytes
    cache_cap = (1 - current_ratio) * (M - current_h * N) / 8 # cache capacity in bytes
    l = estimate_level(N, buffer, current_T, E)
    return [z0, z1, q, w, current_T, l, fpr, cache_cap, buffer]


def get_cost(
    current_T,
    current_h,
    current_ratio,
    alpha,
    c,
    z0,
    z1,
    q,
    w,
    y_cache,
    M,
    N,
):
    fpr = estimate_fpr(current_h)
    buffer = current_ratio * (M - current_h * N)
    cache_cap = (1 - current_ratio) * M / 8
    l = estimate_level(N, buffer, current_T)
    return [alpha, c, z0, z1, q, w, current_T, l, fpr, cache_cap, buffer, y_cache]

# CAMAL
def traverse_var_optimizer_uniform(cost_models, policy, z0, z1, q, w, E, M, N):
    start_time = time.time()
    costs = [] # 保存每个模型对所有配置的预测结果
    xs = [] # 记录所有候选参数组合经过get_cost_uniform()的特征向量
    settings = [] # 保存这些特征向量对应的(T, h, ratio)
    # 98 * 9 * 3=2646个候选设计 这里相当于是穷举法找最小值
    for T in range(2, 100):
        for h in range(2, 11):
            for ratio in [0.6, 0.8, 1.0]:
                x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
                settings.append((T, h, ratio, None))
                xs.append(x)
    for cost_model in cost_models: # ✅训练得到的15折模型
        X = np.array(xs) # 2646组参数配置
        cost = cost_model.predict(X) # 2646组参数配置对应的成本
        costs.append(cost)
    costs = np.array(costs) # (15,2646)
    vars = np.var(costs, axis=0) # 对列计算方差-即15个模型对同一个配置预测的结果的方差值
    costs = np.mean(costs, axis=0) # 对列计算平均值
    # 将2464个候选点打包成tuples 每一个元素是 (cost_mean, cost_var, (T, h, ratio, None))
    # 且会按照mean_cost从小到大进行排序
    candidates = sorted(zip(costs, vars, settings), key=lambda x: x[0])
    # sorted(zip(costs, vars, settings), key=lambda x: (x[0], x[1])) #我觉得可以这样考虑，要不然方差没有用
    candidate = candidates[0] # 选择最小的cost
    print(time.time() - start_time)
    # best_T, best_h, best_ratio, best_var, best_cost
    return (
        candidate[-1][0],
        candidate[-1][1],
        candidate[-1][2],
        candidate[1],
        candidate[0],
    )

# xxq
def traverse_var_optimizer_uniform2(cost_models, cost_hit_models, policy, z0, z1, q, w, E, M, N):
    start_time = time.time()

    # -------------------------------
    # 1️⃣ 穷举所有候选配置
    # -------------------------------    
    xs = [] # 记录所有候选参数组合经过get_cost_uniform()的特征向量
    settings = [] # 保存这些特征向量对应的(T, h, ratio)
    # 98 * 9 * 8=7056个候选设计 这里相当于是穷举法找最小值
    for T in range(2, 100):
        for h in range(2, 11):
            for ratio in np.arange(0.1, 0.9, 0.1):
                x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
                settings.append((T, h, ratio))
                xs.append(x)
    X = np.array(xs) # 7056组参数配置，即X.shape=(7056,9) 9是指9个特征维度
    total_candidates = len(xs)
    print("输出总的候选样本数：", total_candidates)

    # -------------------------------
    # 2️⃣ 第一阶段 —— 延迟模型计算 cost_mean, cost_var
    # -------------------------------
    latency_preds = [] # 保存每个延迟模型对所有配置的预测结果
    for model in cost_models: # ✅训练得到的15折模型
        latency = model.predict(X)
        latency_preds.append(latency)

    latency_preds = np.array(latency_preds)
    latency_mean = np.mean(latency_preds, axis=0) 
    latency_var = np.var(latency_preds, axis=0)

    latency_sorted = sorted(
        zip(latency_mean, latency_var, settings, range(total_candidates)),
        key=lambda x: (x[0], x[1])  # 先按 cost_mean，再按 variance
    )

    # -------------------------------
    # 3️⃣ 选择latency最小的前K个候选
    # -------------------------------
    K = min(int(total_candidates * 0.05), 20)
    topK = latency_sorted[:K]

    # -------------------------------
    # 4️⃣ 第二阶段 —— 命中率模型预测 hit_rate
    # -------------------------------
    hit_preds = []
    for entry in topK:
        _, _, setting, idx = entry # 前两个变量为"_"表示去除latency_mean和latency_var
        x_feat = X[idx].reshape(1, -1)  # 单个样本特征(1行9列)

        # 所有 hit-rate 模型预测
        preds = [m.predict(x_feat)[0] for m in cost_hit_models]
        hit_mean = np.mean(preds)
        hit_preds.append((hit_mean, setting))
    hit_preds.sort(key=lambda x: -x[0])  # 从大到小排序
    best_hit_rate, best_setting = hit_preds[0]
    print("best_setting = ", best_setting, "len=", len(best_setting))
    best_T, best_h, best_ratio = best_setting

    print(f"[Optimizer] 搜索耗时: {time.time() - start_time:.3f}s")
    print(f"[Optimizer] Best(T={best_T}, h={best_h}, ratio={best_ratio}), hit={best_hit_rate}")

    # 返回值格式保持与原始API一致
    return best_T, best_h, best_ratio


def traverse_var_optimizer_uniform_T(cost_models, policy, z0, z1, q, w, M, N):
    start_time = time.time()
    costs = []
    xs = []
    settings = []
    for T in range(2, 78):
        h = 10
        ratio = 1
        x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
        settings.append((T, h, ratio, None))
        xs.append(x)
    for cost_model in cost_models:
        X = np.array(xs)
        cost = cost_model.predict(X)
        costs.append(cost)
    costs = np.array(costs)
    costs = np.mean(costs, axis=0)
    candidates = sorted(zip(costs, settings), key=lambda x: x[0])
    candidate = candidates[0]
    print(time.time() - start_time)
    return candidate[1][0], candidate[1][1], candidate[1][2], None, candidate[0]


def traverse_var_optimizer_uniform_memory(cost_models, policy, z0, z1, q, w, E, M, N):
    start_time = time.time()
    costs = []
    xs = []
    settings = []
    for T in range(2, 78):
        for h in range(2, 15):
            ratio = 1
            x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
            settings.append((T, h, ratio, None))
            xs.append(x)
    for cost_model in cost_models:
        X = np.array(xs)
        cost = cost_model.predict(X)
        costs.append(cost)
    costs = np.array(costs)
    costs = np.mean(costs, axis=0)
    candidates = sorted(zip(costs, settings), key=lambda x: x[0])
    candidate = candidates[0]
    print(time.time() - start_time)
    return candidate[1][0], candidate[1][1], candidate[1][2], None, candidate[0]


def traverse_for_T(cost_models, z0, z1, q, w, E, M, N, h0=10, ratio0=1.0, n=10):
    candidates = []
    for T in range(2, 100):
        h = h0
        ratio = ratio0
        costs = []
        for cost_model in cost_models:
            x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
            costs.append(
                max(cost_model.predict(np.array([x]).reshape((1, -1)))[0], eps)
            )
        candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    return candidates[:n]


def traverse_for_h(cost_models, z0, z1, q, w, E, M, N, T0=10, ratio0=1.0, n=10):
    candidates = []
    for h in range(2, 11):
        T = T0
        ratio = ratio0
        costs = []
        for cost_model in cost_models:
            x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
            costs.append(
                max(cost_model.predict(np.array([x]).reshape((1, -1)))[0], eps)
            )
        candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    return candidates[:n]