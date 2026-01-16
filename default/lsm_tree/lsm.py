import numpy as np
import sys
import random
import xgboost as xgb
import pickle
import pandas as pd
import time
import yaml
import os

E = 1024

def estimate_fpr(h):
    return np.exp(-1 * h * (np.log(2) ** 2))

def estimate_level(N, M, alpha, T, E):
    
    Mbuf = M * alpha # 单位为Bytes
    if (N * E) < Mbuf:
        return 1

    l = np.ceil(np.log((N * E / Mbuf) + 1) / np.log(T))

    return l

def get_cost_uniform(
    N,
    M,
    T,
    alpha,
    read_ratio,
    skewness
):
    Mbuf = M * alpha
    Mcache = M * (1 - alpha)
    l = estimate_level(N, M, alpha, T, E)
    return [read_ratio, skewness, l, Mcache, Mbuf]

def predict_best_alpha(cost_models, N, M, T, read_ratio, skewness):
    xs = [] # 记录所有候选参数组合经过get_cost_uniform()的特征向量
    settings = [] # 保存所有可能的alpha值

    for alpha in np.arange(0.1, 1.0, 0.1):
        x = get_cost_uniform(N, M, T, alpha, read_ratio, skewness)
        settings.append(alpha)
        xs.append(x)

    costs = [] # 保存每个模型对所有配置的预测结果
    for cost_model in cost_models: # 训练得到的9折模型
        X = np.array(xs)
        cost = cost_model.predict(X) # 所有参数配置对应的成本
        costs.append(cost)

    costs = np.array(costs)
    mean_costs = np.mean(costs, axis=0) # 对列计算平均值
    best_idx = np.argmin(mean_costs)

    return settings[best_idx]