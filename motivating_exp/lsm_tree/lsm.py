import numpy as np
import pandas as pd
import time

E = 1024

def estimate_level(N, M, alpha, T, E):
    
    Mbuf = M * alpha # 单位为Bytes
    if (N * E) < Mbuf:
        return 1

    l = np.ceil(np.log((N * E / Mbuf) + 1) / np.log(T))

    return l

def build_features(N, M_MB, T, alpha, r, w, s, feature_type: str = "full") -> list:
    data_size_mb = N * E / (1024 * 1024)
    read_write_ratio = r / w
    base_features = [
        # r, 
        # w,
        read_write_ratio,
        s, # skewness
        M_MB,
        data_size_mb / M_MB,
        alpha,
    ]
    
    if feature_type == "base":
        return base_features
    
    M = M_MB * 1024 * 1024
    L = estimate_level(N, M, alpha, T, E)
    Wamp = L * T / alpha

    extend_features = [
        L,
        Wamp
    ]
    
    return base_features + extend_features

def predict_best_alpha(cost_model, r, w, s, T, M_MB, N, feature_type):
    xs = [] # 记录所有候选参数组合经过get_cost_uniform()的特征向量
    settings = [] # 保存所有可能的alpha值

    for alpha in np.arange(0.1, 1.0, 0.1):
        x = build_features(N, M_MB, T, alpha, r, w, s, feature_type)
        settings.append(alpha)
        xs.append(x)

    X = np.array(xs) # 所有参数配置(不同alpha)
    Hcaches = cost_model.predict(X)
    
    best_idx = np.argmax(Hcaches)
    best_alpha = settings[best_idx]
    
    return best_alpha