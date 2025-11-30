import numpy as np
import pandas as pd
import pickle as pkl
import math
import random

from collections import deque
from scipy import optimize


B = 4


def weight_sampling(l, o, n, r):
    a = []
    b = []
    for i in l:
        a.append(i[o])
        b.append(1.0 / i[-1])
    i = 0
    while len(r) < n:
        if i > 1e5:
            break
        i += 1
        s = random.choices(a, weights=b)[0]
        if s not in r:
            r.append(s)
    return r


def weight_sampling_2d(l, o1, o2, n, r):
    a = []
    b = []
    for i in l:
        a.append([i[o1], i[o2]])
        b.append(1.0 / i[-1])
    i = 0
    while len(r) < n:
        if i > 1e5:
            break
        i += 1
        s = random.choices(a, weights=b)[0]
        if s not in r:
            r.append(s)
    return r

# B / E (Bytes)
def estimate_level(N, B, T, E):
    if (N * E) < B:
        return 1;

    l = np.ceil(np.log((N * E / B) + 1) / np.log(T));

    return l;

def estimate_T(N, mbuf, L, E, get_ceiling=True):
    return int(np.exp(np.log(((N * E) / (mbuf + 1)) + 1)) / L)


def estimate_fpr(h):
    return np.exp(-1 * h * (np.log(2) ** 2))


def T_level_equation(x, q, w):
    return abs(2 * w * x * (np.log(x) - 1) - q * B)

def h_mbuf_level_equation(x, z0, z1, q, w, T, E, M, N):
    return abs(
        ((z0 + z1) * np.exp(-x / N)) / N
        - ((q + 2 * w * (T + 1) / B) / np.log(T) / (M - x))
    )

def h_mbuf_tier_equation(x, z0, z1, w, q, T, E, M, N):
    return abs(
        ((z0 + z1) * T / N * np.exp(-x / N))
        - (((q + w) * T + w / B) / np.log(T) / (M - x))
    )

def T_tier_equation(x, z0, z1, q, w, sel, E, M, N, h0=10):
    mbuf = (M - N * h0) / 8
    l = estimate_level(
        N,
        mbuf,
        x,
        E,
    )
    return abs(
        (z0 + z1) * estimate_fpr(h0)
        + q * sel
        + l * (q * B * x * (np.log(x) - 1) - w) / (B * x * np.log(x))
    )

if __name__ == "__main__":
    pass
