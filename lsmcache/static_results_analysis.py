import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_results(csv_path="grid_search_final.csv"):
    df = pd.read_csv(csv_path)
    
    # 为每个workload找到最优ratio
    results = []
    
    for w in df['w'].unique():
        subset = df[df['w'] == w]
        
        # 按latency找最优
        best_latency_idx = subset['total_latency'].idxmin()
        best_latency_row = subset.loc[best_latency_idx]
        
        # 按hit_rate找最优
        best_hit_idx = subset['cache_hit_rate'].idxmax()
        best_hit_row = subset.loc[best_hit_idx]
        
        results.append({
            'w': w,
            'best_ratio_by_latency': best_latency_row['ratio'],
            'best_latency': best_latency_row['total_latency'],
            'best_ratio_by_hit': best_hit_row['ratio'],
            'best_hit_rate': best_hit_row['cache_hit_rate'],
        })
    
    result_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("最优 Ratio 汇总 (按写比例 w)")
    print("="*70)
    print(result_df.to_string(index=False))
    
    return result_df

def analyze_results2(csv_path="grid_search_final.csv"):
    df = pd.read_csv(csv_path)
    
    # 为每个workload找到最优ratio
    results = []
    
    for w in df['w'].unique():
        subset = df[df['w'] == w].copy()
        total_configs = len(subset)
        
        # 计算排名 (1 = 最好)
        subset['latency_rank'] = subset['total_latency'].rank(method='min').astype(int)
        subset['hit_rate_rank'] = subset['cache_hit_rate'].rank(ascending=False, method='min').astype(int)
        
        # 按latency找最优
        best_latency_idx = subset['total_latency'].idxmin()
        best_latency_row = subset.loc[best_latency_idx]
        
        # 按hit_rate找最优
        best_hit_idx = subset['cache_hit_rate'].idxmax()
        best_hit_row = subset.loc[best_hit_idx]
        
        results.append({
            'w': w,
            'best_ratio_by_latency': best_latency_row['ratio'],
            'best_latency': best_latency_row['total_latency'],
            'hit_rate_rank_at_best_latency': f"{int(best_latency_row['hit_rate_rank'])}/{total_configs}",
            'best_ratio_by_hit': best_hit_row['ratio'],
            'best_hit_rate': best_hit_row['cache_hit_rate'],
            'latency_rank_at_best_hit': f"{int(best_hit_row['latency_rank'])}/{total_configs}",
        })
    
    result_df = pd.DataFrame(results)
    print("\n" + "="*100)
    print("最优 Ratio 汇总完整版 (按写比例 w)")
    print("="*100)
    print(result_df.to_string(index=False))
    
    return result_df

def query_rank(csv_path, w, ratio):
    """
    查询指定 write 比例和 ratio 下的 latency 排名和 cache hit rate 排名
    
    参数:
        csv_path: CSV 文件路径
        w: 写比例
        ratio: ratio 值
    
    返回:
        dict: 包含排名信息的字典
    """
    df = pd.read_csv(csv_path)
    
    # 筛选该 w 的所有数据
    subset = df[df['w'] == w].copy()
    
    if len(subset) == 0:
        print(f"错误: 未找到 w={w} 的数据")
        print(f"可用的 w 值: {sorted(df['w'].unique())}")
        return None
    
    total_configs = len(subset)
    
    # 计算排名
    subset['latency_rank'] = subset['total_latency'].rank(method='min').astype(int)
    subset['hit_rate_rank'] = subset['cache_hit_rate'].rank(ascending=False, method='min').astype(int)
    
    # 查找指定 ratio 的行
    target_row = subset[np.isclose(subset['ratio'], ratio, rtol=1e-5)]
    
    if len(target_row) == 0:
        print(f"错误: 未找到 w={w}, ratio={ratio} 的数据")
        print(f"该 w 下可用的 ratio 值: {sorted(subset['ratio'].unique())}")
        return None
    
    target_row = target_row.iloc[0]
    
    result = {
        'w': w,
        'ratio': ratio,
        'total_latency': target_row['total_latency'],
        'latency_rank': int(target_row['latency_rank']),
        'cache_hit_rate': target_row['cache_hit_rate'],
        'hit_rate_rank': int(target_row['hit_rate_rank']),
        'total_configs': total_configs,
    }
    
    # print("\n" + "="*60)
    # print(f"查询结果: w={w}, ratio={ratio}")
    # print("="*60)
    # print(f"  Total Latency:    {result['total_latency']:.6f}")
    # print(f"  Latency 排名:     {result['latency_rank']}/{total_configs}")
    # print(f"  Cache Hit Rate:   {result['cache_hit_rate']:.6f}")
    # print(f"  Hit Rate 排名:    {result['hit_rate_rank']}/{total_configs}")
    # print("="*60)
    
    return result

def batch_query_rank(csv_path, queries):
    """
    批量查询多个 (w, ratio) 组合的排名
    
    参数:
        csv_path: CSV 文件路径
        queries: 列表，每个元素是 (w, ratio) 元组
    
    返回:
        DataFrame: 包含所有查询结果
    """
    results = []
    
    for w, ratio in queries:
        result = query_rank(csv_path, w, ratio)
        if result:
            results.append(result)
    
    if results:
        result_df = pd.DataFrame(results)
        result_df['latency_rank_str'] = result_df.apply(
            lambda x: f"{x['latency_rank']}/{x['total_configs']}", axis=1
        )
        result_df['hit_rate_rank_str'] = result_df.apply(
            lambda x: f"{x['hit_rate_rank']}/{x['total_configs']}", axis=1
        )
        
        print("\n" + "="*80)
        print("批量查询结果汇总")
        print("="*80)
        display_cols = ['w', 'ratio', 'total_latency', 'latency_rank_str', 
                       'cache_hit_rate', 'hit_rate_rank_str']
        print(result_df[display_cols].to_string(index=False))
        
        return result_df
    
    return None

if __name__ == "__main__":
    csv_path = "data/backup/test5/optimizer_lsmcache_final.csv"
    result_df = analyze_results(csv_path)
    result_df = analyze_results2(csv_path)
    queries = [
        (0.1, 0.3),
        (0.2, 0.45),
        (0.3, 0.6),
        (0.4, 0.75),
        (0.5, 0.9),
    ]

    batch_query_rank(csv_path, queries)

