import pandas as pd

'''
这个文件的目的是实现由CALM的Sampling内容向CAMAL的Sampling内容进行转换
确保二者的采样集合是相同的，以确保模型训练的公平性
'''

def convert_calm_to_camal(input_file, output_file):
    """
    将CALM格式的CSV转换为CAMAL格式
    CALM表头: M_MB,N,Q,T,skewness,read_ratio,write_ratio,alpha,Mbuf_MB,Mcache_MB,...,H_cache,...
    CAMAL表头: M_MB,N,Q,T,skewness,read_ratio_1,write_ratio_1,read_ratio_2,write_ratio_2,alpha1,alpha2,H_cache,...
    """
    # 读取CALM格式的CSV文件
    df = pd.read_csv(input_file)
    
    # 创建新的DataFrame，按照CAMAL文件的格式
    new_df = pd.DataFrame({
        'M_MB': df['M_MB'],
        'N': df['N'],
        'Q': df['Q'],
        'T': df['T'],
        'skewness': df['skewness'],
        'read_ratio_1': df['read_ratio'],
        'write_ratio_1': df['write_ratio'],
        'read_ratio_2': df['read_ratio'],
        'write_ratio_2': df['write_ratio'],
        'alpha1': df['alpha'],
        'alpha2': df['alpha'],
        'H_cache': df['H_cache'],
        'write_io_kb_per_op': df['write_io_kb_per_op'],
        'read_io_kb_per_op': df['read_io_kb_per_op'],
        'total_io_kb_per_op': df['total_io_kb_per_op'],
        'latency': df['latency']
    })
    
    # 保存为新的CSV文件
    new_df.to_csv(output_file, index=False)
    print(f"CALM -> CAMAL 转换完成！已保存到 {output_file}")


def convert_camal_to_calm(input_file, output_file):
    """
    将CAMAL格式的CSV转换为CALM格式
    CAMAL表头: M_MB,N,Q,T,skewness,read_ratio_1,write_ratio_1,read_ratio_2,write_ratio_2,alpha1,alpha2,H_cache,...
    CALM表头: M_MB,N,Q,T,skewness,read_ratio,write_ratio,alpha,Mbuf_MB,Mcache_MB,...,H_cache,...
    注意: 缺失的字段(如compaction_rate等)将被置为0
    """
    # 读取CAMAL格式的CSV文件
    df = pd.read_csv(input_file)
    
    # 创建新的DataFrame，按照CALM文件的格式
    # 由于read_ratio_1 == read_ratio_2, write_ratio_1 == write_ratio_2, alpha1 == alpha2
    # 所以直接取第一个即可
    new_df = pd.DataFrame({
        'M_MB': df['M_MB'],
        'N': df['N'],
        'Q': df['Q'],
        'T': df['T'],
        'skewness': df['skewness'],
        'read_ratio': df['read_ratio_1'],
        'write_ratio': df['write_ratio_1'],
        'alpha': df['alpha1'],
        'Mbuf_MB': 0,
        'Mcache_MB': 0,
        'flush_count': 0,
        'flush_rate': 0,
        'compaction_count': 0,
        'compaction_rate': 0,
        'sst_inv_count': 0,
        'sst_inv_rate': 0,
        'cache_inv_count': 0,
        'cache_inv_rate': 0,
        'H_cache': df['H_cache'],
        'write_io_kb_per_op': df['write_io_kb_per_op'],
        'read_io_kb_per_op': df['read_io_kb_per_op'],
        'total_io_kb_per_op': df['total_io_kb_per_op'],
        'latency': df['latency']
    })
    
    # 保存为新的CSV文件
    new_df.to_csv(output_file, index=False)
    print(f"CAMAL -> CALM 转换完成！已保存到 {output_file}")


if __name__ == "__main__":
    # CALM -> CAMAL
    # calm_input = "data/calm/sampling_exp_results.csv"
    # camal_output = "data/camal/sampling.csv"
    # convert_calm_to_camal(calm_input, camal_output)
    
    # CAMAL -> CALM
    camal_input = "data/files_backup/sampling/sampling_exp_small.csv"
    calm_output = "data/calm/sampling_converted.csv"
    convert_camal_to_calm(camal_input, calm_output)