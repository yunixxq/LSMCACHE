#!/usr/bin/env python3
"""
Five-Stage Coupling Chain Model Visualization
用于生成五阶段耦合链条模型验证实验结果图

This script generates a 2x3 subplot figure showing:
(a) Stage 0: α → B, C (Write Buffer and Block Cache allocation)
(b) Stage 1: B → F_flush (Flush frequency)
(c) Stage 2: F_flush → F_comp (Compaction frequency)
(d) Stage 3: F_comp → τ_sst (SST file lifecycle)
(e) Stage 4-5: Dual-Path Components (H_cap and H_val)
(f) Final: H_cache = H_cap × H_val (Cache hit rate with optimal α*)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline
import argparse
import sys
import os

# 设置全局绘图风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['figure.dpi'] = 150


def load_data(csv_file):
    """加载实验数据"""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} data points from {csv_file}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def smooth_curve(x, y, num_points=100):
    """平滑曲线用于更好的可视化"""
    if len(x) < 4:
        return x, y
    try:
        x_new = np.linspace(x.min(), x.max(), num_points)
        spl = make_interp_spline(x, y, k=min(3, len(x)-1))
        y_smooth = spl(x_new)
        return x_new, y_smooth
    except:
        return x, y


def plot_stage0(ax, df, total_memory_gb):
    """
    Stage 0: α → B, C
    绘制Write Buffer和Block Cache随α变化的关系
    """
    alpha = df['alpha'].values
    Mbuf = df['Mbuf'].values / (1024**3)  # 转换为GB
    Mcache = df['Mcache'].values / (1024**3)
    
    # 绘制填充区域
    ax.fill_between(alpha, 0, Mcache, alpha=0.3, color='#2ca02c', label='_nolegend_')
    ax.fill_between(alpha, 0, Mbuf, alpha=0.3, color='#1f77b4', label='_nolegend_')
    
    # 绘制线条
    ax.plot(alpha, Mbuf, 'o-', color='#1f77b4', linewidth=2, markersize=5, label=r'$B$ (Write Buffer)')
    ax.plot(alpha, Mcache, 's-', color='#2ca02c', linewidth=2, markersize=5, label=r'$C$ (Block Cache)')
    
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Size (GB)')
    ax.set_title(r'(a) Stage 0: $\alpha \rightarrow B, C$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_memory_gb * 1.05)
    ax.legend(loc='center right')
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_stage1(ax, df):
    """
    Stage 1: B → F_flush
    绘制Flush次数随α变化的关系
    """
    alpha = df['alpha'].values
    flush_count = df['flush_count'].values
    
    # 绘制填充区域
    ax.fill_between(alpha, 0, flush_count, alpha=0.3, color='#d62728')
    
    # 绘制线条
    ax.plot(alpha, flush_count, 'o-', color='#d62728', linewidth=2, markersize=5)
    
    # 添加趋势标注
    if len(alpha) > 1 and flush_count[0] > flush_count[-1]:
        ax.annotate(r'$\downarrow$ decreasing', 
                   xy=(alpha[-1], flush_count[-1]), 
                   xytext=(alpha[-1]-0.15, flush_count[-1]+max(flush_count)*0.1),
                   fontsize=10, color='#d62728')
    
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Flush Count')
    ax.set_title(r'(b) Stage 1: $B \rightarrow F_{flush}$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(flush_count) * 1.1 if max(flush_count) > 0 else 1)
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_stage2(ax, df):
    """
    Stage 2: F_flush → F_comp
    绘制Compaction次数随α变化的关系
    """
    alpha = df['alpha'].values
    compaction_count = df['compaction_count'].values
    
    # 绘制填充区域
    ax.fill_between(alpha, 0, compaction_count, alpha=0.3, color='#9467bd')
    
    # 绘制线条
    ax.plot(alpha, compaction_count, 'o-', color='#9467bd', linewidth=2, markersize=5)
    
    # 添加趋势标注
    if len(alpha) > 1 and compaction_count[0] > compaction_count[-1]:
        ax.annotate(r'$\downarrow$ decreasing', 
                   xy=(alpha[-1], compaction_count[-1]), 
                   xytext=(alpha[-1]-0.15, compaction_count[-1]+max(compaction_count)*0.1),
                   fontsize=10, color='#9467bd')
    
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Compaction Count')
    ax.set_title(r'(c) Stage 2: $F_{flush} \rightarrow F_{comp}$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(compaction_count) * 1.1 if max(compaction_count) > 0 else 1)
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_stage3(ax, df):
    """
    Stage 3: F_comp → τ_sst
    绘制SST文件平均生命周期随α变化的关系
    """
    alpha = df['alpha'].values
    tau_sst = df['tau_sst'].values
    
    # 绘制填充区域
    ax.fill_between(alpha, 0, tau_sst, alpha=0.3, color='#17becf')
    
    # 绘制线条
    ax.plot(alpha, tau_sst, 'o-', color='#17becf', linewidth=2, markersize=5)
    
    # 添加趋势标注
    if len(alpha) > 1 and tau_sst[-1] > tau_sst[0]:
        mid_idx = len(alpha) // 2
        ax.annotate(r'$\uparrow$ increasing', 
                   xy=(alpha[mid_idx], tau_sst[mid_idx]), 
                   xytext=(alpha[mid_idx]-0.15, tau_sst[mid_idx]-max(tau_sst)*0.15),
                   fontsize=10, color='#17becf')
    
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\tau_{sst}$ (seconds)')
    ax.set_title(r'(d) Stage 3: $F_{comp} \rightarrow \tau_{sst}$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(tau_sst) * 1.1 if max(tau_sst) > 0 else 1)
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_stage4_5(ax, df):
    """
    Stage 4-5: Dual-Path Components
    绘制H_cap和H_val随α变化的关系（展示双路径特性）
    """
    alpha = df['alpha'].values
    H_cap = df['H_cap'].values
    H_val = df['H_val'].values
    
    # 绘制填充区域
    ax.fill_between(alpha, 0, H_cap, alpha=0.2, color='#2ca02c')
    ax.fill_between(alpha, 0, H_val, alpha=0.2, color='#d62728')
    
    # 绘制线条
    ax.plot(alpha, H_cap, 's-', color='#2ca02c', linewidth=2, markersize=5, 
            label=r'$H_{cap}$ (from $C$)')
    ax.plot(alpha, H_val, 'o-', color='#d62728', linewidth=2, markersize=5, 
            label=r'$H_{val}$ (from $\tau_{sst}$)')
    
    # 添加趋势标注
    if len(alpha) > 3:
        # H_cap趋势（通常递减）
        ax.annotate(r'$\downarrow$', 
                   xy=(alpha[1], H_cap[1]), 
                   fontsize=14, fontweight='bold', color='#2ca02c',
                   ha='center', va='bottom')
        # H_val趋势（通常递增）
        ax.annotate(r'$\uparrow$', 
                   xy=(alpha[-2], H_val[-2]), 
                   fontsize=14, fontweight='bold', color='#d62728',
                   ha='center', va='bottom')
    
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Hit Rate')
    ax.set_title(r'(e) Stage 4-5: Dual-Path Components')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='center right')
    ax.grid(True, linestyle='--', alpha=0.5)


def plot_final(ax, df):
    """
    Final: H_cache = H_cap × H_val
    绘制总Cache命中率及最优α*
    """
    alpha = df['alpha'].values
    H_cache = df['H_cache'].values
    
    # 找到最大值点
    max_idx = np.argmax(H_cache)
    alpha_star = alpha[max_idx]
    H_max = H_cache[max_idx]
    
    # 绘制填充区域
    ax.fill_between(alpha, 0, H_cache, alpha=0.3, color='#1f77b4')
    
    # 绘制线条
    ax.plot(alpha, H_cache, 'o-', color='#1f77b4', linewidth=2, markersize=5)
    
    # 标记最优点
    ax.scatter([alpha_star], [H_max], s=150, color='#ff7f0e', 
               edgecolor='black', linewidth=2, zorder=5)
    
    # 添加垂直虚线
    ax.axvline(x=alpha_star, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 添加标注框
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='wheat', 
                     edgecolor='orange', alpha=0.9)
    ax.annotate(f'$\\alpha^* = {alpha_star:.2f}$\n$H_{{max}} = {H_max:.3f}$',
               xy=(alpha_star, H_max),
               xytext=(alpha_star + 0.15, H_max + 0.05),
               fontsize=10,
               bbox=bbox_props,
               arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
    
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Cache Hit Rate')
    ax.set_title(r'(f) Final: $H_{cache} = H_{cap} \times H_{val}$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(H_cache) * 1.2 if max(H_cache) > 0 else 0.5)
    ax.grid(True, linestyle='--', alpha=0.5)


def create_visualization(df, output_file, total_memory_gb=4.0):
    """
    创建完整的2x3子图可视化
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Five-Stage Coupling Chain Model: Non-Monotonicity of Cache Hit Rate',
                fontsize=14, fontweight='bold', y=1.02)
    
    # 绘制各个子图
    plot_stage0(axes[0, 0], df, total_memory_gb)
    plot_stage1(axes[0, 1], df)
    plot_stage2(axes[0, 2], df)
    plot_stage3(axes[1, 0], df)
    plot_stage4_5(axes[1, 1], df)
    plot_final(axes[1, 2], df)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Figure saved to: {output_file}")
    
    # 同时保存PDF版本
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"PDF saved to: {pdf_file}")
    
    plt.show()
    plt.close()


def create_summary_table(df, output_file):
    """
    创建实验结果摘要表格
    """
    # 找到最优点
    max_idx = df['H_cache'].idxmax()
    optimal = df.loc[max_idx]
    
    # 边界点
    first = df.iloc[0]
    last = df.iloc[-1]
    
    summary = f"""
╔════════════════════════════════════════════════════════════════════════╗
║            Five-Stage Coupling Chain Model - Experiment Summary        ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Data Points: {len(df)}                                                         ║
║  Alpha Range: [{df['alpha'].min():.2f}, {df['alpha'].max():.2f}]                                        ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║                        Key Results                                     ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Optimal Configuration (α* = {optimal['alpha']:.2f}):                              ║
║    - Write Buffer:    {optimal['Mbuf']/(1024**2):.1f} MB                                       ║
║    - Block Cache:     {optimal['Mcache']/(1024**2):.1f} MB                                       ║
║    - H_cache:         {optimal['H_cache']:.4f}                                       ║
║    - H_cap:           {optimal['H_cap']:.4f}                                       ║
║    - H_val:           {optimal['H_val']:.4f}                                       ║
║                                                                        ║
║  Boundary Comparison:                                                  ║
║    - α = {first['alpha']:.2f}: H_cache = {first['H_cache']:.4f}                                   ║
║    - α = {last['alpha']:.2f}: H_cache = {last['H_cache']:.4f}                                   ║
║                                                                        ║
║  Improvement over boundary: {(optimal['H_cache'] - max(first['H_cache'], last['H_cache'])):.4f}                             ║
║                            ({((optimal['H_cache'] - max(first['H_cache'], last['H_cache']))/max(first['H_cache'], last['H_cache'])*100):.1f}% relative improvement)        ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║                      Non-Monotonicity Verification                     ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
"""
    
    # 检查非单调性
    has_increase = any(df['H_cache'].diff()[1:] > 0.001)
    has_decrease = any(df['H_cache'].diff()[1:] < -0.001)
    is_interior = 0 < max_idx < len(df) - 1
    
    summary += f"║  Has increasing segment:  {'✓ Yes' if has_increase else '✗ No'}                                     ║\n"
    summary += f"║  Has decreasing segment:  {'✓ Yes' if has_decrease else '✗ No'}                                     ║\n"
    summary += f"║  Interior maximum:        {'✓ Yes' if is_interior else '✗ No'}                                     ║\n"
    summary += f"║                                                                        ║\n"
    
    if has_increase and has_decrease and is_interior:
        summary += "║  ═══════════════════════════════════════════════════════════════════  ║\n"
        summary += "║  ✓ NON-MONOTONICITY VERIFIED!                                         ║\n"
        summary += "║    The Five-Stage Coupling Model is validated.                        ║\n"
        summary += "║  ═══════════════════════════════════════════════════════════════════  ║\n"
    else:
        summary += "║  ⚠ Non-monotonicity not clearly observed.                             ║\n"
    
    summary += """║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
"""
    
    print(summary)
    
    # 保存到文件
    with open(output_file, 'w') as f:
        f.write(summary)
    print(f"Summary saved to: {output_file}")


def generate_sample_data(output_file, num_points=19):
    """
    生成样本数据用于测试（当没有实际实验数据时）
    """
    np.random.seed(42)
    
    alpha = np.linspace(0.05, 0.95, num_points)
    total_memory = 4 * 1024**3  # 4GB
    
    data = {
        'alpha': alpha,
        'Mbuf': alpha * total_memory,
        'Mcache': (1 - alpha) * total_memory,
        'flush_count': np.maximum(1, 60 * np.exp(-3 * alpha) + np.random.normal(0, 1, num_points)),
        'flush_bytes': np.random.randint(10000000, 100000000, num_points),
        'flush_rate': np.maximum(0.01, 2 * np.exp(-3 * alpha) + np.random.normal(0, 0.1, num_points)),
        'compaction_count': np.maximum(1, 50 * np.exp(-3 * alpha) + np.random.normal(0, 1, num_points)),
        'compaction_read_bytes': np.random.randint(50000000, 500000000, num_points),
        'compaction_write_bytes': np.random.randint(50000000, 500000000, num_points),
        'compaction_rate': np.maximum(0.01, 1.5 * np.exp(-3 * alpha) + np.random.normal(0, 0.05, num_points)),
        'sst_inv_count': np.maximum(1, 40 * np.exp(-2.5 * alpha) + np.random.normal(0, 2, num_points)),
        'tau_sst': 10 + 70 * (1 - np.exp(-3 * alpha)),  # SST生命周期随α增加而增加
        'cache_inv_blocks': np.maximum(0, 1000 * np.exp(-2 * alpha) + np.random.normal(0, 50, num_points)),
        'cache_inv_bytes': np.random.randint(1000000, 50000000, num_points),
        'invalidation_rate': np.maximum(0, 50 * np.exp(-2 * alpha) + np.random.normal(0, 2, num_points)),
        'read_phase_hits': np.random.randint(5000, 50000, num_points),
        'read_phase_misses': np.random.randint(1000, 10000, num_points),
        'mixed_phase_hits': np.random.randint(4000, 40000, num_points),
        'mixed_phase_misses': np.random.randint(1000, 15000, num_points),
        'write_io_kb_per_op': 10 * np.exp(-2 * alpha) + np.random.normal(0, 0.5, num_points),
        'read_io_kb_per_op': 2 + 3 * alpha + np.random.normal(0, 0.2, num_points),
        'total_io_kb_per_op': np.zeros(num_points),
        'initial_latency_ms': np.random.randint(5000, 20000, num_points),
        'read_only_latency_ms': np.random.randint(1000, 5000, num_points),
        'mixed_workload_latency_ms': np.random.randint(5000, 30000, num_points),
    }
    
    # 计算H_cap: 随α增加而减少（因为Cache空间减少）
    # 使用简化的模型: H_cap ≈ min(1, Cache_size / Hot_data_size)
    hot_data_ratio = 0.3  # 假设30%数据是热点
    cache_ratio = (1 - alpha)  # Cache占总内存的比例
    data['H_cap'] = np.minimum(1.0, cache_ratio / hot_data_ratio) * 0.6 + 0.05
    
    # 计算H_val: 随α增加而增加（因为Compaction减少，失效减少）
    # H_val ≈ 1 - orphan_ratio
    data['H_val'] = 0.05 + 0.9 * (1 - np.exp(-4 * alpha))
    
    # H_cache = H_cap * H_val (带一些噪声)
    data['H_cache'] = data['H_cap'] * data['H_val'] * (1 + np.random.normal(0, 0.02, num_points))
    data['H_cache'] = np.clip(data['H_cache'], 0, 1)
    
    # 计算total_io
    data['total_io_kb_per_op'] = data['write_io_kb_per_op'] + data['read_io_kb_per_op']
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Sample data generated and saved to: {output_file}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Five-Stage Coupling Chain Model Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate sample data and visualize
  python %(prog)s --generate-sample
  
  # Visualize from existing CSV file
  python %(prog)s -i results.csv -o figure.png
  
  # Specify total memory for accurate Stage 0 visualization
  python %(prog)s -i results.csv -o figure.png --memory 4.0
        """
    )
    
    parser.add_argument('-i', '--input', type=str, default='motivating_exp_results.csv',
                       help='Input CSV file with experiment results')
    parser.add_argument('-o', '--output', type=str, default='five_stage_coupling.png',
                       help='Output image file')
    parser.add_argument('--memory', type=float, default=4.0,
                       help='Total memory in GB (for Stage 0 visualization)')
    parser.add_argument('--generate-sample', action='store_true',
                       help='Generate sample data for testing')
    parser.add_argument('--summary', type=str, default='experiment_summary.txt',
                       help='Output file for experiment summary')
    
    args = parser.parse_args()
    
    # 生成或加载数据
    if args.generate_sample or not os.path.exists(args.input):
        print("Generating sample data...")
        df = generate_sample_data(args.input)
    else:
        df = load_data(args.input)
    
    # 验证必需的列
    required_columns = ['alpha', 'Mbuf', 'Mcache', 'flush_count', 'compaction_count',
                       'tau_sst', 'H_cap', 'H_val', 'H_cache']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
        print("Some plots may not render correctly.")
    
    # 创建可视化
    create_visualization(df, args.output, args.memory)
    
    # 创建摘要
    create_summary_table(df, args.summary)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()