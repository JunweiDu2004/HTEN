import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局学术风格设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14

# ==========================================
# 2. 模型、数据集与核心预测数据设定
# ==========================================
models = ['RBM', 'ESOA', 'DFN', 'EMCD', 'QGNN', 'SplitGNN', 'CaT-GNN', 'HTEN']
datasets = ['Elliptic', 'DGraph-Fin', 'Ethereum']
fraud_ratios = [0.1, 0.5, 1.0, 2.0]

# 基准 AUC-PR (在 2.0% 比例下)
base_auc_pr = {
    'RBM': [0.440, 0.370, 0.390],
    'ESOA': [0.468, 0.395, 0.425],
    'DFN': [0.525, 0.455, 0.482],
    'EMCD': [0.552, 0.465, 0.502],
    'QGNN': [0.585, 0.505, 0.555],
    'SplitGNN': [0.648, 0.625, 0.635],
    'CaT-GNN': [0.655, 0.605, 0.652],
    'HTEN': [0.668, 0.645, 0.672]
}

# 极端不平衡下 (0.1%) 的最大衰退率 (Degradation Ratio)
max_deg = {
    'RBM': [0.355, 0.375, 0.365],
    'ESOA': [0.322, 0.342, 0.332],
    'DFN': [0.274, 0.294, 0.284],
    'EMCD': [0.258, 0.278, 0.268],
    'QGNN': [0.205, 0.225, 0.215],
    'SplitGNN': [0.182, 0.202, 0.192],
    'CaT-GNN': [0.175, 0.195, 0.185],
    'HTEN': [0.095, 0.112, 0.099]
}

# 衰退曲线的平滑插值比例
deg_scales = [1.0, 0.55, 0.25, 0.0]

# 绘图色彩与标记配置
colors = ['#8C564B', '#C44E52', '#937860', '#8172B3', '#CCB974', '#64B5CD', '#4C72B0', '#E03127']
markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
lineweights = [1.5] * 7 + [2.8]  # 加粗突出 HTEN
markersizes = [6] * 7 + [10]  # 放大突出 HTEN

# ==========================================
# 3. 独立生成控制区（请在这里切换您想生成的图）
# ==========================================
# 取消注释您想要单独生成的图像，并将其他的注释掉：

# target = 'Elliptic_AUC-PR'
# target = 'Elliptic_Degradation'
# target = 'DGraph-Fin_AUC-PR'
# target = 'DGraph-Fin_Degradation'
# target = 'Ethereum_AUC-PR'
target = 'Ethereum_Degradation'
# target = 'Legend_Only'

# ==========================================
# 4. 核心绘图逻辑
# ==========================================
ds_idx_map = {'Elliptic': 0, 'DGraph-Fin': 1, 'Ethereum': 2}

if target == 'Legend_Only':
    # ----- 单独生成图例 (2排4列) -----
    fig, ax = plt.subplots(figsize=(10, 2))
    lines = []
    # 绘制隐藏的线用于提取图例
    for m_idx, model in enumerate(models):
        l, = ax.plot([], [], marker=markers[m_idx], color=colors[m_idx],
                     linewidth=lineweights[m_idx], markersize=markersizes[m_idx], label=model)
        lines.append(l)

    ax.axis('off')  # 隐藏坐标轴
    # ncol=4 确保了 8 个模型刚好分成 2 行 4 列
    legend = ax.legend(handles=lines, labels=models, loc='center', ncol=4,
                       frameon=True, edgecolor='black', title="Methods")
    legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    plt.show()
    # plt.savefig('Legend_Only.pdf', format='pdf', dpi=300, bbox_inches='tight')

else:
    # ----- 单独生成某个数据集的某个指标图 -----
    dataset_name, plot_type = target.split('_')
    d_idx = ds_idx_map[dataset_name]

    fig, ax = plt.subplots(figsize=(6, 5))

    for m_idx, model in enumerate(models):
        current_max_deg = max_deg[model][d_idx]
        deg_values = [current_max_deg * scale for scale in deg_scales]

        base_val = base_auc_pr[model][d_idx]
        abs_values = [base_val * (1 - d) for d in deg_values]

        if plot_type == 'AUC-PR':
            ax.plot(fraud_ratios, abs_values, marker=markers[m_idx], color=colors[m_idx],
                    linewidth=lineweights[m_idx], markersize=markersizes[m_idx], label=model, alpha=0.9)
        else:  # Degradation
            ax.plot(fraud_ratios, deg_values, marker=markers[m_idx], color=colors[m_idx],
                    linewidth=lineweights[m_idx], markersize=markersizes[m_idx], label=model, alpha=0.9)

    # 坐标轴与网格设置
    ax.set_title(f'{dataset_name}', fontweight='bold')
    ax.set_xlabel('Fraud Ratio (%)')
    ax.set_xticks(fraud_ratios)
    ax.set_xlim(0, 2.1)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 根据指标动态调整 Y 轴
    if plot_type == 'AUC-PR':
        ax.set_ylabel('AUC-PR')
        ax.set_ylim(0.2, 0.75)
    else:
        ax.set_ylabel('Performance Degradation Ratio')
        ax.set_ylim(0.0, 0.45)

    plt.tight_layout()
    plt.show()