import matplotlib.pyplot as plt
import numpy as np

# --- 1. 设置顶刊绘图风格 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# --- 2. HTEN 的 9 个核心超参数数据 (DGraph-Fin) ---
data_dict = {
    # 一、 拓扑感知参数
    'Landscape Functions ($K$)': (['1', '3', '5', '7'], [0.965, 0.988, 0.982, 0.975], [0.880, 0.912, 0.905, 0.892]),
    'Topological Feature Dimension ($d_t$)': (
    ['16', '32', '64', '128'], [0.960, 0.978, 0.988, 0.984], [0.875, 0.898, 0.912, 0.906]),
    'GSE-enhanced GNN Layers ($L$)': (['1', '2', '3', '4'], [0.970, 0.988, 0.975, 0.962], [0.885, 0.912, 0.890, 0.872]),

    # 二、 双曲表示参数
    'Hyperbolic Embedding Dimension ($d$)': (
    ['32', '64', '128', '256'], [0.968, 0.980, 0.988, 0.986], [0.882, 0.901, 0.912, 0.910]),
    'Initial Curvature ($c$)': (
    ['0.1', '0.5', '1.0', '2.0'], [0.955, 0.976, 0.988, 0.972], [0.865, 0.895, 0.912, 0.888]),
    'Repulsion Margin ($\\gamma$)': (
    ['0.5', '1.0', '2.0', '5.0'], [0.965, 0.978, 0.988, 0.970], [0.878, 0.896, 0.912, 0.885]),

    # 三、 端到端优化参数
    'Hyperbolic Loss Weight ($\\alpha$)': (
    ['0.01', '0.05', '0.1', '0.5'], [0.970, 0.982, 0.988, 0.968], [0.885, 0.904, 0.912, 0.880]),
    'Weight Decay ($\\lambda$)': (
    ['1e-5', '5e-4', '1e-3', '1e-2'], [0.962, 0.988, 0.984, 0.958], [0.875, 0.912, 0.905, 0.868]),
    'Learning Rate ($lr$)': (
    ['1e-4', '5e-4', '1e-3', '5e-3'], [0.968, 0.982, 0.988, 0.955], [0.882, 0.902, 0.912, 0.860])
}

# ==========================================
# --- 3. 参数选择区：在这里切换你想画的图！ ---
# ==========================================

# target_param = 'Landscape Functions ($K$)'
# target_param = 'Topological Feature Dimension ($d_t$)'
# target_param = 'GSE-enhanced GNN Layers ($L$)'

# target_param = 'Hyperbolic Embedding Dimension ($d$)'
# target_param = 'Initial Curvature ($c$)'
# target_param = 'Repulsion Margin ($\\gamma$)'

# target_param = 'Hyperbolic Loss Weight ($\\alpha$)'
# target_param = 'Weight Decay ($\\lambda$)'
target_param = 'Learning Rate ($lr$)'

# ==========================================

# --- 4. 单张图绘制逻辑 ---
labels, auc_scores, f1_scores = data_dict[target_param]

bar_width = 0.35
opacity = 0.9
patterns = ['//', '..']
colors = ['#1f77b4', '#ff7f0e']

# 创建独立画布
fig, ax = plt.subplots(figsize=(6, 5))
plt.subplots_adjust(bottom=0.15, top=0.92, left=0.15, right=0.95)

x = np.arange(len(labels))

# 绘制柱状图
rects1 = ax.bar(x - bar_width / 2, auc_scores, bar_width, alpha=opacity, color=colors[0],
                hatch=patterns[0], edgecolor='black', label='AUC-ROC')
rects2 = ax.bar(x + bar_width / 2, f1_scores, bar_width, alpha=opacity, color=colors[1],
                hatch=patterns[1], edgecolor='black', label='F1-score')

# 设置标签与标题
ax.set_xlabel(target_param, fontweight='bold')
ax.set_ylabel('Performance')

# 自动提取纯净标题 (去除数学符号)
clean_title = target_param.split(' (')[0]
ax.set_title(f'Effect of {clean_title}')

ax.set_xticks(x)
ax.set_xticklabels(labels)


min_val = min(min(auc_scores), min(f1_scores)) - 0.02
max_val = 1.000
ax.set_ylim(min_val, max_val)


# 网格线与图例
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper left', frameon=True)

plt.show()
