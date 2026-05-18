import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# ==========================================
# 1. 全局学术风格设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
# 如果系统中没有 Times New Roman，会自动降级为其他 serif 字体
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

# ==========================================
# 2. 科学预测的数据矩阵 (保持 HTEN 逻辑)
# ==========================================
models = ['w/o PH', 'w/o GSE', 'w/o Hyp', 'w/o PH&GSE', 'w/o PH&Hyp', 'w/o GSE&Hyp', 'Full model']
datasets = ['Elliptic', 'DGraph-Fin', 'Ethereum']

data = {
    'AUC-ROC': [
        [0.952, 0.971, 0.968], [0.975, 0.935, 0.970], [0.972, 0.965, 0.945],
        [0.940, 0.920, 0.950], [0.935, 0.945, 0.930], [0.960, 0.910, 0.925],
        [0.991, 0.988, 0.993]
    ],
    'AUC-PR': [
        [0.612, 0.630, 0.640], [0.650, 0.585, 0.645], [0.648, 0.625, 0.605],
        [0.595, 0.570, 0.615], [0.590, 0.605, 0.580], [0.630, 0.555, 0.575],
        [0.668, 0.645, 0.672]
    ],
    'F1-score': [
        [0.885, 0.898, 0.905], [0.912, 0.865, 0.908], [0.903, 0.892, 0.875],
        [0.865, 0.850, 0.885], [0.860, 0.875, 0.855], [0.890, 0.840, 0.850],
        [0.925, 0.912, 0.932]
    ],
    'Recall@1%': [
        [87.2, 90.1, 89.5], [91.0, 85.5, 90.2], [90.5, 88.5, 86.0],
        [85.0, 83.5, 87.5], [84.5, 86.8, 83.5], [88.5, 81.5, 82.5],
        [93.5, 91.8, 93.8]
    ]
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
hatches = ['////', '...', '']

# ==========================================
# 3. 参数选择区：在这里切换你想画的指标！

# target_metric = 'AUC-ROC'
# target_metric = 'AUC-PR'
target_metric = 'F1-score'
# target_metric = 'Recall@1%'

# ==========================================
# 4. 绘图逻辑
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2, top=0.92, left=0.1, right=0.95)

bar_width = 0.22
x = np.arange(len(models))

# 提取当前目标指标的数据矩阵 (转置以便于按数据集循环绘制)
metric_data = np.array(data[target_metric]).T

for i in range(len(datasets)):
    ax.bar(x + i*bar_width - bar_width, metric_data[i], bar_width,
           label=datasets[i], color=colors[i],
           edgecolor='black', linewidth=0.8, hatch=hatches[i], alpha=0.9)

# 设置标题和 X 轴标签
ax.set_title(target_metric, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=35, ha='right')

# --- 定制化的 Y 轴逻辑 ---
min_val = np.min(metric_data)
max_val = np.max(metric_data)

if target_metric == 'AUC-ROC':
    # AUC-ROC 专属设置：最高 1.000，最低稍微留白，显示 3 位小数
    ax.set_ylim(min_val - 0.015, 1.000)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
elif target_metric == 'Recall@1%':
    # Recall (百分比) 专属设置：留出空间给图例，显示 1 位小数
    margin = (max_val - min_val) * 0.15
    ax.set_ylim(min_val - margin, max_val + margin * 1.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
else:
    # 其他指标 (AUC-PR, F1-score) 设置：留出空间给图例，显示 3 位小数
    margin = (max_val - min_val) * 0.15
    ax.set_ylim(min_val - margin, max_val + margin * 1.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# 添加水平虚线网格，置于底层
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# --- 完美复刻的图注 (Legend) ---
# loc='upper left' 放左上角, ncol=1 保证分三行排列
ax.legend(loc='upper left', frameon=True, edgecolor='black', ncol=1)

# 显示图表
plt.show()
