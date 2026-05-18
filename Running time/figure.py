import matplotlib.pyplot as plt

# --- Data extraction ---
# 【修改点3】: 更新为最新的基线模型和HTEN数据
methods = [
    'DFN',
    'TA-Struc2Vec',
    'HTEN (Ours)',
    'SplitGNN',
    'CaT-GNN',
    'FraudGNN-RL'
]
latency = [14.5, 15.2, 18.7, 44.0, 51.5, 69.2]

# --- Plotting configuration ---
# Standard academic font family
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] # 推荐使用Times New Roman

# 稍微加宽画板，防止长名字的方法标签重叠
plt.figure(figsize=(10, 6))

# 【修改点1 & 修改点2】: 颜色全部统一为 '#1f77b4'，添加 pattern '//' (在matplotlib中使用hatch参数)
bars = plt.bar(methods, latency,
               color='#1f77b4',       # 统一蓝色
               edgecolor='black',
               hatch='//',            # 设置斜线纹理
               width=0.6,
               zorder=3)

# Add data labels directly on top of each bar for precise reading
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.0,
             f'{yval}', ha='center', va='bottom', fontsize=12)

# --- Labeling and formatting ---
# 标题上方黑色加粗
plt.title('Running Time Comparison',
          fontsize=14, pad=15, fontweight='bold', color='black')

# 坐标轴取消加粗
plt.xlabel('Methods', fontsize=13)
plt.ylabel('Running Time (ms)', fontsize=13)

# X轴标签稍微倾斜防止长名字重叠
plt.xticks(rotation=15, fontsize=12)
plt.yticks(fontsize=12)

# 上方留出一点空间，防止数据标签被标题遮挡
plt.ylim(0, max(latency) * 1.15)

# Add a subtle y-axis grid to help read values
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Adjust layout so labels don't get cut off
plt.tight_layout()

# Save the figure as a high-quality PDF for your paper (optional)
# plt.savefig('running_time_HTEN.pdf', format='pdf', bbox_inches='tight')

# Display the plot
plt.show()