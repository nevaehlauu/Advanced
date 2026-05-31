import matplotlib.pyplot as plt
import numpy as np

data = np.array([0, 0, 0, 1, 1, 3, 3, 4, 4, 6, 6])

# 定义颜色映射
color_map = {0: 'green', 1: 'red', 3: 'orange', 4: 'yellow', 6: 'purple'}

# 绘制曲线
fig, ax = plt.subplots(figsize=(8, 12))
ax.plot(data, np.arange(len(data)), '-')

# 在曲线上标出不同颜色所表示的数字
for i, val in enumerate(data):
    ax.text(val, i, str(val), color=color_map[val], va='center', ha='right', fontsize=10)

ax.set_ylim(len(data)-1, -1)
ax.set_xlabel('Value')
ax.set_ylabel('Depth')
plt.savefig("1.jpg")
plt.close()