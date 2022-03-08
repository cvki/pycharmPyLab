
import matplotlib.pyplot as plt

# 支持中文
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

n_samples = np.arange(10)
random_state = np.random.rand(10,10)
# x, y = load_iris(True) # 莺尾花
plt.plot(n_samples,random_state)
plt.title("原始数据分布")
plt.show()
