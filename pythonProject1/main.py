import matplotlib.pyplot as plt
import numpy as np

# x=np.array([i for i in np.arange(-10,10,0.5)])
# y1=x+2
# y2=2*(x+2)
# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.show()


# 创建画布
fig = plt.figure(figsize=(12,8),
                 facecolor='lightyellow',
                 dpi=100
                 )

# 创建 3D 坐标系
ax = fig.gca(fc='whitesmoke',
             projection='3d'
             )

# 二元函数定义域
x = np.linspace(-9, 9)
y = np.linspace(-9, 9)
X, Y = np.meshgrid(x, y)

# -------------------------------- 绘制 3D 图形 --------------------------------
# 平面 z=3 的部分
ax.plot_surface(X,
                Y,
                Z=X * 0 + 3,
                color='g'
                )
# 平面 z=2y 的部分
ax.plot_surface(X,
                Y=Y,
                Z=Y * 1,
                color='orange',
                alpha=0.6
                )
# 平面 z=-2y + 10 部分
ax.plot_surface(X=X,
                Y=Y,
                Z=-Y * 2 + 6,
                color='r',
                alpha=0.7
                )
# --------------------------------  --------------------------------

# 设置坐标轴标题和刻度
ax.set(xlabel='X',
       ylabel='Y',
       zlabel='Z',
       xlim=(-9, 9),
       ylim=(-9, 9),
       zlim=(-9, 9),
       xticks=np.arange(-10, 10, 2),
       yticks=np.arange(-10, 10, 1),
       zticks=np.arange(-10, 10, 1)
       )

# 调整视角
ax.view_init(elev=5,  # 仰角
             azim=5  # 方位角
             )

# 显示图形
plt.show()