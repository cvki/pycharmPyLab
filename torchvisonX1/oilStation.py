# 21-12-16  buaa算法大作业——题目4: 汽车加油行驶问题
# 注: 之前都用C/C++做的，这个大作业因为要实现画图，就用python了,比C++方便好用
# 具体的分析和算法内容，在文档中，这里只加一些注释
# python = 3.8.12, numpy=1.21.2, matplotlib=3.4.3

import numpy as np
import random
import matplotlib.pyplot as plt

INF = 9999   #开始的路径权值无穷大

N,K,A,B,C,R=30,3,2,5,4,4   # 可在该处修改参数
# N--地图路线长宽，K--满油最大行驶距离(最大油量)，A--加油付费，B--逆行付费，C--新建油库付费，R--地图上生成两油库间的平均距离
seed = lambda: 1 if random.randint(0, 11) % R == 0 else 0   # 随机种子，用于生成油库点
grid = np.zeros((N + 1, N + 1), dtype=int)   # 地图，1表示该点有油库，0表示该点无油库
oil_xy = []   # 记录油库点横纵坐标(x,y)
bestcount=0    #记录全局最少费用

# 迭代寻路
def findRoute():
    global N, K, A, B, C, bestcount
    # 初始化地图和油库信息
    for i in range(N):
        for j in range(N):
            grid[i + 1][j + 1] = seed()     # 或者手工指定油库
            if grid[i + 1][j + 1] == 1:
                oil_xy.append([i + 1, j + 1])

    # 事务处理矩阵dp
    dp = np.zeros((N + 1, N + 1, 2), dtype=int)
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            dp[i][j][0] = INF   # 第一个二维数组表示(x,y)处cost花费,初始化为INF
            dp[i][j][1] = K   # 第二个二维数组表示当前(x,y)处的剩余油量,初始化为K

    # 4个方向和移动权重，前两个表示右下，花费0，后两个表示逆行左上，花费B
    s = [[-1, 0, 0], [0, -1, 0], [1, 0, B], [0, 1, B]]

    dp[1][1][0], dp[1][1][1] = 0, K    # 初始化动态规划的表格(第一行和第一列)
    tmpx, tmpy = 0, 0
    path = np.zeros((N + 1, N + 1, 2), dtype=int)   # 记录路线矩阵和代价矩阵

    # 开始建立动态规划表并求解
    for x in range(1, N + 1):
        for y in range(1, N + 1):

            if x == 1 and y == 1: continue
            mincost, mink, tmpcost, tmpk = INF, 0, 0, 0    # 分别记录花费和步数

            for i in range(4):  # 4个方向进行试探
                if x + s[i][0] < 1 or x + s[i][0] > N or y + s[i][1] < 1 or y + s[i][1] > N:    # 出界
                    continue
                tmpcost = dp[x + s[i][0]][y + s[i][1]][0] + s[i][2]
                tmpk = dp[x + s[i][0]][y + s[i][1]][1] - 1

                if grid[x][y] == 1:   # 如果是油库
                    tmpcost += A
                    tmpk = K

                if grid[x][y] == 0 and tmpk == 0 and (x != N or y != N):   # 如果不是油库而且油已经用完时，建油库
                    tmpcost += A + C
                    tmpk = K

                if mincost > tmpcost:    # 更新本次花费的cost和step、本次路线的x方向和y方向
                    mincost = tmpcost
                    mink = tmpk
                    tmpx = x + s[i][0]
                    tmpy = y + s[i][1]

            if (dp[x][y][0] > mincost):    # 结束本次试探后，更新最优
                dp[x][y][0] = mincost
                dp[x][y][1] = mink
                path[x][y][0] = tmpx
                path[x][y][1] = tmpy
    bestcount=dp[N][N][0]
    print('min cost of the best route: ',bestcount)   # 输出最少花费

    # 回溯找到最佳路径
    path_getxy = []
    x, y, tmp = N, N, 0
    while ((x != 1) or (y != 1)):
        path_getxy.append([x,y])
        tmp = x
        x = path[x][y][0]
        y = path[tmp][y][1]
    path_getxy.append([x,y])
    return oil_xy, path_getxy


# 绘制最佳路径图
def draw_pic(oil_xy, path_getxy):
    global N
    plt.figure(figsize=(N//2,N//2),dpi=120)   # 背景
    plt.grid(linestyle=":", color="lime",alpha=0.3)    # 网格
    plt.title('Min Cost:'+str(bestcount))
    ax = plt.gca()    # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')    # 将X坐标轴移到上面
    ax.invert_yaxis()   # 反转Y坐标轴
    plt.xticks([x for x in range(1, N + 1)])    # x刻度
    plt.xlabel("axisX")
    plt.yticks([x for x in range(1, N + 1)])    # y刻度
    plt.ylabel("axisY")
    path_getxy=np.array(path_getxy)    # 注意转换ndarray类型，否则报错list
    oil_xy=np.array(oil_xy)
    plt.scatter(oil_xy[:,0], oil_xy[:,1], color="Magenta", label="oilStation")   #散点画出油库点
    plt.scatter(1, 1, color="black") # 起点
    plt.scatter(N, N, color="black") # 终点
    plt.plot(path_getxy[:,0], path_getxy[:,1], ls="-.", lw=2, c="OrangeRed", label="route")     #折线画出路线
    plt.legend(loc='lower left')   # 运行了很多次，发现置于左下最不易遮住路线
    plt.show()   #显示


# 运行
oil_xy, path_getxy = findRoute()
draw_pic(oil_xy, path_getxy)
