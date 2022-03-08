import numpy as np
import matplotlib.pyplot as plt

# 定义数据集的大小为30
m = 30
# x0为1
x = np.ones((m, 1))
# 设置三个特征向量取值数组的拼接
x = np.concatenate((x, np.arange(1, m + 1).reshape(m, 1), np.arange(2, 2 * m + 1, 2).reshape(m, 1)), axis=1)
y_data = np.loadtxt('data.txt', delimiter=',')
# 从数组提取y的取值
y = np.array(y_data[:, 1][0:30]).reshape(m, 1)
# 设置学习效率
alpha = 0.001
# 设置模型参数theta的初始迭代点(一般都是从0开始)
theta = np.zeros((3, 1))
# 分别定义一个存储损失函数值和迭代次数的列表
costlist, num = [], []


# 定义一个代价函数
def costfunction(theta, x, y):
    error = np.dot(x, theta) - y
    return (1 / 2 * m) * (np.dot(error.T, error))


# 定义一个梯度迭代的函数，即偏导数部分
def gradientdescent(theta, x, y):
    error = np.dot(x, theta) - y
    return (1 / m) * (np.dot(x.T, error))


# 下降迭代过程函数
def GDworkfunction(theta, alpha, x, y):
    gd = gradientdescent(theta, x, y)
    # 迭代10万次终止
    for i in range(100000):
        theta = theta - alpha * gd
        cost1 = costfunction(theta, x, y)
        costlist.append(cost1)
        num.append(i)
        gd = gradientdescent(theta, x, y)
    return theta


consulttheta = GDworkfunction(theta, alpha, x, y)
print('cost值取得最小时的函数为：y={}+{}*x1+{}*x2'.format(consulttheta[0], consulttheta[1], consulttheta[2]))
cost = costfunction(consulttheta, x, y)[0][0]
print('此时的最小损失值为{}'.format(cost))
costlist1 = []
# 对cost值进行特征缩小，缩小的范围自己主观决定，怎么方便怎么来！
for j in costlist:
    costlist1.append(j / 4500)

plt.figure(num=3, figsize=(8, 5))
# 在绘图时，需要对cost值进行降维成一维，因为保存的是二维的数组
plt.plot(num, np.squeeze(np.array(costlist1)))
plt.xlim((1, 80000))
plt.ylim(1, 10)
plt.xlabel('iterate_nums')
plt.ylabel('costvalue')
# 设置x轴刻度表示范围
my_x_ticks = np.arange(1, 80000, 6000)
plt.xticks(my_x_ticks)
plt.savefig('D:')  # 保存图片
plt.show()
