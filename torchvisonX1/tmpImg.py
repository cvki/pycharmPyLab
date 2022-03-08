import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

# imgnp=cv2.imread('wzx.jpg')
# print(np.shape(imgnp))
# imgsub1=imgnp[:,:,0]
# imgsub2=imgnp[:,:,1]
# imgsub3=imgnp[:,:,2]
#
# # cv2.imshow('imgsub1',imgsub1)
# # cv2.waitKey(2000)
# # cv2.imshow('imgsub2',imgsub2)
# # cv2.waitKey(2000)
# # cv2.imshow('imgsub3',imgsub3)
# # cv2.waitKey(2000)
# # cv2.imshow('imgnp',imgnp)
# # cv2.waitKey(2000)
#
# plt.subplot(2,2,1)
# plt.imshow(imgsub1)
# plt.subplot(2,2,2)
# plt.imshow(imgsub2)
# plt.subplot(2,2,3)
# plt.imshow(imgsub3)
# plt.subplot(2,2,4)
# plt.imshow(imgnp)
# plt.show()

# M=2
# N=4
# a=np.array([[1, 1], [1, 5], [1, 9], [2, 3], [2, 4], [3, 4], [4, 6], [4, 8], [5, 1], [5, 2], [5, 6], [5, 8], [7, 6], [7, 8], [7, 9], [8, 1], [9, 2], [9, 6]])
# # print(a.shape)
# ax=a[:,0]
# ay=a[:,1]
# print(ax)
# print(ay)


# class MyNumbers:
#   def __iter__(self):
#     self.a = 9
#     return self
#
#   def __next__(self):
#     self.a -= 1
#     return self.a

# myclass = MyNumbers()
# myiter = iter(myclass)
#
# v=next(myiter)
# while v:  # 必须要有结束条件
#     print(v)
#     v = next(myiter)

# print(next(myiter))
# print(next(myiter))
# print(next(myiter))
# print(next(myiter))
# print(next(myiter))

# class MyNumbers:
#   def __iter__(self):
#     self.a = 0
#     return self
#
#   def __next__(self):
#     # if self.a <= 20:
#       # x = self.a
#       # self.a += 1
#       # return x
#     ## 或者
#     if self.a < 20:
#       self.a += 1
#       return self.a
#
#     else:
#       raise StopIteration
#
# myclass = MyNumbers()
# myiter = iter(myclass)
#
# for x in myiter:
#   print(x)


'''对高维数据的一个循环迭代: np.nditer(arr)'''
# arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# for x in np.nditer(arr):
#   print(x)

# arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# for x in np.nditer(arr[:, ::2]):
#   print(x)

## nditer不会就地修改dtype，因此需要辅助空间flags
# arr = np.array([1, 2, 3])
# for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
#   print(x)

## ndenumerate() 进行枚举迭代，迭代时需要元素的相应索引
# arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# for idx, x in np.ndenumerate(arr):
#   print(idx, x)


''' np数组的连接(或者stack()函数，或者vstack和hstack)'''
## dstack() 沿高度堆叠，该高度与深度相同
# arr1 = np.array([[1, 2], [3, 4]])
# arr2 = np.array([[5, 6], [7, 8]])
# arr = np.concatenate((arr1, arr2), axis=0)
# print(arr)


''' 分割1 或者使用vsplit和hsplit'''
# arr = np.array([1, 2, 3, 4, 5, 6])
# newarr = np.array_split(arr, 3)
# print(newarr)
## 分割2
# arr = np.array([1, 2, 3, 4, 5, 6])
# newarr = np.array_split(arr, 4)
# print(newarr)
## 结果: [array([1, 2]), array([3, 4]), array([5]), array([6])]

# ## np.where()
# arr = np.array([[1, 2, 3, 4, 5, 6],[9,10,11,12,13,14]])
# x = np.where(arr%2)  #返回的是符合条件元素的索引
# print(x)


''' np数组排序'''
# arr1 = np.array([3, 2, 0, 1])
# ## 注释：此方法返回数组的副本，而原始数组保持不变。还可以对字符串数组或任何其他数据类型进行排序
# ## 布尔数组进行排序， 对n-D数组排序
# arr2 = np.array(['banana', 'cherry', 'apple'])
# print(np.sort(arr1))
# print(sorted(arr1))   # 也不能改变原数组，说明它也是对arr进行的副本操作
# print(arr1)
# print(np.sort(arr2))
# arr3 = np.array([[[3, 2, 4], [5, 0, 1]],[[1,6,4],[9,8,5]]])
# print(np.sort(arr3))
# arr4 = np.array([True, False, True])
# print(np.sort(arr4))
# # ## 输出:
# # [0 1 2 3]
# # ['apple' 'banana' 'cherry']
# # [[[2 3 4]
# #   [0 1 5]]
# #
# #  [[1 4 6]
# #   [5 8 9]]]
# # [False  True  True]


''' 搜索排序:有一个名为 searchsorted() 的方法，该方法在数组中执行二进制搜索，并返回将在其中插入指定值以维持搜索顺序的索引。'''
# arr = np.array([6, 7, 8, 9])
# x = np.searchsorted(arr, 7)
# print(x)
# ## 输出结果: 1, 表示插入后它在第2个位置
# ## 我们可以给定 side='right'，以返回最右边的索引: x = np.searchsorted(arr, 7, side='right'),此时输出2，同理只是从右边数
# arr = np.array([1, 3, 5, 7])
# x = np.searchsorted(arr, [2, 4, 6])
# print(x)


'''数组过滤'''
'''NumPy 中，我们使用布尔索引列表来过滤数组。尔索引列表是与数组中的索引相对应的布尔值列表。
如果索引处的值为 True，则该元素包含在过滤后的数组中；如果索引处的值为 False，则该元素将从过滤后的数组中排除'''
# arr = np.array([61, 62, 63, 64, 65])
# x = [True, False, True, False, True]
# newarr = arr[x]
# print(newarr)

## 创建过滤器数组, 当然有更方便的方法：## 更方便的方法:np.where(arr>62)
## 不过当过滤规则比较复杂，需要自定义时，where就不好用了。

# arr = np.array([61, 62, 63, 64, 65])
# # 创建一个空列表
# filter_arr = []
# # 遍历 arr 中的每个元素
# for element in arr:
#   # 如果元素大于 62，则将值设置为 True，否则为 False：
#   if element > 62:
#     filter_arr.append(True)
#   else:
#     filter_arr.append(False)
# newarr = arr[filter_arr]
# print(filter_arr)
# print(newarr)

## 更方便的方法，相当于where
# arr = np.array([61, 62, 63, 64, 65])
# filter_arr = arr > 62   # 还比如 filter_arr=arr%2==0 # 即取偶数
# newarr = arr[filter_arr]
# print(filter_arr)
# print(newarr)


'''随机意味着无法在逻辑上预测的事物,计算机在程序上工作,程序一定是有逻辑的。如果存在生成随机数的程序，则可以预测它，因此它就不是真正的随机数。
我们通过生成算法生成的随机数称为伪随机数。我们可以从某个外部来源获取随机数据。外部来源通常是我们的击键、鼠标移动、网络数据等。这种就是真随机数。
但在这里面，我们使用伪随机数（一般是跟时间相关的随机，即由时间(CPU时钟)根据确定好的公式进行随机，因此是伪随机）'''

#  random.randint(100)   # 生成一个 0 到 100 之间的随机整数
#  x = random.rand()   # 生成一个 0 到 100 之间的随机浮点数
#  random.randint(100, size=(5))   # 生成一个 1-D 数组，其中包含 5 个从 0 到 100 之间的随机整数
#  x = random.rand(5)    # 生成包含 5 个随机浮点数的 1-D 数组

# ## choice()方法
# ## choice() 方法使您可以基于值数组生成随机值。choice() 方法将数组作为参数，并随机返回其中一个值或数组。
# x = np.random.choice([3, 5, 7, 9])
# print(x)
# x = np.random.choice([3, 5, 7, 9], size=(3, 5))
# print(x)


'''ufuncs 指的是“通用函数”（Universal Functions），它们是对 ndarray 对象进行操作的 NumPy 函数'''
'''ufunc 用于NumPy中实现矢量化，这比迭代元素要快得多。还提供广播和其他方法，例如减少、累加等，对计算非常有帮助'''
## zip()方法，
# x = [1, 2, 3, 4]
# y = [4, 5, 6, 7]
# z = []
# for i, j in zip(x, y):
#   z.append(i + j)
# print(z)
'''zip还可以遍历列表，元组，字符串等等。另外，zip还能两个容器x,y长度不一样的，默认多余部分被丢弃'''

## enumerate
'''解决了我们又想直接迭代又需要知道元素下标的情形.enumerate还支持传入参数。
比如在某些场景当中，我们希望下标从1开始，而不再是0开始，我们可以额外多传入一个参数实现这点'''
# x = [1, 2, 3, 4]
# z = []
# for i, j in enumerate(x):
#   print(i,j)

## 还有:
# x = [1, 2, 3, 4]
# z = []
# for i, j in enumerate(x,2): #指定下标从2开始.是0索引变成了2，而不是从2处截断数组输出
#   print(i,j)
# ## 输出:
# # 2 1
# # 3 2
# # 4 3
# # 5 4

'''需要注意，如果我们迭代的是一个多元组数组，我们需要注意要将index和value区分开。举个例子'''
# data = [(1, 3), (2, 1), (3, 3)]
## 在不用enumerate的时候，我们有两种迭代方式，这两种都可以运行。
# for x, y in data:
#   for (x, y) in data:
#       pass
# ## 但是如果我们使用enumerate的话，由于引入了一个index，我们必须要做区分，否则会报错，所以我们只有一种迭代方式：
# for i, (x, y) in enumerate(data):
#     pass


'''除此还有一些高级API，如-
  loadtxt，加载数据，以二进制加载，
  readtxt，读取数据，以二进制读取，
  savetxt，保存数据，以二进制保存，
  ones，zeros，和生成对角阵...
  mearn,average,median,var,std...
  percentile查看百分位数:percentile(ages, 75) #查看百分之75位置的分位数
  transpose,T 矩阵转置
  dot,@   矩阵乘积
  解方程函数，求行列式，判断矩阵相似...
  eig，求特征值和特征向量
  随机产生矩阵，随机产生分布型的矩阵数据uniform,normal...
  
  等等...需要就查API
'''


'''scipy小小练习————一元线性回归练习'''
# import matplotlib.pyplot as plt
# from scipy import stats
#
# # x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# # y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
#
# x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
# y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
#
# # y=np.random.normal(5,1.6,1000)   #均值，方差，数据个数
# # x=np.array([i+1 for i in range(1000)])
#
# # slope为斜率，intercept为截距，std_err为标准差，p是？？
# # r为度量x轴的值和y轴的值之间的关系(相关系数),r平方值的范围是0到1，其中0表示不相关，而1表示100％相关
# slope, intercept, r, p, std_err = stats.linregress(x, y)
# def myfunc(x):
#   return slope * x + intercept
# mymodel = list(map(myfunc, x))
# print(r)
# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()


'''小小练习————多项式回归'''
# import numpy
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# # 创建表示 x 和 y 轴值的数组：
# x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
# y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
# # NumPy 有一种方法可以让我们建立多项式模型：
# mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
# # 然后指定行的显示方式，我们从位置 1 开始，到位置 22 结束：
# myline = numpy.linspace(1, 22, 100)
# #相关系数来评价模型
# print(r2_score(y, mymodel(x)))
# # 绘制原始散点图：
# plt.scatter(x, y)
# # 画出多项式回归线：
# plt.plot(myline, mymodel(myline))
# # 显示图表：
# plt.show()


'''pandas多元回归小练习'''
# import pandas
# from sklearn import linear_model
# df = pandas.read_csv("cars.csv")    #pd读取csv文件，这里得有文件才行
# X = df[['Weight', 'Volume']]    #关键字为数据中的标题
# y = df['CO2']
# regr = linear_model.LinearRegression()
# regr.fit(X, y)
# print(regr.coef_)   #输出特征的系数
# # 预测重量为 2300kg、排量为 1300ccm 的汽车的二氧化碳排放量：
# predictedCO2 = regr.predict([[2300, 1300]])
# print(predictedCO2)










