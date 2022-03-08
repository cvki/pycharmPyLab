
import numpy as np
import torch

# '''Tensor和numpy中的ndarray基本一样
#     两者的相互转换: '''
# t = torch.ones(5)
# print(type(t),f"t: {t}")
# n = t.numpy()
# print(type(n),n.shape,f"n: {n}")
#
# ’在这里改变t或者n，另一个也会改变。这说明两者所用内存是一样的‘
# t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")
#
# n = np.ones(5)
# t = torch.from_numpy(n)
#
# np.add(n, 1, out=n)
# print(f"t: {t}")
# print(f"n: {n}")
#
'************************************************************************************************************************'
# '''Tensor初始化方法:
#     1.通过直接赋值，
#     2.通过numpy的ndarray类型
#     3.通过另一个Tesnor'''
# data = [[1, 2],[3, 4]]
# x_data = torch.tensor(data)
#
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
#
# x_ones = torch.ones_like(x_data) # retains the properties ofr x_data
# print(f"Ones Tensor: \n {x_ones} \n")
#
# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")

'************************************************************************************************************************'
# 'torch.cat()和torch.stack()的区别'
# t24=torch.tensor(np.array(range(12)).reshape(3,4))
# t24m=torch.tensor(np.array(range(-30,-18)).reshape(3,4))
# print(t24,'\n',t24m)
# 'stack()函数多一个维数，比如下面的拼接矩阵，在第一个参数中用多少子矩阵拼接，最终的第三维数就是多少'
# 'cat()函数将子矩阵按照行列放入新矩阵中，维数不发生变化，子矩阵拼接完后还是两维数矩阵'
# '它们都有dim参数，dim取值[-2,1],一般常用1和0，表示行列拼接，-2和-1会将子矩阵展开成向量，然后按行列拼接'
# r1=torch.cat([t24,t24m],1)
# r2=torch.stack([t24,t24m,t24,t24,t24],-1)
# print(r1,'\n',r2)
# print(r2.shape)

'************************************************************************************************************************'
# '矩阵乘法: 矩阵元素相乘和矩阵相乘'
# t33=torch.tensor(np.array(range(9)).reshape(3,3))
# t33m=-t33
# print(t33,'\n',t33m)
# t1=t33*t33m     #矩阵元素相乘
# t2=t33@t33m     #矩阵相乘
# t3=t33.mul(t33m)    #矩阵元素相乘
# t4=t33.matmul(t33m)     #矩阵相乘
# print(t1,'\n',t2,'\n',t3,'\n',t4)

'************************************************************************************************************************'
# '赋值和自动赋值'
# t33=torch.tensor(np.array(range(9)).reshape(3,3))
# print(t33, "\n")
# t33.add_(5) #函数后面加’_‘后，会原地操作
# print(t33)
#
# t33=torch.tensor(np.array(range(9)).reshape(3,3))
# print(t33, "\n")
# t33.add(5)#函数后面不加’_‘，则生成副本
# print(t33)
# '''相对于一般赋值的比较:，自动赋值的优缺点:
#     1.由于自动赋值是原地操作，因此节省了内存和计算时间
#     2.但对于需要中间过程(副本)的算法来说，是不能用的，因为它不生成中间副本，所以无法使用。比如反向传播'''



from torch.utils.data import dataset





