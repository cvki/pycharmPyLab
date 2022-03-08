import torch
import time

# ##1. cpu和gpu计算时间比较
# print(torch.__version__)
# print(torch.cuda.is_available())
#
# a=torch.randn(10000,10000)
# b=torch.rand(10000,20000)
#
# #使用cpu
# t0=time.time()
# c=torch.matmul(a,b)
# t1=time.time()
# print(a.device,t1-t0,c.norm(2))
#
# #使用GPU
# device=torch.device('cuda') # cuda要小写
# a=a.to(device)
# b=b.to(device)
#
# # 第一次运行时，由于一些加载等前置工作，所以时间较长，这个不准
# t2=time.time()
# c=torch.matmul(a,b)
# t3=time.time()
# print(a.device,t3-t2,c.norm(2))
# # 第二次运行，此时只进行运算，看它的时间
# t2=time.time()
# c=torch.matmul(a,b)
# t3=time.time()
# print(a.device,t3-t2,c.norm(2))

##2.pytorch自动求导,如计算y=a^2 * x + b * x + c
from torch import autograd
a=torch.tensor(2.,requires_grad=True)
b=torch.tensor(3.,requires_grad=True)
c=torch.tensor(4.,requires_grad=True)
#这里想计算y对abc的偏导，所以x就不用自动求导了
x=torch.tensor(5.)
y=a**2 * x + b * x + c
print("before autograd:",a.grad,b.grad,c.grad)
# 注意这里要使用grads存储求完的导数，要么输出啥
grads=autograd.grad(y,[a,b,c]) #表示y对abc分别求偏导
print("after autograd:",grads)

##3. 线性回归问题


##4. 手写数字识别，MNIST数据集





