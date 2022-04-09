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
''' 主要过程: 
 a.导入必要的库/包, import...(可以在下面过程中遇到什么导入什么)
 b.定义超参数, epoch, batch_size...
 c.定义数据(这里是图像)预处理操作, transforms...
 d.下载/加载数据集, dataset, dataloader...
 e.创建网络模型, nn, cov2d/cov3d, relu, softmax...
 f.定义损失函数和优化器, loss, optim... 
 g.定义测试方法
 h.定义画图, pyplot, tensorboard...
 i.开始训练 for...  '''

from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import cv2

EPOCHS=10    # 训练循环次数
BATCH_SIZE=300  # 每批次处理数据数量
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda或cpu
LEARN_RATE=0.07  # 学习率

# 定义预处理
pipline=transforms.Compose([   # Compose用来组合transforms的预处理操作
        transforms.ToTensor(),   # 将图片转为Tensor格式
        transforms.Normalize((0.1307,), (0.3081,))    # 标准化数据集. 这里是MINIST的mean和std，将所有数据展开(6w,1,28,28)，然后用mean和std求解就行，结果一样
        #  注意这里的参数mean和std是sequence,API在官网:
        #  https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html?highlight=normalize#torchvision.transforms.Normalize
        #  这里没有shuffle，shuffle在这里设计没意义, 在Dataloader中加入shuffle才使得操作有意义，因此shuffle在Dataloader中
])
# 下载和预处理数据, dataset主要负责对数据和数据的预处理(训练前)操作的封装
train_set=datasets.MNIST(root=r'../../Datas/MNIST',train=True,transform=pipline,download=True)
test_set=datasets.MNIST(root=r'../../Datas/MNIST',train=False,transform=pipline,download=True)
# 加载数据, Dataloader主要是负责对数据在训练过程的封装
train_data=DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True)
test_data=DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=False)
'''可以进入目录查看相关数据集，发现其格式不能直接查看图片，那可以插入python代码，使用python查看其中的图片'''
with open(r'D:\Pycharm\Datas\MNIST\MNIST\raw','rb') as f:   #用'rb'即二进制读的权限进行读取(先解码后读取)
        file=f.read()   # file拿到了所有二进制数据流('rb')，file此时存储的是原生二进制数据流
# 这里是将拿到的二进制流使用ascii编码显示，16是官方数据的编码解释(第16位开始是图片(0开始),像素为28*28=784)，MNIST的数据集编码格式可以取官网查到
# 这里先转为str进行’ascii‘方式的encode，而后转为int像素值，注意这里的编码格式，只能用ascii，使用unicode等编码格式会报错(原因可能是在上面python读取时或官方数据集编码时用的ascii)
image_tmp=[int(str(item).endcode('ascii'),16) for item in file[16: 16+784]]
print(image_tmp)
cv2.imread(image_tmp)
cv2.imshow('')










