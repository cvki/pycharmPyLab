'''pytorch--MNIST'''

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch.nn import  functional as F

# 1.超参数定义
EPOCH=20
BATCH_SIZE=300

# 2.transforms数据预处理
data_transforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

# 3.datasets封装数据和预处理操作
train_set=datasets.MNIST(root=r'../../Datas/MINIST',train=True,transform=data_transforms,download=True)
test_set=datasets.MNIST(root=r'../../Datas/MINIST',train=False,transform=data_transforms,download=True)

# 4.DataLoader封装datasets与运行时的数据设定和操作
train_data=DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True)
test_data=DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=False)

# 5.定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,10,3,padding='same'),   # 输入batchsize*1*28*28, 输出batchsize*10*26*26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 输入batchsize*10*26*26, 输出batchsize*10*13*13
        ),
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 3, padding='same'),   # 输入batchsize*10*13*13, 输出batchsize*20*11*11
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)    # 输入batchsize*20*11*11, 输出batchsize*20*6*6
        ),
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 30, 3, padding='same'),     # 输入batchsize*20*6*6, 输出batchsize*30*4*4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)      # 输入batchsize*30*4*4, 输出batchsize*30*2*2
        ),
        self.fc=nn.Linear(120, 10)     # 全连接
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.fc(x)
        return x
network=Net()   # 创建一个模型对象
# 6.定义损失函数和优化器
# loss_fun=F.

# 7.定义模型评估量


# 8.画图查看训练过程


# 9.开始训练


# 10.进行测试

