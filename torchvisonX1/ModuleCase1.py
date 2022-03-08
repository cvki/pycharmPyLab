'''21-12-16'''
import numpy as np
import cv2
from torch.utils import tensorboard
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn

'''本文件主要进行nn.module的练习'''


trans_pic=transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize(size=(720,720)),
    transforms.ToTensor()
])

trainsets_CF10=datasets.CIFAR10(r'dataset',train=True,transform=trans_pic,download=True)
testsets_CF10=datasets.CIFAR10(r'dataset',train=False,transform=trans_pic,download=True)
dataset_CF10=trainsets_CF10+testsets_CF10
dataLoader_CF10=DataLoader(dataset_CF10,batch_size=5,shuffle=False,drop_last=False)

# wzx=cv2.imread('wzx.jpg')
# wzx=trans_pic(dataLoader_CF10)
# print(wzx.shape)
class ModuleCs1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3,stride=1,padding=0)
        # self.conv2=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=1,stride=1,padding=0)

    def forward(self,x): # 该函数为自定义的前向函数，不是Module自带的
        x=self.conv1(x)
        # nn.ReLU()
        # x=self.conv2(x)
        return x

# a=np.random.randint(8,size=(2,))
# print(a,'\n',np.shape(a),np.ndim(a))

writer=tensorboard.SummaryWriter(r'log\stepCF10')
md1=ModuleCs1()
for data in dataLoader_CF10:
    imgs,targets=data
    writer.add_image('befor', imgs)
    dt1=md1(imgs)
    writer.add_image('after', dt1)
    print(type(dt1))








