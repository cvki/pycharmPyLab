# import numpy as np
# import torch
# import time
#
# # ##1. cpu和gpu计算时间比较
# # print(torch.__version__)
# # print(torch.cuda.is_available())
# #
# # a=torch.randn(10000,10000)
# # b=torch.rand(10000,20000)
# #
# # #使用cpu
# # t0=time.time()
# # c=torch.matmul(a,b)
# # t1=time.time()
# # print(a.device,t1-t0,c.norm(2))
# #
# # #使用GPU
# # device=torch.device('cuda') # cuda要小写
# # a=a.to(device)
# # b=b.to(device)
# #
# # # 第一次运行时，由于一些加载等前置工作，所以时间较长，这个不准
# # t2=time.time()
# # c=torch.matmul(a,b)
# # t3=time.time()
# # print(a.device,t3-t2,c.norm(2))
# # # 第二次运行，此时只进行运算，看它的时间
# # t2=time.time()
# # c=torch.matmul(a,b)
# # t3=time.time()
# # print(a.device,t3-t2,c.norm(2))
#
# ##2.pytorch自动求导,如计算y=a^2 * x + b * x + c
# from torch import autograd
#
# a=torch.tensor(2.,requires_grad=True)
# b=torch.tensor(3.,requires_grad=True)
# c=torch.tensor(4.,requires_grad=True)
# #这里想计算y对abc的偏导，所以x就不用自动求导了
# x=torch.tensor(5.)
# y=a**2 * x + b * x + c
# print("before autograd:",a.grad,b.grad,c.grad)
# # 注意这里要使用grads存储求完的导数，要么输出啥
# grads=autograd.grad(y,[a,b,c]) #表示y对abc分别求偏导
# print("after autograd:",grads)

##3. 线性回归问题

##4. 手写数字识别，MNIST数据集
''' 主要过程: 
 a.导入必要的库/包, import...(可以在下面过程中遇到什么导入什么)
 b.定义超参数和全局模型评估参数, epoch, batch_size...
 c.定义数据(这里是图像)预处理操作, transforms...
 d.下载/加载数据集, dataset, dataloader...
 e.创建网络模型, nn, cov2d/cov3d, relu, softmax...
 f.定义损失函数和优化器, loss, optim...
 g.定义画图, pyplot, tensorboard...
 h.定义训练和测试方法
 i.开始训练 for...
 j.测试并分析结果     '''
import time

from torch import Tensor

'''a. 导入包'''
import torch
import numpy as np
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

# torch.cuda.set_device(0)

'''b. 定义超参和全局模型评估参数'''
EPOCHS=1  # 训练循环次数 (该网络模型Adam优化方法,不需要epoch)
BATCH_SIZE=200  # 每批次处理数据数量
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda或cpu
LR=0.002  # 学习率
# MMTM=0.8      # SGD动量法优化时

# 定义评估参数
acc_f=[]     # 每次Epoch的当前准确率
loss_f=[]    # 每次Epoch的当前loss
var_f=[]     # 每次Epoch的当前方差




'''c. 定义预处理操作,tv.transforms'''
pipline=transforms.Compose([   # Compose用来组合transforms的预处理操作
        transforms.ToTensor(),   # 将图片转为Tensor格式
        transforms.Normalize((0.1307,), (0.3081,))    # 标准化数据集. 这里是MINIST的mean和std，将所有数据展开(6w,1,28,28)，然后用mean和std求解就行，结果一样
        #  注意这里的参数mean和std是sequence,API在官网:
        #  https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html?highlight=normalize#torchvision.transforms.Normalize
        #  这里没有shuffle，shuffle在这里设计没意义, 在Dataloader中加入shuffle才使得操作有意义，因此shuffle在Dataloader中
])

'''d. datasets和DataLoader进行输入数据的加载和封装'''
# 下载和预处理数据, dataset主要负责对数据和数据的预处理(训练前)操作的封装
train_set=datasets.MNIST(root=r'../../Datas/MNIST',train=True,transform=pipline,download=True)
test_set=datasets.MNIST(root=r'../../Datas/MNIST',train=False,transform=pipline,download=True)
# 加载数据, Dataloader主要是负责对数据在训练过程的封装
train_data=DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True)
test_data=DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=False)
# '''可以进入目录查看相关数据集，发现其格式不能直接查看图片，那可以插入python代码，使用python查看其中的图片'''
# with open(r'D:\Pycharm\Datas\MNIST\MNIST\raw\train-images-idx3-ubyte','rb') as f:   #用'rb'即二进制读的权限进行读取(先解码后读取)
#         file=f.read()   # file拿到了所有二进制数据流('rb')，file此时存储的是原生二进制数据流
# # 这里是将拿到的二进制流使用ascii编码显示，16是官方数据的编码解释(第16位开始是图片(0开始),像素为28*28=784)，MNIST的数据集编码格式可以取官网查到
# # 这里先转为str进行’ascii‘方式的encode，而后转为int像素值，注意这里的编码格式，只能用ascii，使用unicode等编码格式会报错(原因可能是在上面python读取时或官方数据集编码时用的ascii)
# image_tmp=[int(str(item).encode('ascii'),16) for item in file[16: 16+784]]   # python:将str转为bytes为encode, 将bytes转为str用decode
# image_t=np.array(image_tmp,dtype=np.uint8).reshape((28,28))    # 将list转为ndarr, 用于保存或显示图片
# cv2.imshow('01',image_t)     # 显示图片
# cv2.waitKey(0)

# 获得一个epoch进行多少次循环:
every_loop=len(train_set)/BATCH_SIZE
print('everyloop: ',every_loop)


'''e. 构建网络模型'''
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Sequential(                               # 输入: batch*1*28*28
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3,padding='same'),    # 得到 batch*10*28*28,因为有padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # 得到 batch*10*14*14,padding左右各填充一层
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=10, out_channels=20,kernel_size=3,padding='same'),   # 得到 batch*20*14*14
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)     # 得到 batch*20*7*7，同理，padding=‘same’模式, 左右各填充一层
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3,padding=0),      # 得到 batch*30*4*4,此时无padding
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)     # 得到 batch*30*2*2
        )
        self.fc=nn.Linear(120,out_features=10)      # 输入features batch*30*2*2 输出为待分类数目
        # self.softmax=nn.Softmax()
    def forward(self,x):        # x为batch输入
        # print('the x dim: ',np.shape(x))
        x=self.conv1(x)
        # print('after conv1: ',np.shape(x))
        x=self.conv2(x)
        # print('after conv2: ',np.shape(x))
        x=self.conv3(x)
        # print('after conv3: ',np.shape(x))      # 三层之后, 此时是batch_size*120
        x=x.reshape(-1,120)     # 自动求出-1，但上面定义的全连接是120,因此如果下一步用那层全连接,第二维必须是120
        # print('after flatten: ', np.shape(x))
        x=self.fc(x)     # torch.flatten(x) 展开为一维向量送入全连接
        # print('after fc: ',np.shape(x))
        # x=self.softmax
        # print('after softmax--end: ',np.shape(x))
        return x

'''注意创建一个模型实例，下面会用到'''
nn_net=Net().cuda()     # 使用默认GPU
print(nn_net)
'''f.定义损失函数和优化器'''
loss_fun=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=nn_net.parameters(),lr=LR)

'''g.定义画图'''
# def plt_md(acc_model,loss_model,var_model):    # 精度图,loss图,方差图
def plt_md(*kwg):    # 精度图,loss图,方差图
    y_loss, y_acc= kwg
    x=np.linspace(1,int(every_loop),int(every_loop))      # 由于比较简单,因此不用EPOCH
    # y_var=var
    plt.plot(x,y_loss,label='loss')
    plt.plot(x,y_acc,label='acc')
    # plt.plot(x,y_var,label='var')
    plt.legend()
    plt.show()


'''h.定义训练和测试方法'''
def train():
    run_loss=0.0        # 每次epoch的损失清零
    run_acc=0.0         # 训练精度
    # 简单多分类问题中，暂时用不到
    # predicate_acc=0.0         # 查准率
    # predicate_all=0.0         # 查全率
    # predicate_opt=[]           # 预测正样本的数目

    # nn_net.train()
    # for epoch in range(EPOCHS):
    #     for batch_idx,data in enumerate(train_data):  # batch轮数,训练集
    #         pic,tag = data    # 解包: (数据，标签)=训练集
    #         pic=pic.cuda()     # 使用默认GPU
    #         tag=tag.cuda()       # 使用默认GPU
    #         # pic,tag=pic.to(DEVICE).float(),tag.to(.to(DEVICE))
    #         tag_ = nn_net(pic).cuda()  # 训练结果tag_
    #         loss = loss_fun(tag_, tag)  # 计算loss
    #         loss.backward()
    #         optimizer.zero_grad()  # 每次epoch训练后梯度清零(因为pytorch等DL框架会一直累积梯度)
    #         optimizer.step()
    #
    #         print('epoch:',epoch,'\t','loss:',loss)

    begin=time.time()
    for epoch in range(EPOCHS):
        for batch_idx,(pic,tag) in enumerate(train_data):  # batch轮数,训练集
            # 数据放到GPU上,.cuda()表示放到默认GPU,指定GPU时用cuda中的device设定
            pic=pic.cuda()
            tag=tag.cuda()
            '前馈网络训练'
            tag_ = nn_net(pic)    # nn_net在实例化已放入GPU,tag_是nn_net返回值,是batchsize*10的shape,是全连接的最后结果,但是注意,
            #在nn_net()中没有e^x和log以及NLLoss,因此它并不是概率值,但它经过e^x和log(softmax,NLLoss)仍然是单调的,那此时值的大小可以理解为概率大小
            # print('tag_:', tag_)
            loss = loss_fun(tag_, tag)  # 计算loss
            '计算准确率,注意理解每一步在干什么'
            tag_v=torch.max(tag_,1)[1].cuda()    # 获得batchsize里最大概率对应的标签
            acc=(tag_v==tag).sum()*1.0/BATCH_SIZE       # 获得预测准的总和,除去总预测的数据条目,化作准确率
            '反向传播','更新梯度'
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # 每次epoch训练后梯度清零(因为pytorch等DL框架会一直累积梯度)
            print('batch:',batch_idx,'\t','loss:',loss,'\t','accurancy:',acc)
            loss_f.append(loss)         # 存储每个batch_size的损失和准确率,用来画图,分析
            acc_f.append(acc)
    loss_plt=torch.Tensor(loss_f).cpu().numpy()        # loss_f和acc_f应该在GPU中,如果不转换到cpu上,会报设备错误
    acc_plt=torch.Tensor(acc_f).cpu().numpy()          #与上同理
    end=time.time()
    print('use time:',end-begin)      # 测试运行时间
    # print(np.shape(acc_plt),'\t',np.shape(loss_plt))    # 输出维度,这里是总数据/batch_size个,所以画图函数中的维度也是这个
    print(loss_plt,'\n',acc_plt)
    # 保存网络模型
    torch.save(nn_net,r'Model\MNIST.pt')
    plt_md(loss_plt,acc_plt)    # 画出loss和accuracy图


def test():
    pass



'''i.开始训练'''


'''j.测试并分析'''



loss=train()








