from numpy.core.fromnumeric import mean

'使用第一次MNIST，复用cifar10'  #4层CNN,未进行数据增强，0.75准确率

# import tensorflow as tf
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import cv2 as cv
import os
from torch.utils.tensorboard import SummaryWriter

# tpu训练时部分代码
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)
# strategy = tf.distribute.experimental.TPUStrategy(tpu)

'''b. 定义超参和全局模型评估参数'''
EPOCHS = 17 # 训练循环次数 (该网络模型Adam优化方法,不需要epoch)
BATCH_SIZE = 200  # 每批次处理数据数量
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda或cpu
LR = 0.005  # 学习率
# MMTM=0.8      # SGD动量法优化时ws

# 定义和存储评估参数
acc_f = []  # 每次Epoch的当前准确率
loss_f = []  # 每次Epoch的当前loss
var_f = []  # 每次Epoch的当前方差
test_loss = []  # test-loss
test_acc = []  # test-acc

# tensorboard使用
writer=SummaryWriter(r'logs/cifar10_cnn/')

'''c. 定义预处理操作,tv.transforms'''
pipline = transforms.Compose([  # Compose用来组合transforms的预处理操作
    transforms.ToTensor(),  # 将图片转为Tensor格式
    transforms.Normalize((torch.Tensor([0.4914, 0.4821, 0.4465])), (torch.Tensor([0.2470, 0.2435, 0.2616])))
    # 标准化数据集. 这里是CIFAR10的mean和std，将所有数据展开(6w,1,28,28)，然后用mean和std求解就行，结果一样
    #  注意这里的参数mean和std可以查询或者去官网
    #  这里没有shuffle，shuffle在这里设计没意义, 在Dataloader中加入shuffle才使得操作有意义，因此shuffle在Dataloader中
])

'''d. datasets和DataLoader进行输入数据的加载和封装'''
# 下载和预处理数据, dataset主要负责对数据和数据的预处理(训练前)操作的封装
train_set = datasets.CIFAR10(root=r'./Datas/CIFAR10', train=True, transform=pipline, download=True)
test_set = datasets.CIFAR10(root=r'./Datas/CIFAR10', train=False, transform=pipline, download=True)
# 加载数据, Dataloader主要是负责对数据在训练过程的封装
train_data = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)
# '''可以进入目录查看相关数据集，发现其格式不能直接查看图片，那可以插入python代码，使用python查看其中的图片'''
# with open(r'D:\Pycharm\Datas\MNIST\MNIST\raw\train-images-idx3-ubyte','rb') as f:   #用'rb'即二进制读的权限进行读取(先解码后读取)
#         file=f.read()   # file拿到了所有二进制数据流('rb')，file此时存储的是原生二进制数据流
# # 这里是将拿到的二进制流使用ascii编码显示，16是官方数据的编码解释(第16位开始是图片(0开始),像素为28*28=784)，MNIST的数据集编码格式可以取官网查到
# # 这里先转为str进行’ascii‘方式的encode，而后转为int像素值，注意这里的编码格式，只能用ascii，使用unicode等编码格式会报错(原因可能是在上面python读取时或官方数据集编码时用的ascii)
# image_tmp=[int(str(item).encode('ascii'),16) for item in file[16: 16+784]]   # python:将str转为bytes为encode, 将bytes转为str用decode
# image_t=np.array(image_tmp,dtype=np.uint8).reshape((28,28))    # 将list转为ndarr, 用于保存或显示图片
# cv2.imshow('01',image_t)     # 显示图片
# cv2.waitKey(0)

# 训练时，计算一个epoch进行多少次循环:
print(len(train_set))
print(len(test_set))
every_train_loop = len(train_set) / BATCH_SIZE
print('every_train_loop: ', every_train_loop)
# 测试时，计算一个epoch进行多少次循环:
every_test_loop = len(test_set) / BATCH_SIZE
print('every_test_loop: ', every_test_loop)

'''e. 构建网络模型'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # 输入: batch*3*32*32
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),  # 得到 batch*10*31*31,因为有padding
            # nn.BatchNorm2d(10),      # num_features:为输入的数据的通道数,即特征数。通过测试，加上该层收敛较快，波动较小，有dropout的感觉，不加收敛较慢，波动较小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 得到 batch*10*16*16,padding左右各填充一层
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),  # 得到 batch*20*16*16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 得到 batch*20*8*8，同理，padding=‘same’模式, 左右各填充一层
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),  # 得到 batch*30*15*15
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 得到 batch*30*4*4
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),  # 得到 batch*50*7*7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 得到 batch*128*2*2
        )
        self.fc = nn.Linear(512, out_features=10)  # 输入features batch*30*2*2 输出为待分类数目
        # self.softmax=nn.Softmax()     # 这里加上softmax，后面再用loss交叉熵，很容易下溢

    def forward(self, x):  # x为batch输入
        # print('the x dim: ',np.shape(x))
        input_size = x.size(0)  # batch_size*1*28*28,那size(0)=batch_size
        x = self.conv1(x)
        # print('after conv1: ',np.shape(x))
        x = self.conv2(x)
        # print('after conv2: ',np.shape(x))
        x = self.conv3(x)
        # print('after conv3: ',np.shape(x))      # 三层之后, 此时是batch_size*120
        x = self.conv4(x)
        # print('after conv3: ',np.shape(x))
        x = x.reshape(-1, 512)  # 自动求出-1，但上面定义的全连接是120,因此如果下一步用那层全连接,第二维必须是120
        # print('after flatten: ', np.shape(x))
        # x=x.view(input_size,-1)   # 将x拉成向量，inputsize已知时，-1处自动计算它的值
        x = self.fc(x)  # torch.flatten(x) 展开为一维向量送入全连接
        # print('after fc: ',np.shape(x))
        # x=self.softmax(x)     # 这里加上softmax，后面再用loss交叉熵，很容易下溢
        # print('after softmax--end: ',np.shape(x))
        return x  # x不是概率或损失，而是全连接的最终输出值


'''注意创建一个模型实例，下面会用到'''
nn_net = Net().cuda()  # 使用默认GPU
print(nn_net)

'''f.定义损失函数和优化器'''
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=nn_net.parameters(), lr=LR)

'''g.定义画图'''
# def plt_md(acc_model,loss_model,var_model):    # 精度图,loss图,方差图
def plt_md(str, *kwg):  # 精度图,loss图,方差图
    y_loss, y_acc, x_range = kwg
    x = np.linspace(0, x_range,x_range)  # x的范围
    # y_var=var
    ytick = np.arange(0, 2.5, 0.1)  # 设置y的坐标和间隔标准
    plt.xlabel('x axis')  # 设置x和y的标题标签
    plt.ylabel('y axis')
    plt.title(str)
    plt.yticks(ytick)  # 设置y的坐标和间隔
    plt.plot(x, y_loss, label='loss')
    plt.plot(x, y_acc, label='acc')
    # plt.plot(x,y_var,label='var')
    plt.legend()
    plt.savefig(r'./pic/pic_{}.png'.format(str))
    plt.close()     # 因为要多次用到它画图，画完一个后要关闭，否则前面的plt内容会累积到后面的plt中


'''h.定义训练和测试方法'''
def train(epoch):
    run_loss = 0.0  # 每次epoch的损失清零
    run_acc = 0.0  # 训练精度
    # 简单多分类问题中，暂时用不到
    # predicate_acc=0.0         # 查准率
    # predicate_all=0.0         # 查全率
    # predicate_opt=[]           # 预测正样本的数目

    nn_net.train()  # 模型训练

    for batch_idx, (pic, tag) in enumerate(train_data):  # batch轮数,训练集
        # 数据放到GPU上,.cuda()表示放到默认GPU,指定GPU时用cuda中的device设定
        pic = pic.cuda()
        tag = tag.cuda()
        '前馈网络训练'
        tag_ = nn_net(pic)  # nn_net在实例化已放入GPU,tag_是nn_net返回值,是batchsize*10的shape,是全连接的最后结果,但是注意,
        # 在nn_net()中没有e^x和log以及NLLoss,因此它并不是概率值,但它经过e^x和log(softmax,NLLoss)仍然是单调的,那此时值的大小可以理解为概率大小
        # print('tag_:', tag_)

        # 计算每轮平均损失和精度
        loss_avg_tm = []
        acc_avg_tm = []

        '计算loss'
        loss = loss_fun(tag_, tag)
        loss_f.append(loss.item())  # 存储每个batch_size的损失和准确率,用来画图,分析
        loss_avg_tm.append(loss.item())

        '计算准确率,注意理解每一步在干什么'
        # tag_v=torch.max(tag_,1)[1]    # 获得batchsize里最大概率对应的标签,或者使用下面的一句
        tag_v = torch.argmax(tag_, dim=1)
        acc = (tag_v == tag).sum() * 1.0 / BATCH_SIZE  # 获得预测准的总和,除去总预测的数据条目,化作准确率,注意这里的准确率计算是分batch的，而不是整个训练集
        acc_f.append(acc.item())
        acc_avg_tm.append(acc.item())

        '增加一个tensorboard来查看分错的图片'
        if(tag_v.item()!=tag.item()):
            writer.add_image('TRAIN:epoch{}'.format(epoch),pic,tag_v)

        '反向传播'
        loss.backward()
        '更新梯度'
        optimizer.step()
        optimizer.zero_grad()  # 每次epoch训练后梯度清零(因为pytorch等DL框架会一直累积梯度)
        # 不加item()会输出该Tensor，里面有具体数据，也有些其他的，比如Device设备等
        print('batch_idx:{}\tepoch:{}\tloss:{:.5}\taccurancy:\t{:.3}'.format(batch_idx, epoch, loss.item(), acc.item()))

        # 计算本轮epoch的平均损失和精度
    loss_tm = mean(loss_avg_tm)
    acc_tm = mean(acc_avg_tm)
    return loss_tm, acc_tm  # 返回本次epoch的平均loss和acc


def test():
    # 加载模型
    nn_net = torch.load(r'./Module/CIFAR10.pt')
    # test不用epoch
    nn_net.eval()  # 模型验证
    with torch.no_grad():  # 因为这边是测试，因此不需要梯度，或者说不用计算梯度，也不能反向传播
        # 部署数据到CUDA
        for (data, tag) in test_data:  # 这里用enumerate()会出问题
            data, tag = data.to(DEVICE), tag.to(DEVICE)  # 放入cuda
            tag_test = nn_net(data)  # 进行测试
            los_avg = 0.0  # 存储平均损失
            v_t = None

            # 计算损失
            loss = loss_fun(tag_test, tag)
            # test_loss.append(loss_t.item())     # 将损失放到test-loss中
            loss_t = 0.0
            loss_t += loss.item()  # 在测试中，test_loss保存的是累积损失值
            test_loss.append(loss_t)  # 加到全局损失变量中，用于分析和作图

            # 计算准确度
            acc_count = torch.argmax(tag_test, dim=1).to(DEVICE)  # 获取测试tag的标签值并放于CUDA中
            acc_t = (acc_count == tag).sum() * 1.0 / BATCH_SIZE  # 计算测试准确度,注意，这里也是分batch测试的，而不是整个test_set
            test_acc.append(acc_t.item())

            # 或者使用下面的代码计算准确度
            # pred = tag_test.max(1, keepdim=True)[1]  # 获取预测类别值
            # correct = 0.0
            # correct += pred.eq(tag_test.view_as(pred)).sum().item()
            # v_t.append(np.array([acc_t.item(), correct]))
            print('TEST---loss:{}\taccurancy:{}'.format(loss, acc_t))

        los_avg = np.mean(test_loss)  # 计算平均损失
        acc_avg = np.mean(test_acc)  # 计算平均精度
        # print('v_t',v_t)
        print('TEST---: ', 'los_avg:', los_avg, '\t', 'acc_avg:', acc_avg)
        plt_md('test_pic', test_loss, test_acc, int(every_test_loop))  # 画出测试图


'''i.开始训练'''
def train_on():
    epoch_avg = []  # 记录每轮平均信息
    los_avg = np.array([])
    acc_avg = np.array([])

    begin = time.time()

    for epoch in np.arange(1, EPOCHS + 1):
        los, acc = train(epoch)
        los_avg = np.append(los_avg, los)  # 注意这里要再赋值给原数组，因为这方法不改变原数组(append,insert都一样)
        acc_avg = np.append(acc_avg, acc)
        str = ''.join("THE EPOCH{}: LOSS---{} '\t' acc---{},".format(epoch, los, acc))
        epoch_avg.append(str)
        str = ''
        # print("THE EPOCH{}: LOSS---{}\tacc---{}".format(epoch,los,acc))
    end = time.time()
    print('use time:', end - begin)  # 测试运行时间

    # 使用每轮的每个batch进行画图，当训练量大了，可用上面的每个epoch的均损失和均精度进行画图分析
    loss_plt = torch.Tensor(loss_f).cpu().numpy()  # loss_f和acc_f应该在GPU中,如果不转换到cpu上,会报设备错误
    acc_plt = torch.Tensor(acc_f).cpu().numpy()  # 与上同理
    # print(np.shape(acc_plt),'\t',np.shape(loss_plt))    # 输出维度,这里是总数据/batch_size个,所以画图函数中的维度也是这个
    # print(loss_plt, '\n', acc_plt)
    # 保存网络模型
    torch.save(nn_net, r'./Module/CIFAR10.pt')
    plt_md('train_pic_batch', loss_plt, acc_plt, int(EPOCHS * every_train_loop))  # 画出以batch为基的loss和accuracy图
    plt_md('train_pic_epoch', los_avg, acc_avg, EPOCHS)  # 画出以epoch为基的loss和accuracy图
    print(epoch_avg)


train_on()

'''j.测试并分析'''
test()

'''识别自己的手写数据(自己画图测试的几个)'''


class GetImg:
    def __init__(self, pathx):
        self.pathx = pathx  # 文件路径
        self.dirs = os.listdir(pathx)  # 所有文件名
        self.files = []  # 存放每个文件的路径+文件名
        for filename in self.dirs:
            file_tmp = os.path.join(pathx, filename)  # 获取每个文件的路径+文件名
            self.files.append(file_tmp)

    def pre_possess(self, path):
        idx = np.arange(1, len(self.dirs) + 1)
        for i, imgpath in zip(idx, self.files):  # 这里注意zip和enumerate的区别
            print(type(i))
            img = cv.imread(imgpath)  # 通过文件路径读图片
            imgv = cv.resize(img, (28, 28))  # resize并保存，用来测试
            # 批量保存图片并自动命名
            cv.imwrite(path + '\\' + 'pic%d.jpg' % i, imgv)
            # cv.imshow('img{}'.format(i),imgv)
            # cv.waitKey(0)

    def getfile(self):
        return self.pathx, self.files


# 该例子已经处理好自己手写的图片了，就不再运行
# gi = GetImg(r'D:\Pycharm\workplace\DLPytorch\Model\Manual_num')
# gi.pre_possess()

def test_man(path):  # path为手写图片的存储路径
    res = []  # 用来存储预测值
    nn.net = torch.load(r'./Module/CIFAR10.pt')  # 加载模型
    img_files = GetImg(path)  # 加载图片
    # img_files.pre_possess(path)     # 这里已经有处理好的图片了，所以就执行它了
    # print(img_files.getfile())
    _, pics = img_files.getfile()  # 我们只需要self.files，即图片路径+图片名
    for imgs in pics:
        img_np = cv.imread(imgs, flags=0)  # 灰度图读入
        # print(img_np.shape)      #28*28
        img_np = img_np.reshape(1, 1, 28, 28)  # batchsize=1,扩充通道,读取维度在Net中有分析
        img_tensor = torch.Tensor(img_np)  # 要将数据放入cuda，否则报两设备错误
        tag_ = nn_net(img_tensor)  # 计算结果
        # print(tag_)
        res.append(torch.argmax(tag_, dim=1))
    print(res)  # 注: 1和6很少出错，3和5几乎不出错，4基本不对(通过resize可以看出，4比较像7了，这和预处理相关)
    # 主要原因还是，自己写的数据和国外测试数据差别是有的，除了相对复杂的3和5，写起来比较相近，其他的写法有些差异，这是数据特征的问题
    # 其次，预处理图片时，对图片的主要特征保留也至关重要(比如这里的4预处理后变成7，就失真了)，所以预处理也是很重要的，必要时需要进行预训练

# path=r'D:\Pycharm\workplace\DLPytorch\Model\Manual_num\pic_28'
# test_man(path)

