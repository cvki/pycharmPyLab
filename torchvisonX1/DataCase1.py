'''21-12-15'''
'''本文件中，主要包含tensorboard，torchvison.datasets，dataloader的练习,transform回顾'''

import torchvision
from torch.utils.tensorboard import SummaryWriter
# DataLoader & Dataset 同时使用时一起导入
from torch.utils.data import Dataset,DataLoader


#dataset、datasets和dataloader的区别

#先transform来定义预处理操作，再加载transform到数据集中
dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor() #由于图片本身较小，因此只做Totensor，若正常大图，可裁剪，resize等transform的其它操作
])#将预处理操作放到Compose里封装

#运行后会有网址提示，建议复制该连接去迅雷下载，然后将下载的离线包放在指定目录中
trainSet=torchvision.datasets.CIFAR10(root=r'../DL_Course/Datas/CIFAR10',train=True,transform=dataset_transform,download=False)  #只要数据集下载过，那他就不会重复下载了，建议一直设置为True
testSet=torchvision.datasets.CIFAR10(root=r'../DL_Course/Datas/CIFAR10',train=False,transform=dataset_transform,download=False)  #这里由于上传git很慢，就改成了false

# print(trainSet[0]) #从输出看，它包含图片信息和target两部分
# print(trainSet.classes) #标签类别,以字典存储，可以ctrl+left
# print(testSet.classes[trainSet[0][1]])
# trainSet[0][0].show()

'''之后数据被变成tensor，可以用tensorboard进行展示了，正好回顾复习一下'''
# for i in range(7,17):
#     imgi, targeti = trainSet[i]
#     #print(imgi,targeti)
#     writer1=SummaryWriter(r'log\step1')
#     writer1.add_image('pic10',imgi,targeti)
# writer1.close()

'''DatalLoader使用'''
testLoader=DataLoader(testSet,batch_size=24,shuffle=True,drop_last=False)
#CITAR10的__getitem__返回值
imgi,targeti=testSet[0]
print(imgi.shape)

writer2=SummaryWriter(r'log\step2')
step=2
for batchi in testLoader: #同理，batchi得到的是img和target，两个都是tensor
    #print(batchi[0].shape) #imgi的格式，batchsize一组的图像，4*3*32*32
    #print(batchi[1].shape) #标签数组，一维，batchsize个，即4
    #print(batchi)
    ##writer2.add_image('testdata',batchi[0],step)
    ##注意要使用方法add_images(),而不是add_image()否则会报错
    ##报错size of input tensor and input format are different. tensor shape: (4, 3, 32, 32), input_format: CHW
    imgi,targeti=batchi
    writer2.add_images('testData',imgi,step)
    step+=1     #每个for循环作为一步来查看debug
writer2.close()



