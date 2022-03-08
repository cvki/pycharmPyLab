import os
import numpy as np
import torch
import cv2.cv2 as cv
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


"第一类，数据格式和文件处理"
# class Mydata(Dataset):
#     def __init__(self,path_a,label):
#         self.path_dir=os.path.join(path_a,label)
#         self.path_list=os.listdir(self.path_dir)
#         self.label=label
#     def __len__(self):
#         return len(self.path_list)
#     def __getitem__(self, item):
#         #opencv 显示图片
#         # pic=cv.imread(os.path.join(self.path_dir,self.path_list[item]))
#         # cv.imshow(self.label+str(item),pic)
#         # cv.waitKey(1500)
#         # cv.destroyAllWindows()
#         return self.path_list[item],self.label
#
# train_ants=Mydata(r'hymenoptera_data\train','ants')
# train_bees=Mydata(r'hymenoptera_data\train','bees')
# train_data=train_ants+train_bees

# for i in range(120,130):
#     print(train_data.__getitem__(i))

'''将第一类文件格式转化为第二类文件存储格式'''
# #os.mkdir("data2")
# class Mydata2(Dataset):
#     def __init__(self):
#         pass
#     def __getitem__(self, item):
#         pass
#     def __len__(self):
#         pass


'''tensorboard使用'''
#tensorboard terminal 设置端口： tensorboard --logdir=文件夹名 --port=端口号
#查看tensorboard日志内容： tensorboard --logdir=文件路径


'''测试画曲线图：'''
# writer1=SummaryWriter('logs\log_test1')
# # # #会报错：AssertionError: scalar should be 0D
# # # x=np.arange(20)
# # # for j in np.arange(4):
# # #     writer1.add_scalar('y={}x'.format(j), j*x, x)
# #改为：
# for i in range(4):
#     for j in range(20):
#         writer1.add_scalar('y={}x'.format(i), i*j, j)
# writer1.close()


'''测试显示图片(图片的测试数据)'''
# 方法1，报错,img_tensor使用string，报错，原因：没有搞清楚string/blobname是什么含义
# writer2=SummaryWriter('logs\log_test2')
# m_pth=r'hymenoptera_data\train\ants'
# vlabel='ants'
# m_ant_list=os.listdir(m_pth)
# print(m_ant_list)
# for i in range(6,10):
#     vname=os.path.join(m_pth,m_ant_list[i])
#     writer2.add_image(vlabel+str(i),vname,i)
#     #报错：Can't find blob: hymenoptera_data\train\ants\1262877379_64fcada201.jpg. 这说明img_tensor的string/blobname不是指的图像路径文件名，而是有其他的含义
# writer2.close()


'''法2，使用数组ndarray来做'''
# writer3=SummaryWriter('logs\log_test2')
# m_pth=r'hymenoptera_data\train\ants'
# m_ant_list=os.listdir(m_pth)
# img=[]
# for i in range(6,10):
#     vname=os.path.join(m_pth,m_ant_list[i])
#     img.append(cv.imread(vname))
# for i in range(4):
#     writer3.add_image('ants{}'.format(i),img[i],dataformats='HWC') #不在一个图中
#     #writer3.add_image('ants{}'.format(i), img[i], i, dataformats='HWC')  #也不在一个图中
#     #writer3.add_image('ants', img[i], i, dataformats='HWC')  #在一个图中,这说明，第一个参数targ用来控制是否在一个图中
#     #注意图片格式，opencv读写图片是HWC，这里默认格式是CHW,需要使用dataformats参数
# writer3.close()


'''Transform的使用 '''
'1.Opencv读取图像(ndarray格式)并转换为tensor格式'
imgt=cv.imread(r'hymenoptera_data/train/ants/0013035.jpg')
#print(np.shape(imgt))
tmg_tensort=transforms.ToTensor()(imgt)
'2.求各通道均值和标准差，注意opcv读取格式为GBR，PIL读取格式才是RGB'
g_arr=imgt[:,:,0]  #切通道
b_arr=imgt[:,:,1]
r_arr=imgt[:,:,2]
g_mean=np.mean(g_arr)  #求均值
b_mean=np.mean(b_arr)
r_mean=np.mean(r_arr)
g_std=np.std(g_arr)     #求标准差
b_std=np.std(b_arr)
r_std=np.std(r_arr)
#print((g_mean,b_mean,r_mean),(g_std,b_std,r_std))
'3.图像标准化'
normalize=transforms.Normalize([g_mean,b_mean,r_mean],[g_std,b_std,r_std])
#normalize=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
tmg_tensortm=normalize(tmg_tensort)
#print(tmg_tensortm)
'tensorboard显示图像'
writer4=SummaryWriter(r'logs\log_test3')
writer4.add_image('normalize_img',tmg_tensortm,1) #虽然有负值，但在日志中能正常显示，所以这里肯定会对图像负值像素做了处理

'尝试用给CV2显示图像，报错'
# cv.imshow('std_img',tmg_tensortm) #因为有负值，而且该类型是tensor，无法显示图像，所以该方法肯定未图像负值像素做处理
# 'cv2.error: OpenCV(4.5.4-dev) :-1: error: (-5:Bad argument) in function 'imshow''
# cv.waitKey(0)

"Resize的使用"
imgp=Image.open(r'hymenoptera_data/train/bees/16838648_415acd9e3f.jpg')
v=transforms.ToTensor()(imgp)
writer4.add_image('bees1',v,0)
print(imgp.size)
tmg_resize=transforms.Resize((368,512))
tmg_tensor2=tmg_resize(imgp)
print(tmg_tensor2)
u=transforms.ToTensor()(tmg_tensor2)
writer4.add_image('bees1',u,1)


'Compose用法：'

writer4.close() #不关闭会报错，提示没有读到内容




