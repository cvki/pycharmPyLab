'''buaa_cv,作业1: 实现图像分段线性变换，实现图像直方图均衡化'''

import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button,RadioButtons

# 中文显示
def set_chinese():
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False

# 界面显示
def plot_every_img():
    fig=plt.figure(figsize=(120,70),dpi=230)
    fig.add_subplot(121)


# 线性变换
def linear_transpose(path,low_in=0,high_in=1,low_out=0,high_out=0):
    if low_out-low_in<=0 or high_out-high_in<=0:    # 输入有误
        raise "PIXEL RANGE INPUT ERROR!"
    k=(high_out*1.-high_in)/(low_out*1.-low_in) #斜率
    img=cv.imread(path,flags=0)   # 以灰度图读入
    height, width = img.shape
    # print(img.shape) # height*width
    res=img.astype(np.float32)  # 存储变换后的图片
    # 对像素进行处理变换
    for i in range(height):
        for j in range(width):
            # 只变换指定范围内的像素
            if (img[i,j]>=low_in and img[i,j]<=high_in):
                res[i,j]=k*(img[i,j]-low_in)+low_out   # 线性变换
    res=res.astype(np.uint8)

# 对数变换



# 伽马变换


#直方图均衡化
def hist_normalization(path,grade):    # 图片路径和灰度级数
    img = cv.imread(path, flags=0)  # 以灰度图读入
    # 获取直方图p(r)
    imhist, bins = np.histogram(img.flatten(), grade, normed=True)
    # 获取T(r)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]
    # 获取s，并用s替换原始图像对应的灰度值
    result = np.interp(img.flatten(), bins[:-1], cdf)
    return result.reshape(img.shape), cdf

# 手动实现反色变换
def color_inverse(input):   #input为输入图像矩阵
    res=np.asarray(input)
    max_val = np.max(input)
    height, width,chanel=np.shape(res)
    for i in range(height):
        for j in range(width):
            res[i,j]=max_val-input[i,j]
    return res

# linear_transpose("img.png",10,50,100,200)
img= np.asarray(Image.open("px.png"))
res=color_inverse(img)

