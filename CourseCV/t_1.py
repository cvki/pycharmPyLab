'''buaa_cv,作业1: 实现图像分段线性变换，实现图像直方图均衡化'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 界面显示
def plot_img(*img):
    fig=plt.figure(figsize=(100,70),dpi=150)
    for i in range(2):
        print(img[i].shape)
        plt.subplot(1,2,i+1)
        plt.imshow(img[i],cmap='gray')
        plt.title('pic'+str(i+1))
    plt.show()

# 线性变换
def linear_transpose(img,low_in=0,high_in=1,low_out=0,high_out=1):
    if high_in-low_in<=0 or high_out-low_out<=0:    # 输入有误
        raise "PIXEL RANGE INPUT ERROR!"
    k=(high_out*1.-high_in)/(low_out*1.-low_in) #斜率
    height, width = img.shape
    # print(img.shape) # height*width
    res=img.astype(np.float32)  # 存储变换后的图片
    # 对像素进行处理变换
    for i in range(height):
        for j in range(width):
            # 只变换指定范围内的像素
            if (img[i,j]>=low_in and img[i,j]<=high_in):
                res[i,j]=k*(img[i,j]-low_in)+low_out   # 线性变换

            # elif img[i,j]<low_in:
            #     img[i,j]=low_out
            # else:
            #     img[i,j]=high_out

    return res.astype(np.uint8)


#直方图均衡化
def hist_normalization(img,grade):    # 图片路径和灰度级数
    # 获取直方图p(r)
    imhist, bins = np.histogram(img.flatten(), grade, normed=True)
    # 获取T(r)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = grade * cdf / cdf[-1]
    # 获取s，并用s替换原始图像对应的灰度值
    result = np.interp(img.flatten(), bins[:-1], cdf)
    return result.reshape(img.shape), cdf


#test

path='Chap02_1_sample.tif'
img=cv.imread(path,flags=0)   # 以灰度图读入
img_l=linear_transpose(img,100,120,50,200) # 指定分段线性变换
# img_h=hist_normalization(img,255)
# plot_img(img,img_l,img_h)
plot_img(img,img_l)



