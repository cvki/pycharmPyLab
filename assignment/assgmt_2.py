'buaa_CV_assignment2,Butterworth高通滤波算子'

import numpy as np
import matplotlib.pylab as plt
import cv2.cv2 as cv

def BtrWrth_up(img):
    height, width = img.shape
    fft = np.fft.fftshift(np.fft.fft2(img))    # 傅里叶变换
    mask0 = np.zeros(img.shape, np.float32)
    R0 = 160  # 截止频率     # 30, 60, 160
    n = 2  # 阶数
    for i in range(0, height):
        for j in range(0, width):
            Rxy = ((i - height / 2) ** 2 + (j - width / 2) ** 2) ** (1 / 2)
            mask0[i, j] = 1 - 1 / (1 + (Rxy / R0) ** (2 * n))   # 低 + 高 = 1

    p0 = fft * mask0    # 相乘
    new0 = np.abs(np.fft.ifft2(np.fft.ifftshift(p0)))     # 反变换
    new0 = (new0 - np.amin(new0)) / (np.amax(new0) - np.amin(new0))     # 调整大小范围便于显示
    plt.figure(), plt.title('origin pic'), plt.imshow(img, 'gray'), plt.xticks([]), plt.yticks([])
    plt.figure(), plt.title('after shift'), plt.imshow(new0, 'gray'), plt.xticks([]), plt.yticks([])
    plt.show()

path='Chap02_2_sample.tif'
img=cv.imread(path,flags=0)
BtrWrth_up(img)
