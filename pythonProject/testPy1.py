import numpy as np

global alpha,theta0,theta1,count
x0 = [-1.291, -0.3142, 1.419, 3.027, 4.414, 5.864, 2.838, 3.595, 4.036, 2.018, -0.44, -1.417, 1.766, 2.775, 6.179] #特征数据
y0 = [0.6724, 1.324, 2.48, 3.551, 4.476, 5.443, 3.455, 4.01, 4.144, 2.778, 1.21, 0.559, 2.823, 3.391, 5.661] #标签
alpha=0.02 #学习率
theta0=1.0 #待求参数k
theta1=1.0 #待求参数b
count=0 #训练轮数

global x,y,gdt0,gdt1
gdt0,gdt1=0,0
x=np.array(x0)
y=np.array(y0)

# a,b=0,0
# for i,j in zip(x,y):
#     a=(i+2-j)
#     b=(j*3-5)
# print(a,"\n",b)

lth=len(x)
tmp=x.dot(theta1)+theta0-y
gdt0-=np.sum(tmp)/lth
print(type(gdt0))
print(x.shape)
tmp = (x.dot(theta1) + theta0 - y)*x
gdt0 -= np.sum(tmp) / lth
gdt1 -= np.sum(tmp) / lth
print(gdt0,gdt1)
sumd=0
for i, j in zip(x, y):
    sumd += pow(theta1 * i + theta0 - j, 2)
print(sumd / lth)