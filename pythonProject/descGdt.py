###buaa 2021-10-26 梯度下降算法
#批量梯度下降
import numpy as np
import matplotlib.pyplot as plt

x0,y0=[],[]
alpha=0.05 #学习率
theta0=1.0 #待求参数k
theta1=1.0 #待求参数b
count=0 #训练轮数

#数据处理，文件数据导入和处理，str为文件名
def datamd(str):
    #filestr = []
    with open(str,'r',) as file:
        filestr=file.read()
    #print(file.encoding)
    filestr=filestr.replace('\n',' ')
    i=filestr.split()
    v=[float(tmp) for tmp in i]
    arr=np.array(v)
    for i in range(len(arr)):
        if(i%2):
            y0.append(arr[i])
        else:
            x0.append(arr[i])
#传入文件
datamd(r'testData1.txt')

x=np.array(x0)
y=np.array(y0)

#梯度下降，每次计算使用全部样本
def desgdt():
    global theta1,theta0,count
    lth=len(x)
    tmp1=x.dot(theta1)+theta0-y
    theta0-=np.sum(tmp1)/lth*alpha
    tmp2 = (x.dot(theta1) + theta0 - y)*x
    theta1 -= np.sum(tmp2) / lth*alpha
    count+=1
    return

#损失函数，每次计算使用全部样本
def lossfun():
    sumd=0
    for i,j in zip(x,y):
        sumd+=pow(theta1*i+theta0-j,2)
    if count>9995:
        print(loss)
    return sumd / 2 / len(x)

#训练
loss = lossfun()
while (loss>1e-8 and count<10000):
    desgdt()
    loss=lossfun()
print("theta1=",theta1,"theta0=",theta0,'count=',count,'loss=',loss)
print('拟合模型: y=',theta1,'x+',theta0)

#正规方程
a=np.ones(shape=x.shape)
xt=np.array([x,a])
tmp=np.linalg.pinv(np.dot(xt,xt.T))
val=np.dot(np.dot(tmp,xt),y)
print("正规方程解: ",val)

#画图
plt.scatter(x,y,color='r')
x1=np.arange(min(x),max(x),0.5)
y1=theta1*x1+theta0
plt.plot(x1,y1,'s-y')
plt.show()



