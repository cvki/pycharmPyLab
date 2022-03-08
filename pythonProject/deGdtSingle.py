###buaa 2021-10-26 梯度下降算法
#随机梯度下降
import numpy as np
import matplotlib.pyplot as plt

x0,y0=[],[]
alpha=0.08 #学习率
theta0=1.0 #待求参数k
theta1=1.0 #待求参数b
count=0 #训练轮数

#数据处理，文件数据导入和处理，str为文件名
def datamd(str):
    filestr = []
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
datamd(r"testData1.txt")

x=np.array(x0)
y=np.array(y0)

#梯度下降，每次使用一个样本
def desgdt(xt,yt):
    global theta0,theta1,count
    lth=len(x)
    tmp=xt*theta1+theta0-yt
    theta0-=tmp*alpha
    tmp = (xt*theta1+theta0-yt)*xt
    theta1 -= tmp*alpha
    count+=1
    return

#损失函数，每次计算使用一个样本
def lossfun(xt,yt):
    loss=pow(theta1*xt+theta0-yt,2)
    if count>9995:
        print(loss)
    return loss

#训练
loss = 100
while (loss>1e-8 and count<10000):
    for i,j in zip(x,y):
        desgdt(i,j)
        loss=lossfun(i,j)
print("theta1=",theta1,"theta0=",theta0,'count=',count,'loss=',loss)
print('拟合模型: y=',theta1,'x+',theta0)
print("正规方程解:  [0.80322661 1.18205622]")

#画图
plt.scatter(x,y,color='r')
x1=np.arange(min(x),max(x),0.5)
y1=theta1*x1+theta0
plt.plot(x1,y1,'s-y')
plt.show()



