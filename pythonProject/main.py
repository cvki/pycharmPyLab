
import numpy as np

global x ,y,alpha,theta0,theta1,count
x = [] #特征数据
y = [] #标签
alpha=0.02 #学习率
theta0=1.0 #待求参数k
theta1=1.0 #待求参数b
count=0 #训练轮数

def desgdt():#梯度下降，每次计算使用全部样本
    gdt0,gdt1,lth=0,0,len(x)
    for i,j in zip(x, y):
        gdt0+=(theta1*i+theta0-j)
        gdt1+=(i*theta1+theta0-j)*i
        gdt0 /= lth
        gdt1 /= lth
        theta0-=alpha*gdt0
        theta1-=alpha*gdt1
        count+=1

def lossfun():#损失函数，每次计算使用全部样本
    sumd=0
    for i,j in zip(x,y):
        sumd+=pow(theta1*i+theta0-j,2)
    return sumd/len(x)

def datamd(str):#数据处理,str为文件名
    print(str,type(str))
    filef=open(str,'r')
    typecodeing=filef.encoding
    #print(typecodeing,type(typecodeing))
    filestr = []
    with open(str,'r',encoding=typecodeing) as file:
        filestr=file.read()
    print(file.encoding)
    filestr=filestr.replace('\n',' ')
    i=filestr.split()
    v=[float(tmp) for tmp in i]
    arr=np.array(v)
    for i in range(len(arr)):
        if(i%2):
            y.append(arr[i])
        else:
            x.append(arr[i])
#开始梯度下降法
datamd("D:\\MyApp\\VSCode\\testdata\\torchTest.txt")
loss = lossfun()
while (loss>1e-8 and count<10000):
    desgdt()
    loss=lossfun()
print("theta1=",theta1,"\t","theta0=",theta0)





