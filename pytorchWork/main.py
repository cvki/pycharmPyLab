import numpy as np

x = [] #��������
y = [] #��ǩ
alpha=0.02 #ѧϰ��
theta0=1.0 #
theta1=1.0 #
count=0 #ѵ������

def desgdt():#�ݶ��½���ÿ�μ���ʹ��ȫ������
    gdt0,gdt1,lth=0,0,len(x)
    for i,j in zip(x, y):
        gdt0+=(theta1*i+theta0-j)
        gdt1+=(i*theta1+theta0-j)*i
        gdt0 /= lth
        gdt1 /= lth
        theta0-=alpha*gdt0
        theta1-=alpha*gdt1
        count+=1

def lossfun():#��ʧ������ÿ�μ���ʹ��ȫ������
    sumd=0
    for i,j in x,y:
        sumd+=pow(theta1*i+theta0-j,2)
    return sumd/len(x)

def datamd(str):#���ݴ���,strΪ�ļ���
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
    #��ʼ�ݶ��½���
    loss = lossfun()
    while (loss>1e-8 and count<10000):
        desgdt()
        loss=lossfun()
print("theta1=",theta1,"\t","theta0=",theta0)



