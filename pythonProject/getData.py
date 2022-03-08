import numpy as np

global x ,y
x = [] #特征数据
y = [] #标签

def datamd(str):#数据处理,str为文件名
    print(str,type(str))
    filef=open(str,'r')
    typecodeing=filef.encoding
    print(typecodeing,type(typecodeing))
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

datamd("D:\\MyApp\\VSCode\\testdata\\torchTest.txt")
print(x,'\n',y)