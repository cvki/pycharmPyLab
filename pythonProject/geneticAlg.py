import numpy as np

SELECT=3000 #自然选择循环次数
CITYNUM=10 #城市数目
GEN=100 #每代保留个体数(染色体数)
GALL=2*GEN-1 #每代总体数目(子代+父代)
MUTMOUNT=2 #每代变异个体数,加概率
willselect=np.zeros((GALL,CITYNUM+1),dtype=int)  #每代种群，待选，包含父代和子代
citystr=[] #以列表形式保存城市名，(基因)编码即为城市下标序号
city={} # 以字典存储城市名称和经纬度
gen_solution=np.zeros((GEN,CITYNUM+1),dtype=int) #每一代的优秀解空间,最后一位表示回到起点
gen_weight=np.zeros(GALL,dtype=float) #存储每一代权值
OPTSOV=[[1e8],[]] #存储最终的最优解

#传入文件数据处理
def dataRead(filestr):
    with open(filestr,'r',encoding='utf-8') as file:
        datacity=file.readlines()
    for i in datacity:
        citystr.append(i.split(';')[0].strip())
        x=float(i.split(';')[1].strip())
        y=float(i.split(';')[2].strip())
        city[i.split(';')[0].strip()]=[x,y]

#计算每代权重矩阵，权重用路径权和表示,作为适应度函数，适应度越高，值越小
def genWeight(v):
    for i in np.arange(GALL):
        x=willselect[i]
        sum=0.0
        # max,min,idx=0.0,0.0,0
        for j in np.arange(CITYNUM-1):
            v1=city.get(citystr[x[j+1]])
            v2=city.get(citystr[x[j]])
            sum+=np.sqrt(pow(v1[0] - v2[0], 2) + pow(v1[1] - v2[1], 2))
        gen_weight[i]=sum

# #适应度函数。规则是解越好，则适应度越高。而路径权和与其成反比(其实用适应度低的选择也行)，不能简单使用路径权和，这里参考老师按比例的适应度函数设计
# def envSuit(v):
#     max,min,val=v
#     return (max-val)/(max-min+0.00001) #分母防止为0

#交叉规则
CROSSBEGIN=np.random.randint(CITYNUM) #随机生成基因交叉起始位
CROSSEND=np.random.randint(CITYNUM) #随机生成基因交叉结束位
#不交叉最后一位，因为最后一位和第一位一致，交叉结束后修改最后一位的值
#保证起始位<=结束位
if(CROSSBEGIN>CROSSEND):
    TMP=CROSSBEGIN
    CROSSBEGIN=CROSSEND
    CROSSEND=TMP
def crossMethod(vf,vm): #交叉规则：交换染色体上x个基因
    for i in np.arange(CROSSBEGIN,CROSSEND):
        if(CROSSEND!=CROSSBEGIN):
            tmp1 = vm[i]
            tmp2 = vf[i]
            #vf内部交换
            idf=np.argwhere(vf==tmp1)[0]
            tmp=vf[i]
            vf[i]=vf[idf]
            vf[idf]=tmp
            #vm内部交换
            idm = np.argwhere(vm == tmp2)[0]
            tmp = vm[i]
            vm[i] = vm[idm]
            vm[idm] = tmp
    vf[CITYNUM]=vf[0]
    vm[CITYNUM]=vm[0]
    #print(vf,vm)

def genCross(v): #交叉操作
    lenv=np.shape(v)[0]
    #print(lenv)
    for i in np.arange(lenv-1):
        crossMethod(v[i],v[i+1])

def genMutate(v): #变异规则和操作
    lenv = np.shape(v)[0]
    for i in np.arange(lenv):
        #随机选定两个位置(不含首尾)，进行染色体内的基因互换，表示变异
        tmpint=np.random.randint(1,CITYNUM-1,[1,2])
        v[i,tmpint[0,0]],v[i,tmpint[0,1]]=v[i,tmpint[0,1]],v[i,tmpint[0,0]]

'''算法开始'''
#读取文件，处理数据
dataRead(r'cityDis.txt')

#随机生成初代
for i in np.arange(GEN):
    a=np.random.choice(CITYNUM,size=(1,CITYNUM),replace=False)
    np.random.shuffle(a)
    a=np.append(a,a[0,0])
    gen_solution[i]=a
#print(gen_solution)

#初始化相关信息
# genWeight(gen_solution)
# maxid,maxweight=np.argmax(gen_weight,axis=0)[1],np.max(gen_weight,axis=0)[1]
# minid,minweight=np.argmin(gen_weight,axis=0)[1],np.min(gen_weight,axis=0)[1]
# # print(gen_weight)
# # print(maxid,maxweight,minid,minweight)
# print(gen_solution)
# genCross(gen_solution)

# genWeight(gen_solution)
# print(gen_weight)

#自然选择大循环
for selectN in np.arange(SELECT):#
    for i in np.arange(GEN):
        willselect[i] = gen_solution[i]
    genCross(gen_solution)# 交叉
    for i in np.arange(GEN,GALL): #记录父代和子代
        willselect[i]=gen_solution[GEN-i]

    #求该代所有个体的适应度,并得到下一代优秀群体
    genWeight(willselect)
    idx1=np.argsort(gen_weight)[:GEN]
    gen_solution = willselect[idx1]

    #变异
    if(not np.random.randint(0, 1000, 1)):#千分之一变异
        randv=np.random.randint(0,GEN,2)
        mval = []
        for i in np.arange(MUTMOUNT):
            mval.append(gen_solution[randv[i]])
        mval=np.array(mval)
        genMutate(mval)
        #将变异的个体放入种群原位置
        for i in np.arange(MUTMOUNT):
            gen_solution[randv[i]]=mval[i]

    # 存储当前最优个体
    minweight = np.min(gen_weight[idx1[0]])
    minroute = willselect[idx1[0]]
    if (np.random.randint(0, 1000, 1)):  # 千分之一概率不替换，应对局部最优问题
        if (OPTSOV[0] > minweight):
            OPTSOV[0] = minweight
            OPTSOV[1] = minroute

#输出最优解
print(OPTSOV)
ctag = 0
for i in OPTSOV[1]:
    if(ctag):
        print('-->',citystr[i],end='')
    else:
        print(citystr[i],end='')
    ctag+=1

















# #计算两城市之间的距离
# def getdist(str1,str2):#传入城市名,也可以直接传入经纬度提高效率
#     v1=city.get(str1)
#     v2=city.get(str2)
#     return math.sqrt(pow(v1[0]-v2[0],2)+pow(v1[1]-v2[1],2))
#
# #生成城市距离的邻接矩阵（这是个对称阵，这里可以取对角线一侧优化计算）
# def getMatDist():
#     for i in range(CITYNUM):
#         for j in range(CITYNUM):
#             city_distMat[i][j]=getdist(citystr[i],citystr[j])
