import math
import numpy as np

SELECT = 500  # 群体选择次数
CITYNUM = 10  # 城市数目
GEN = 60  # 每代个体数(染色体数)
OUTSTANDING = 8  # 每代选的最优个体数
MUTMOUNT = 2  # 每次变异个体数
willselect = np.zeros((2 * GEN, CITYNUM + 1), dtype=int)  # 种群最大时，待选，包含父代和子代
citystr = []  # 以列表形式保存城市名，(基因)编码即为城市下标序号
city = {}  # 以字典存储城市名称和经纬度
gen_solution = np.zeros((GEN, CITYNUM + 1), dtype=int)  # 每一代的解空间(种群),最后一位表示回到起点
gen_weight = np.zeros((GEN, 2), dtype=float)  # 存储每一代权值
for i in np.arange(GEN):  # 初始化，i为个体(染色体)编号，这里从0开始
    gen_weight[i, 0] = i


# 传入文件数据处理
def dataRead(filestr):
    with open(filestr, 'r', encoding='utf-8') as file:
        datacity = file.readlines()
    for i in datacity:
        citystr.append(i.split(';')[0].strip())
        x = float(i.split(';')[1].strip())
        y = float(i.split(';')[2].strip())
        city[i.split(';')[0].strip()] = [x, y]


# 计算每代权重矩阵(这里距离来描述，权重越小距离越短),用来表示适应度，其值越小越好
def genWeight(v):
    for i in np.arange(GEN):
        x = gen_solution[i]
        sum = 0.0
        max, min, idx = 0.0, 0.0, 0
        for j in np.arange(CITYNUM - 1):
            v1 = city.get(citystr[x[j + 1]])
            v2 = city.get(citystr[x[j]])
            sum += math.sqrt(pow(v1[0] - v2[0], 2) + pow(v1[1] - v2[1], 2))
        gen_weight[i, 1] = sum


# #适应度函数。规则是解越好，则适应度越高。而路径权和与其成反比(用适应度低的选择也行)，不能简单使用路径权和，这里参考老师按比例的适应度函数设计
# def envSuit(max,min,val):
#     return (max-val)/(max-min+0.00001) #分母防止为0

# 交叉规则
CROSSBEGIN = np.random.randint(CITYNUM)  # 随机生成基因交叉起始位
CROSSEND = np.random.randint(CITYNUM)  # 随机生成基因交叉结束位
# 不交叉最后一位，因为最后一位和第一位一致，交叉结束后修改最后一位的值
# 保证起始位<=结束位
if (CROSSBEGIN > CROSSEND):
    TMP = CROSSBEGIN
    CROSSBEGIN = CROSSEND
    CROSSEND = TMP


def crossMethod(vf, vm):  # 交叉规则：交换染色体上x个基因
    for i in np.arange(CROSSBEGIN, CROSSEND):
        if (CROSSEND != CROSSBEGIN):
            tmp1 = vm[i]
            tmp2 = vf[i]
            # vf内部交换
            idf = np.argwhere(vf == tmp1)[0]
            tmp = vf[i]
            vf[i] = vf[idf]
            vf[idf] = tmp
            # vm内部交换
            idm = np.argwhere(vm == tmp2)[0]
            tmp = vm[i]
            vm[i] = vm[idm]
            vm[idm] = tmp
    vf[CITYNUM] = vf[0]
    vm[CITYNUM] = vm[0]
    print(vf, vm)


def genCross(v):  # 交叉操作
    lenv = np.shape(v)[0]
    print(lenv)
    for i in np.arange(lenv - 1):
        crossMethod(v[i], v[i + 1])


def genMutate(v):  # 变异规则和操作
    lenv = np.shape(v)[0]
    for i in np.arange(lenv):
        # 随机选定两个位置(不含首尾)，进行染色体内的基因互换，表示变异
        tmpint = np.random.randint(1, CITYNUM - 1, [1, 2])
        v[i, tmpint[0, 0]], v[i, tmpint[0, 1]] = v[i, tmpint[0, 1]], v[i, tmpint[0, 0]]


'''算法开始'''
# 读取文件，处理数据
dataRead(r'cityDis.txt')

# 随机生成初代
for i in np.arange(GEN):
    a = np.random.choice(10, size=(1, CITYNUM), replace=False)
    np.random.shuffle(a)
    a = np.append(a, a[0, 0])
    gen_solution[i] = a

# 初始化相关信息
# 权重矩阵计算和排序
genWeight(gen_solution)

# 取权重矩阵最值和其索引
# maxid,maxweight=np.argmax(gen_weight,axis=0)[1],np.max(gen_weight,axis=0)[1]
# minid,minweight=np.argmin(gen_weight,axis=0)[1],np.min(gen_weight,axis=0)[1]
# print(gen_weight)
# print(maxid,maxweight,minid,minweight)
# print(gen_solution)
# genCross(gen_solution)

# 自然选择大循环
for selectN in np.arange(SELECT):
    # 每代种群内部的个体循环
    for per in np.arange(GEN):
        pass

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
