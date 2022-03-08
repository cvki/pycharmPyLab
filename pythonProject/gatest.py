import numpy as np

# CITYNUM=6
# GEN=5
# gen_solution=np.zeros((GEN,CITYNUM+1),dtype=int)
#
# #交叉规则
# CROSSBEGIN=np.random.randint(CITYNUM) #随机生成基因交叉起始位
# CROSSEND=np.random.randint(CITYNUM) #随机生成基因交叉结束位
# #不交叉最后一位，因为最后一位和第一位一致，交叉结束后修改最后一位的值
# #保证起始位<=结束位
# if(CROSSBEGIN>CROSSEND):
#     TMP=CROSSBEGIN
#     CROSSBEGIN=CROSSEND
#     CROSSEND=TMP
# def crossMethod(vf,vm): #交叉规则：交换染色体上x个基因
#     count = 0
#     for i in np.arange(CROSSBEGIN,CROSSEND):
#         if(CROSSEND!=CROSSBEGIN):
#             count+=1
#             tmp1 = vm[i]
#             tmp2 = vf[i]
#             #vf内部交换
#             idf=np.argwhere(vf==tmp1)[0]
#             tmp=vf[i]
#             vf[i]=vf[idf]
#             vf[idf]=tmp
#             #vm内部交换
#             idm = np.argwhere(vm == tmp2)[0]
#             tmp = vm[i]
#             vm[i] = vm[idm]
#             vm[idm] = tmp
#     vf[CITYNUM]=vf[0]
#     vm[CITYNUM]=vm[0]
#     print('count:\n',count)
#
# def genCross(v): #交叉操作
#     lenv=np.shape(v)[0]
#     print('lev:\n',lenv)
#     for i in np.arange(lenv-1):
#         crossMethod(v[i],v[i+1])
#
# for i in np.arange(GEN):
#     a=np.random.choice(CITYNUM,size=(1,CITYNUM),replace=False)
#     np.random.shuffle(a)
#     a=np.append(a,a[0,0])
#     gen_solution[i]=a
# print(gen_solution,'\n')
# for j in np.arange(GEN):
#     genCross(gen_solution)
# #print('after cross gen_sov:\n',gen_solution)

