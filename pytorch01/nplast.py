import numpy as np

# #np.random.seed(4) #加上之后，每次运行都会生成一样的随机数
#
# #正常生成不同的随机数
# for i in range(4):
#     t=np.random.randint(0,12,(2,3))
#     print(t)
#
#
# np.random.seed(4) #加上之后，每次运行都会生成一样的随机数
# #但在for中生成不同的随机数，和C/C++中正好反过来。C/C++不加seed，在for中生成相同的随机数，加了seed，每次重新运行会产生不同随机数，不加seed，每次重新运行都会产生一样的随机数
# #np正好相反，加了seed每次重新运行会生成相同随机数，不加seed每次重新运行会生成不同随机数，加上seed，for内仍然是不同随机数
# #综上两个。seed对for无影响，因为seed种子的改变相对于for中每层循环的运行时间太长了，无法改变它。因此for内的随机，取决于语言的整体设计，整体是不修改，则for不修改，整体会修改则for会修改。
# #而seed只能改变程序重新执行的每次随机数。它的time相对于程序执行时间，也是太长，来不及改种子，程序便已执行完，除非程序中加延时
#
# #正常生成不同的随机数
# for i in range(4):
#     t=np.random.randint(0,12,(2,3))
# print(t)



