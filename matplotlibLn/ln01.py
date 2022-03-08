import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import font_manager

#自定义字体格式, 需要提前下载汉语字体然后使用文件路径配置，需要用到上面注释的导入文件
#my_font=font_manager.FontProperties(fname=r'C:\Users\buaa203\anaconda3\envs\env_pytorch\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\cmtt10.ttf')//该ttf为英文字体，不支持中文

#或者
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 一天24h气温变化画图
x=np.arange(24)
y1=[-12.3,-14.2,-13.3,-11.2,-14.6,-15.8,-10.2,-8.2,-2.1,3.5,6.7,7.8,9.2,10.4,11.6,10.3,9.3,7.2,3.1,-1.5,-3.6,-5.7,-8.2,-10.4]
y2=[3.7,1.2,0.3,-0.8,-1.6,0.9,3.2,8.2,10.1,15.5,21.7,25.25,26.2,30.4,27.6,20.3,11.3,8.2,7.1,4.5,3.6,2.7,1.2,0.4]
# print(len(y1))
# print(len(y2))

#背景大小
plt.figure(figsize=(20,8),dpi=80)

#设置横纵坐标的标签
xtk=['{}时'.format(i) for i in x]
plt.xticks(x,xtk)
# plt.xticks(x,xtk,fontproperties=my_font)
#ytk=['{}度'.format(i) for i in y]
#plt.yticks(y,ytk) #不要设置yticks，否则y轴数据不等比例

#title
plt.title("某两地区一天24时温度变化图",fontsize=18)
plt.xlabel("时间:h",fontsize=14)
plt.ylabel("温度:摄氏度",fontsize=14,rotation=90)

#网格
plt.grid(alpha=0.4) #设置网格透明度

# 折线图
plt.plot(x,y1,label='地区1')
plt.plot(x,y2,label='地区2')

#图例,依赖于plot函数中的label参数,因此必须放在label后面
plt.legend()

#这些也可以添加label

#散点图
plt.scatter(x,y1)
plt.scatter(x,y2)

#条形图
plt.bar(x,y1)
#plt.bar(x,y2)
plt.barh(x,y2) #将x和y互换，这是画横着的条形图

#显示
plt.show()

