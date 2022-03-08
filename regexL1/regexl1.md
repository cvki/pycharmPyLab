import re

# 2022-2-17，rregular expression learning from 麦叔 at bilibili

## 这里‘\’表示该行的字符串未结束，和下行连接。也可以用双引号或三引号
text1='姓名: 张三, 性别: male, 年龄: 28, 手机: 13812344567, e-mail: zhangsan@git.com'\
      '姓名: li4, 性别: male, 年龄: 21, 电话: 0521-63544567, e-mail: li4@outlook.com'\
      '姓名: wang五, 性别: female, 年龄: 36, 手机: 15903632558, e-mail: wang5@gmail.com'\
      '姓名: 陈6, 性别: female, 年龄: 50, 电话: 0312-6342378, e-mail: chenliu@xxxx.com'\
      '姓名: tian7, 性别: male, 年龄: 82, 电话: 023-326342378, e-mail: @git.com'

### "1. 找出指定数字。 方法1：使用'in'"
dest='344568'
if dest in text1:
    print("is in")
else:
    print("is not in")
    "方法2: 正则表达式re包中的findall()"
print(re.finditer(r'344',text1))  # 顾名思义，返回的是找到第一个位置的迭代器
print(re.findall(r'344',text1))  # 顾名思义，返回的是所有符合条件组成的列表
print(re.findall(r'\D',text1))  # 通配符等格式控制符都行，返回字符(串)组成的列表
print(re.findall(r'\d+[1,4]',text1))  # 重复多位数字，而且只要出现1或,或4就符合条件，返回符合条件的字符串组成的列表
print(re.findall(r'\d{1,4}',text1))  # 只要是长度在1到4之间的数字串就符合条件，返回符合条件的字符串组成的列表(注意这个表示数字序列的长度，不是1到4之间的数字)

'''常用通配符:\d,\D(非数字),\w(不含标点的字符),\W,+(表示它前面的可以重复多次),*(前面包含0个或多个),[](至少符合里面的一个条件)
    \s(空格),\S，{}(包含一个区间长度), 
'''

