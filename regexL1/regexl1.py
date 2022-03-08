import re


# 2022-2-17，rregular expression learning from 麦叔 at bilibili

# 这里‘\’表示该行的字符串未结束，和下行连接。也可以用双引号或三引号
text1='姓名: 张三, 性别: male, 年龄: 28, 手机: 13812344567, e-mail: zhangsan@git.com'\
      '姓名: li4, 性别: male, 年龄: 21, 电话: 0521-63544567, e-mail: li4@outlook.com'\
      '姓名: wang五, 性别: female, 年龄: 36, 手机: 15903632558, e-mail: wang5@gmail.com'\
      '姓名: 陈6, 性别: female, 年龄: 50, 电话: 0312-6342378, e-mail: chenliu@xxxx.com'\
      '姓名: tian7, 性别: male, 年龄: 82, 电话: 023-326342378, e-mail: @git.com'

"1. 找出指定数字。 方法1：使用'in'"
dest='344568'
if dest in text1:
    print("is in")
else:
    print("is not in")
    "方法2: 正则表达式re包中的findall()"
print(re.finditer(r'344',text1))  # 顾名思义，返回的是找到第一个位置的迭代器
print(re.findall(r'344',text1))  # 顾名思义，返回的是所有符合条件组成的列表
print(re.findall(r'\w',text1))  # 通配符等格式控制符都行，返回字符(串)组成的列表
print(re.findall(r'\d+[1,4]',text1))  # 重复多位数字，而且只要出现1或,或4就符合条件，返回符合条件的字符串组成的列表
print(re.findall(r'\d{1,4}',text1))  # 只要是长度在1到4之间的数字串就符合条件，返回符合条件的字符串组成的列表(注意这个表示数字序列的长度，不是1到4之间的数字)

'''常用通配符:\d,\D(非数字),\w(任意字母数字下划线中文),\W,+(表示它前面的可以重复多次),*(前面包含0个或多个),？，[](至少符合里面的一个条件)
    \s(空格),\S，{}(包含一个区间长度),{}内用','表示区间范围，[]内用'-'表示区间范围。如{1,4},[f-p],
'''


text2='姓名: 张三, 性别: male, 年龄: 28, 手机: 13812344567, e-mail: zhangsan@git.com'\
      '姓名: li4, 性别: male, 年龄: 21, 电话: 0521-6354547, e-mail: li4@outlook.com'\
      '姓名: wang五, 性别: female, 年龄: 36, 手机: 015903632558, e-mail: wang5@gmail.com'\
      '姓名: 陈6, 性别: female, 年龄: 50, 电话: 16363423728, e-mail: chenliu@xxxx.com'\
      '姓名: tian7, 性别: male, 年龄: 82, 电话: 023-326342378, e-mail: @git.com'\
      '姓名: wang8, 性别: male, 年龄: 0232-, 电话: 4323-98742378, e-mail: ut@.com'
print(re.findall(r'0\d{3,4}-\d{7,8}',text2)) # 一般座机号码是0开头，区号3或4位，尾号7或8位
print(re.findall(r'1[3,5,7,8]\d{9}|0\d{3,4}-\d{7,8}',text2))  #找出手机号或座机号



