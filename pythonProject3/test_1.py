def print1(*v):
    for i in v:
        print('i is:',i)

def print2(**v):
    for key1,val1 in v:
        print('key is: ',key1,'----','val is: ',val1,'\n')

v1=[1,'hahaha',3.1415,'yy']
v2=(3,'*#','sht',9.201,32,'dd')
v3={'a':1,'b':'bb','cvvvv':3333}
v4={'ftd':2322,'ggg':'uu999'}
print1(v1,v2)
print1(v2,v1)
print2(v3,v4)
print2(v4,v3)