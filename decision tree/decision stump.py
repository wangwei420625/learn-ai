import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def change2dict(some_set):
    result = {}
    for i in some_set:
        if i not in result:
            # 上面的代码时间复杂度是o(1),字典是以hashtable的方式在内存中存储的
            result[i]=1
        else:
            result[i]+=1
    return result
def entropy(some_set):
    count_dict = change2dict(some_set)
    set_len =len(some_set)
    result = 0
    for i in count_dict:
        ent = count_dict[i]/set_len*(-np.log2(count_dict[i]/set_len))
        result += ent
    return result
# 上面的函数可以计算某一个集合自己的熵
def info_gain(org_set,sp1_set,sp2_set):
    set_rate1 = len(sp1_set)/len(org_set)
    set_rate2 = len(sp2_set)/len(org_set)
    return entropy(org_set)-set_rate1*entropy(sp1_set)-set_rate2*entropy(sp2_set)
#根据三个集合的熵来计算这次分裂带来的信息增益值



iris = load_iris()
print(iris)
#读取鸢尾花数据集
# print(iris['data'])
# print(iris['target'])
data = iris['data'][:,2]
data_set_org = np.array(list(zip(data,iris['target'])))
data_sorted = data_set_org[np.lexsort(data_set_org[:,::-1].T)]
#把数组的第一列作为排序的key，不打乱行内数据的关系。重新对数组中的行进行排序
pd_data = pd.DataFrame(data_sorted,columns=list('ab'))
# print(pd_data)
# print(data_set_org)
split_set = set(data_set_org[:,0])
# sp1 = pd_data[pd_data.a<2.8]
# sp2 = pd_data[pd_data.a>=2.8]
# print(info_gain(pd_data.b,sp1.b,sp2.b))
maxgain=0
for split_point in split_set:
    sp1 = pd_data[pd_data.a < split_point]
    sp2 = pd_data[pd_data.a >= split_point]
    gain = info_gain(pd_data.b,sp1.b,sp2.b)
    print(gain)
    if gain > maxgain:
        maxgain = gain
        best_split_point = split_point
print('best_split_point is',best_split_point,maxgain )
print()
DecisionTreeClassifier