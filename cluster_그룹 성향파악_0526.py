# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:54:58 2022

@author: Yurim
"""
# =============================================================================
# 수빈수정. 0531
# 그룹별 플랏.
# =============================================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir(r'E:\2022\0_study\화물\0614 dec')

raw_data = pd.read_csv('2022-6-14 3s Cargo_y_train_pred_kmeans data.csv', encoding='cp949')
raw_data.columns
# colname = ['intMemberSeq','strUpZone', 'strDwZone', 'intCarTypecode', 'intCarUpType', 'intPaytypecode', 
#            'intTotalPrice', 'intMemberPrice', 'intArrangePrice', 'intDistance', 
#            'intDefaultPrice', 'intUpDateType', 'intDwDateType', 'intFeeSend',
#            'weekUp', 'weekDown', 'timeUp', 'timeDown', 'FeeKm']
colname = raw_data.columns
data = raw_data[colname]
data['FeeKm'] = round(data['FeeKm'])

# k = len(raw_data['cluster_id'].value_counts())
k = len(raw_data['y_pred'].value_counts())

for i in range(k):
    globals()['group{}'.format(i)] = data[raw_data['cluster_id']==i]
#%%
#group0 - [36011, 2244, 21818, 11802, 54914, 33005, 71764, 40124, 3564, 48528]
view_group_num = 5

view_group = globals()['group'+str(view_group_num)]

#%%


#%%

## 플랏 크기 설정 plt.rcParams
aa=25
bb=10
#%%
## 플랏 라벨 설정
fee=range(10)
tp=range(10)
mp=range(1,6)
dis=range(3)
#%% FeeKm plot

plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=0.5)
plt.rcParams['figure.figsize'] = (aa,bb)


col='FeeKm'
va=view_group[col].tolist()
va.sort()
# print(set(va))
max(va)
min(va)

dele=[]
for i in range(len(va)):
    if va[i] <= 10000:
        dele.append(va[i])
# print(dele)

plt.rcParams['figure.figsize'] = (aa,bb)
plt.subplot(3,4,1)
hist=plt.hist(dele, label=col)

x = np.append(hist[0], hist[0][-1])

v=hist[1]
for i in fee:#range(fee):
    plt.text(v[i], x[i], int(v[i]))
plt.title('Group '+str(view_group_num))

plt.legend()
plt.grid()


# col='FeeKm'
# va=view_group[col].tolist()
# va.sort()
# max(va)
# min(va)
# dele=[]
# for i in range(len(va)):
#     if va[i] >= 10000:
#         if va[i] not in dele:
#             dele.append(va[i])
# # print(dele)
# for i in range(len(dele)):
#     va.remove(dele[i])

# plt.subplot(3,4,1)
# x = np.arange(0,10000,1000)
# plt.hist(va, label=col,bins=1000)
# plt.xlim([0,10000])
# plt.legend()
# plt.grid()
# plt.xticks(x)



#%% Total Price
col='intTotalPrice'
va=view_group[col].tolist()
va.sort()
# print(set(va))
max(va)
min(va)

dele=[]
for i in range(len(va)):
    if va[i] <= 1000000:
        dele.append(va[i])
# print(dele)

plt.rcParams['figure.figsize'] = (aa,bb)
plt.subplot(3,4,2)
hist=plt.hist(dele, label=col)

x = np.append(hist[0], hist[0][-1])

v=hist[1]
for i in tp:
    plt.text(v[i], x[i], int(v[i]))
plt.title('Group '+str(view_group_num))
plt.legend()
plt.grid()





#%% intMemberPrice
col='intMemberPrice'
va=view_group[col].tolist()
va.sort()
# print(set(va))
max(va)
min(va)

dele=[]
for i in range(len(va)):
    if va[i] <= 1000000:
        dele.append(va[i])
# print(dele)

plt.rcParams['figure.figsize'] = (aa,bb)
plt.subplot(3,4,3)
hist=plt.hist(dele, label=col)

x = np.append(hist[0], hist[0][-1])

v=hist[1]
for i in mp:
    plt.text(v[i], x[i], int(v[i]))
plt.title('Group '+str(view_group_num))
plt.legend()
plt.grid()


#%% intDistance
col='intDistance'
plt.rcParams['figure.figsize'] = (aa,bb)
plt.subplot(3,4,4)
hist = plt.hist(view_group[col], label=col)
x = np.append(hist[0], hist[0][-1]) #라벨 설정 하려고
v=hist[1]
for i in dis:
    plt.text(v[i], x[i], int(v[i]))
plt.title('Group '+str(view_group_num))

plt.legend()
plt.grid()

# plt.savefig(str(view_group_num)+' 1savefig '+col+'.png')
# plt.show()

#%%
#%%
# view_group_num = 5

# view_group = globals()['group'+str(view_group_num)]

## 범주형
#cate_col = [x for x in colname if x not in conti_col]
cate_col = ['strUpZone', 'strDwZone', 'intUpDateType', 'intDwDateType', 'weekUp', 'weekDown', 'timeUp', 'timeDown']

# plt.subplots_adjust(left=0.125, bottom=0.1,  right=0.9, top=0.9, wspace=0.2, hspace=0.4)

for col in cate_col:
    num=cate_col.index(col)+4
    plt.rcParams['figure.figsize'] = (aa,bb)
    x = view_group[col].value_counts()
    plt.subplot(3,4,num+1)
    
    hist = plt.bar(x.index, np.array(x), label=col)
    for i, v in enumerate(x.index):
        plt.text(v, np.array(x)[i], v)
    plt.title('Group '+str(view_group_num))
    plt.legend()
    plt.grid()

plt.savefig(str(view_group_num)+' 2savefig '+col+'.png')
plt.show()
