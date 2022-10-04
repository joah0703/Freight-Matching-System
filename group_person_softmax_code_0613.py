# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:47:17 2022

@author: Yurim
"""


import os
import pandas as pd
import numpy as np
import random
from datetime import datetime
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#from keras.callbacks import ModelCheckpoint
#from sklearn.model_selection import train_test_split

os.chdir(r'E:\2022\0_study\화물')
#model_path = './model/{epoch:02d}-{accuracy:.4f}.h55'

#%%

## 차주별 train:test = 8:2
# group_num = group0
# #member_num = group5_list[4]
# member_list = group0_count.index[0:8].values.tolist()
# member_list.extend([3564, 48528])

# for member in member_list:
#     member_num = member
#     ## 해당 차주만 1, 아니면 0
#     true_y = np.where(group_num['intMemberSeq'] == member_num, 1, 0)
#     group_num['true_y'] = true_y
    
    
#     ## y값 원핫인코딩
#     encoder = OneHotEncoder()
#     encoder.fit(group_num['true_y'].values.reshape(-1, 1))
    
#     ##train:test = 8:2
#     y1_test_idx = random.sample(list(group_num[group_num['true_y']==1].index), 5)
#     y0_test_idx = random.sample(list(group_num[group_num['true_y']==0].index), round(len(group_num)*0.2)-5)
    
#     test_idx = y1_test_idx + y0_test_idx
#     all_train_idx = [x for x in list(group_num.index) if x not in test_idx]
    
#     train_data = group_num.loc[all_train_idx]
#     test_data = group_num.loc[test_idx]
    
#     train_data.to_csv('grouop0_'+str(member_num)+'_train.csv')
#     test_data.to_csv('grouop0_'+str(member_num)+'_test.csv')


#%%

raw_data2 = pd.read_csv('./0531/t_2500_done_var_1_kmeans_6.csv')



# group0 - [36011, 2244, 21818, 11802, 54914, 33005, 71764, 40124, 3564, 48528]
member_num = 36011

raw_train2 = pd.read_csv('./0531/grouop0_'+str(member_num)+'_train.csv')
raw_test2 = pd.read_csv('./0531/grouop0_'+str(member_num)+'_test.csv')
del raw_train2['Unnamed: 0']
del raw_test2['Unnamed: 0']
#raw_train.columns
  

cate_col = ['strUpZone', 'strDwZone', 'intCarTypecode', 'intCarUpType', 'intPaytypecode', 'intUpDateType', 'intDwDateType', 'weekUp', 'weekDown']
conti_col = ['intTotalPrice', 'intMemberPrice', 'intArrangePrice', 'intDistance', 'intDefaultPrice', 'timeUp', 'timeDown', 'FeeKm'] 


## 범주형 one hot encoding
raw_data = pd.get_dummies(raw_data2, columns=cate_col)
all_cate_cols = list(raw_data.loc[:,'strUpZone_11':].columns)

raw_train = pd.get_dummies(raw_train2, columns=cate_col)
raw_test = pd.get_dummies(raw_test2, columns=cate_col)


# 열 개수 맞춰주기
non_train_col = [x for x in all_cate_cols if x not in list(raw_train.loc[:,'strUpZone_11':].columns)]
non_test_col = [x for x in all_cate_cols if x not in list(raw_test.loc[:,'strUpZone_11':].columns)]

if len(non_train_col)!=0:
    raw_train[non_train_col] = 0
if len(non_test_col)!=0:
    raw_test[non_test_col] = 0
#intMemberSeq, intFeeSend, true_y, cluster_id 빼고 총 변수 79개


# 학습 train, test 만들기
y1_test_idx = list(raw_test[raw_test['true_y']==1].index)
y0_test_idx = random.sample(list(raw_test[raw_test['true_y']==0].index), 95)

train_X = raw_train
test_X = raw_test.loc[y1_test_idx+y0_test_idx]


## 연속형 Scaling
scaler = MinMaxScaler()
scaler.fit(train_X[conti_col])

conti_test_input = scaler.transform(test_X[conti_col])
test_output = test_X['true_y']

cate_test_input = np.array(test_X.loc[:,'strUpZone_11':])
test_input = np.hstack((conti_test_input, cate_test_input))


#%%
epoch_num = 1000000

## Step4. 그룹별 개인 모델 학습

num_features = len(test_input[0])
# =============================================================================
# softmax model (categorical_crossentropy)
# =============================================================================
sf_model=Sequential()
embedding_layer=InputLayer(input_shape=(num_features,))
sf_model.add(embedding_layer)
sf_model.add(Flatten())
sf_model.add(Dense(10))
sf_model.add(Dense(5))
sf_model.add(Dense(2, activation='softmax')) #
adam = optimizers.Adam(lr=0.1)
sf_model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])
sf_model.summary()


## y값 원핫인코딩
encoder = OneHotEncoder()
encoder.fit(raw_train['true_y'].values.reshape(-1, 1))


start_time = datetime.now()
# all_eval = []
# for i in range(50):
# iteration마다 sampling
y1_train_idx = list(raw_train[raw_train['true_y']==1].index)
y0_train_idx = random.sample(list(raw_train[raw_train['true_y']==0].index), len(y1_train_idx))
train_idx = y1_train_idx+y0_train_idx

conti_train_input = scaler.transform(train_X.loc[train_idx,conti_col])
cate_train_input = np.array(train_X.loc[train_idx,'strUpZone_11':])
train_input = np.hstack((conti_train_input, cate_train_input))
train_output2 = train_X.loc[train_idx, 'true_y']
train_output = encoder.transform(train_output2.values.reshape(-1, 1)).toarray()

#cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='accuracy', verbose=1, save_best_only=True)
history = sf_model.fit(train_input, train_output, epochs=epoch_num, batch_size=64) #, callbacks=[cb_checkpoint]
# eval_value = sf_model.evaluate(train_input, train_output)
# all_eval.append(eval_value[1])
end_time = datetime.now()
print('===== Runtime :',end_time-start_time,'=====')
sf_model.save('./0613/2500t_group0_'+str(member_num)+'_epoch'+str(epoch_num)+'_lr0.1.h5')
#print('train_acc :', sum(all_eval)/len(all_eval))


#%%
# =============================================================================
# loss plot
# =============================================================================
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))

plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

    

# =============================================================================
# train predict confusion matrix
# =============================================================================
tr_pred = sf_model.predict(train_input)
tr_pred_label=[]
tr_pred_proba=[]
for i in range(len(tr_pred)):
    tr_pred_label.append(tr_pred[i].argmax())
    tr_pred_proba.append(tr_pred[i].max())
    
print(confusion_matrix(train_output2, tr_pred_label))


#%%
# 100번 평균 예측값
y1_test_idx = list(raw_test[raw_test['true_y']==1].index)
iter100_result = pd.DataFrame([])

for i in range(100):
    iter_result = []
    
    # test 만들기
    y0_test_idx = random.sample(list(raw_test[raw_test['true_y']==0].index), 95)
    test_X = raw_test.loc[y1_test_idx+y0_test_idx]

    ## Scaling
    conti_test_input = scaler.transform(test_X.loc[:,conti_col])
    cate_test_input = np.array(test_X.loc[:,'strUpZone_11':])
    test_input = np.hstack((conti_test_input, cate_test_input))
    test_output = test_X['true_y']


    #예측값
    ts_pred = sf_model.predict(test_input) 
    ## 정확도 추출
    ts_pred_label=[]
    ts_pred_proba=[]
    for i in range(len(ts_pred)):
        ts_pred_label.append(ts_pred[i].argmax())
        ts_pred_proba.append(ts_pred[i].max())
    
    iter_result.append(accuracy_score(test_output, ts_pred_label))
    iter_result.append(Counter(ts_pred_label)[1])
    iter_result.append(Counter(ts_pred_label)[0])
    
    
    correct_y1 = 0
    correct_y0 = 0
    for i in range(len(test_output)):
        if test_output.iloc[i]==1:
            if test_output.iloc[i]==ts_pred_label[i]:
                correct_y1 += 1
        else:
            if test_output.iloc[i]==ts_pred_label[i]:
                correct_y0 += 1
    
    iter_result.append(correct_y1)
    iter_result.append(correct_y0)
    iter100_result = iter100_result.append(pd.DataFrame([iter_result]))


print(iter100_result.mean())
print(confusion_matrix(test_output, ts_pred_label))




#%%
## 추천 리스트 Ver1
raw_test_X = raw_test2.loc[y1_test_idx+y0_test_idx]
raw_test_X['pred_label'] = ts_pred_label
raw_test_X['pred_proba'] = ts_pred_proba

pred_label_1 = raw_test_X[raw_test_X['pred_label']==1]
top_list = pred_label_1.sort_values(by='pred_proba', ascending=False)
top_list.to_csv('./0613/2500t_group0_'+str(member_num)+'_epoch'+str(epoch_num)+'_lr0.1_top_list.csv')


# =============================================================================
# 이하 생략
# =============================================================================




#%%
## 10% acc
top10_list = top_list[0:int(len(test_X)*0.1)]

list_num = len(top10_list[top10_list['intMemberSeq']==member_num])
per_test_num = len(test_X[test_X['intMemberSeq']==member_num])
print('top10% acc :', list_num/per_test_num)
print('top10% count :', Counter(top10_list['true_y']))


## 25% acc
top25_list = top_list[0:int(len(test_X)*0.25)]

list_num = len(top25_list[top25_list['intMemberSeq']==member_num])
per_test_num = len(test_X[test_X['intMemberSeq']==member_num])
print('top25% acc :', list_num/per_test_num)
print('top25% count :', Counter(top25_list['true_y']))


## 50% acc
top50_list = top_list[0:int(len(test_X)*0.5)]

list_num = len(top50_list[top50_list['intMemberSeq']==member_num])
per_test_num = len(test_X[test_X['intMemberSeq']==member_num])
print('top50% acc :', list_num/per_test_num)
print('top50% count :', Counter(top50_list['true_y']))







