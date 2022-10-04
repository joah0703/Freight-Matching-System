# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 13:38:34 2022

@author: Yurim

tensorboard 사용방법
https://yeongjin13.tistory.com/17
"""

import os
os.chdir(r'E:\2021\study\5_화물배차\wide deep code\유림수정')

import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import urllib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train_conti  1230.csv',encoding='CP949')
test_data = pd.read_csv('TEST CONTI DATA 1230.csv',encoding='CP949')



""" 전처리 """
del train_data['Unnamed: 0']
del train_data['index']
del test_data['Unnamed: 0']
del test_data['index']

data = train_data.append(test_data)

COLUMNS = data.columns
CONTINUOUS_COLUMNS=['intMemberPrice']
CATEGORICAL_COLUMNS=['strDwZone', 'dateUpint', 'dateDownint', 'addresscode', 'strBirthcode']

le = LabelEncoder()
labeltrs = le.fit_transform(data['strTrsNum'])

train_data['label'] = labeltrs[:len(train_data['strTrsNum'])]
train_data.drop('strTrsNum', axis = 1, inplace=True)

test_data['label'] = labeltrs[len(train_data['label']):]
test_data.drop('strTrsNum', axis = 1, inplace=True)



""" Wide model feature (categorical) """
print(CATEGORICAL_COLUMNS)

for f in CATEGORICAL_COLUMNS:
    train_data[f] = train_data[f].astype(str)
    test_data[f] = test_data[f].astype(str)
    data[f] = data[f].astype(str)
    

strDwZone = tf.contrib.layers.sparse_column_with_keys(column_name="strDwZone", keys=np.unique(data['strDwZone']))
dateUpint = tf.contrib.layers.sparse_column_with_keys(column_name="dateUpint", keys=np.unique(data['dateUpint']))
dateDownint = tf.contrib.layers.sparse_column_with_keys(column_name="dateDownint", keys=np.unique(data['dateDownint']))
addresscode = tf.contrib.layers.sparse_column_with_keys(column_name="addresscode", keys=np.unique(data['addresscode']))
strBirthcode = tf.contrib.layers.sparse_column_with_keys(column_name="strBirthcode", keys=np.unique(data['strBirthcode']))



""" Deep model feature (continuous) """

print(CONTINUOUS_COLUMNS)

intMemberPrice = tf.contrib.layers.real_valued_column("intMemberPrice")



""" Wide and Deep model """

wide_columns = [strDwZone, dateUpint, dateDownint, addresscode, strBirthcode]

deep_columns = [tf.contrib.layers.embedding_column(strDwZone, dimension=8),
                tf.contrib.layers.embedding_column(dateUpint, dimension=8),
                tf.contrib.layers.embedding_column(dateDownint, dimension=8),
                tf.contrib.layers.embedding_column(addresscode, dimension=8),
                tf.contrib.layers.embedding_column(strBirthcode, dimension=8),
                intMemberPrice]

# Define the column names for the data sets.
LABEL_COLUMN = 'label'

df_train = train_data
df_test = test_data


def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  #label = tf.constant(keras.utils.to_categorical(df[LABEL_COLUMN]))
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

#df_train[CATEGORICAL_COLUMNS].dtypes



""" Main """

##파일명 반드시 변경!!
model_filename = "220103_WD model" 

m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir = "C:/Users/Yurim/Desktop/WD model/" + model_filename,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50],
    n_classes = len(np.unique(data['strTrsNum'])))

start_time = datetime.datetime.now()

hist = m.fit(input_fn=train_input_fn, steps=10)
results = m.predict_proba(input_fn=eval_input_fn)

finish_time = datetime.datetime.now()


### top 10 acc
count=0
for prednum in range(len(df_test['label'])):
    pred = next(results)
    prediction10 = sorted(range(len(pred)),key= lambda i: pred[i])[-10:]

    if df_test['label'][prednum] in prediction10:
        count+=1

print("Top 10 Accuracy : ", count/len(df_test['label']))
print("Run time : ", finish_time - start_time)



# 모델 불러와서 predict
# =============================================================================
# modeling = tf.contrib.learn.DNNLinearCombinedClassifier(
#     model_dir = r"E:\2021\study\5_화물배차\wide deep code\유림수정\220103_WD model",
#     linear_feature_columns=wide_columns,
#     dnn_feature_columns=deep_columns,
#     dnn_hidden_units=[100, 50],
#     n_classes = len(np.unique(data['strTrsNum'])))
# 
# start_time = datetime.datetime.now()
# #hist = testm.fit(input_fn=train_input_fn, steps=10)
# results = modeling.predict_proba(input_fn=eval_input_fn)
# 
# finish_time = datetime.datetime.now()
# 
# 
# ### top 10 acc
# count=0
# for prednum in range(len(df_test['label'])):
#     pred = next(results)
#     prediction10 = sorted(range(len(pred)),key= lambda i: pred[i])[-10:]
# 
#     if df_test['label'][prednum] in prediction10:
#         count+=1
# 
# print("Top 10 Accuracy : ", count/len(df_test['label']))
# print("Run time : ", finish_time - start_time)
# =============================================================================
