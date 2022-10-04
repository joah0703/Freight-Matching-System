"""
CV : https://jonsyou.tistory.com/23
"""
###############################################################################
##data read
###############################################################################
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
#from sklearn.metrics import classification_report,confusion_matrix

#dataset = pd.read_csv('F:/2020/study/축산도체/등급 예측 구현(일반)/train.csv')
#testset = pd.read_csv('F:/2020/study/축산도체/등급 예측 구현(일반)/test.csv')
os.chdir(r'F:\2021\study\5_화물배차\wide deep code\유림수정')

raw_data=pd.read_csv('cargolist.cargo_full.csv', encoding='cp949')


"""데이터 전처리"""
data_group = raw_data.loc[raw_data['intStatus']==3,]
# data_group = data_group.loc[(data_group['intDistance']<=600)&(data_group['intDistance']>=0),]
# data_group = data_group.loc[-((data_group['intDistance']==0)&(data_group['dwUpPoiX']!=data_group['dwDwPoiX'])),]
data_group = data_group.loc[data_group['intMemberPricecode']!=0,]
data_group = data_group.loc[data_group['dateUpcode']!=0,]
data_group = data_group.loc[data_group['dateDowncode']!=0,]
data_group = data_group.loc[data_group['strUpZonecode']!=0,]
data_group = data_group.loc[data_group['strDwZonecode']!=0,]
data_group = data_group.loc[data_group['intCarTypecode2']!=0,]
data_group = data_group.loc[data_group['intCarSizecode2']!=0,]
data_group = data_group.loc[data_group['intDistancecode']!=0,]


data_group['Group'] = 0
group=1
for type in range(1,5):
    for size in range(1,4):
        data_group.loc[(data_group['intCarTypecode2']==type)&(data_group['intCarSizecode2']==size),'Group'] = group
        group+=1


############ data preprocessing
CONTINUOUS_COLUMNS=['intDistance', 'intMemberPrice','dateUpint', 'dateDownint']
CATEGORICAL_COLUMNS=['strUpZone', 'strDwZone']



"""그룹별 for문"""
group_num=12
data_group_1 = data_group.loc[data_group['Group']==group_num,].reset_index(drop=True)
member100=data_group_1['strTrsNum'].value_counts()
data = data_group_1.loc[data_group_1['strTrsNum'].isin(member100[member100>=3].index)].reset_index(drop=True)
    

for c in CATEGORICAL_COLUMNS:
    le = LabelEncoder()
    data[c] = le.fit_transform(data[c])
"""
#표준화
scaler = StandardScaler()
data[CONTINUOUS_COLUMNS] = scaler.fit_transform(data[CONTINUOUS_COLUMNS])
"""
#정규화
from sklearn.preprocessing import MinMaxScaler

# Normalization 최소값 0 / 최대값 1
scaler = MinMaxScaler()
data[CONTINUOUS_COLUMNS] = scaler.fit_transform(data[CONTINUOUS_COLUMNS])


data.head()

"""## 라벨 값 생성"""
le = LabelEncoder()
result = le.fit_transform(data['strTrsNum'])

y = result
x = data[CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS]


X_train2, X_test, y_train2, y_test = train_test_split(x,y, test_size = 0.2, stratify=y, random_state = 1)

X_test=X_test.reset_index(drop=True)
X_train2=X_train2.reset_index(drop=True)


#y_train=keras.utils.to_categorical(y_train)
#y_test=keras.utils.to_categorical(y_test)
#y_train2=np.array(y_train2)
#y_test2=np.array(y_test2)

print(len(X_train2), len(y_train2), len(X_test), len(y_test))
len(np.unique(y))

###############################################################################
##logistic regression
###############################################################################
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(penalty='l2', solver='lbfgs', random_state=2, multi_class='multinomial')
logmodel = LogisticRegression(penalty='none',solver='newton-cg', random_state=2, multi_class='multinomial')

## best hyperparameter search
params = {'C':[0.01, 0.1, 1, 5, 10]}
params = {'solver':['newton-cg','lbfgs','sag','saga']}

grid_model = GridSearchCV(logmodel, param_grid=params, scoring='accuracy', cv=10)
grid_model.fit(X_train2, y_train2)
print('최적 하이퍼 파라미터 : {}, 최적 평균 정확도 : {}'.format(grid_model.best_params_, grid_model.best_score_))
# penalty='l2' #최적 하이퍼 파라미터 : {'C': 1} #최적 하이퍼 파라미터 : {'solver': 'newton-cg'}


ss = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_accuracy = []
n_iter =0
logmodel_idx=[]
for tr_idx, val_idx in ss.split(X_train2, y_train2):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할
    X_train, X_val = X_train2.loc[tr_idx,], X_train2.loc[val_idx,]
    y_train, y_val = y_train2[tr_idx], y_train2[val_idx]
    
    logmodel.fit(X_train, y_train)
    pred = logmodel.predict_proba(X_val)
    
    count=0
    for prednum in range(len(pred)):
        prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
        if y_val[prednum] in prediction10:
            count+=1
    accuracy = count/len(y_val)
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, train_size, val_size))
    n_iter += 1
    cv_accuracy.append(accuracy)
    logmodel_idx.append(tr_idx)

print("max_val : ",max(cv_accuracy)*100)
train_idx = logmodel_idx[cv_accuracy.index(max(cv_accuracy))]

X_train, y_train = X_train2.loc[train_idx,], y_train2[train_idx]
logmodel = LogisticRegression(penalty='none',solver='newton-cg', random_state=2, multi_class='multinomial')
logmodel.fit(X_train,y_train)
print("train acc : ", accuracy_score(logmodel.predict(X_train), y_train)*100)
#0.6309263311451495

pred = logmodel.predict_proba(X_test)
# 상위 10에 들면 True
count=0
for prednum in range(len(pred)):
    prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
    if y_test[prednum] in prediction10:
        count+=1
print("Accuracy : ", count/len(y_test)*100)
#0.6180758017492711





# =============================================================================
# logmodel.score(X_test,y_test)
# ############################
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0, multi_class = 'auto')
# classifier.fit(X_train, y_train)
# classifier.score(X_train, y_train)
# #predicting the values
# y_pred = classifier.predict(X_test)
# 
# #confusion matrix
# from sklearn.metrics import confusion_matrix
# cm_Log = confusion_matrix(y_test, y_pred)
# cm_Log
# 
# #Accuracy and report of the classifier
# from sklearn.metrics import accuracy_score, classification_report
# accuracy_score(y_test,y_pred)
# #0.6151603498542274
# 
# 
# #report
# report_Log = classification_report(y_test, y_pred)
# print(report_Log)
# 
# =============================================================================
###############################################################################
##decision tree
###############################################################################
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=10, max_features='auto', splitter='random')

## best hyperparameter search
# params = {'criterion':['gini','entropy'],
#           'splitter':['best','random'],
#           'max_features':['auto','sqrt','log2'],
#           'max_depth':[6,8,10,12,16,20,24]}
# grid_model = GridSearchCV(dtree, param_grid=params, scoring='accuracy', cv=10)
# grid_model.fit(X_train2, y_train2)
# print('최적 하이퍼 파라미터 : {}, 최적 평균 정확도 : {}'.format(grid_model.best_params_, grid_model.best_score_))
#최적 하이퍼 파라미터 : {'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto', 'splitter': 'random'}, 최적 평균 정확도 : 0.17429768812747534

ss = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
cv_accuracy = []
n_iter =0
dtree_idx=[]
for tr_idx, val_idx in ss.split(X_train2, y_train2):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할
    X_train, X_val = X_train2.loc[tr_idx,], X_train2.loc[val_idx,]
    y_train, y_val = y_train2[tr_idx], y_train2[val_idx]
    
    dtree.fit(X_train, y_train)
    pred = dtree.predict_proba(X_val)
    
    count=0
    for prednum in range(len(pred)):
        prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
        if y_val[prednum] in prediction10:
            count+=1
    accuracy = count/len(y_val)
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, train_size, val_size))
    n_iter += 1
    cv_accuracy.append(accuracy)
    dtree_idx.append(tr_idx)

print("max_val : ",max(cv_accuracy)*100)
train_idx = dtree_idx[cv_accuracy.index(max(cv_accuracy))]

X_train, y_train = X_train2.loc[train_idx,], y_train2[train_idx]

dtree = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=10, max_features='auto', splitter='random')
dtree.fit(X_train,y_train)
print("train acc : ", accuracy_score(dtree.predict(X_train), y_train)*100)
#0.9948942377826404
pred = dtree.predict_proba(X_test)
# 상위 10에 들면 True
count=0
for prednum in range(len(pred)):
    prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
    if y_test[prednum] in prediction10:
        count+=1
print("Accuracy : ", count/len(y_test)*100)
#0.6705539358600583



#dtree.score(X_test,y_test)
#0.6705539358600583

###############################################################################
##Randomforest
###############################################################################
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=2)

## best hyperparameter search
# params = {'criterion':['gini','entropy'],
#           'n_estimators':[100,200,300,400,500,600,700,800,900,1000],
#           'class_weight':['balanced','balanced_subsample','none'],
#           'max_features':['auto','sqrt','log2'],
#           'max_depth':[6,8,10,12,16,20,24]}
# grid_model = GridSearchCV(rfc, param_grid=params, scoring='accuracy', cv=10)
# grid_model.fit(X_train2, y_train2)
# print('최적 하이퍼 파라미터 : {}, 최적 평균 정확도 : {}'.format(grid_model.best_params_, grid_model.best_score_))
#


ss = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_accuracy = []
n_iter =0
rfc_idx=[]
for tr_idx, val_idx in ss.split(X_train2, y_train2):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할
    X_train, X_val = X_train2.loc[tr_idx,], X_train2.loc[val_idx,]
    y_train, y_val = y_train2[tr_idx], y_train2[val_idx]
    
    rfc.fit(X_train, y_train)
    pred = rfc.predict_proba(X_val)
    
    count=0
    for prednum in range(len(pred)):
        prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
        if y_val[prednum] in prediction10:
            count+=1
    accuracy = count/len(y_val)
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, train_size, val_size))
    n_iter += 1
    cv_accuracy.append(accuracy)
    rfc_idx.append(tr_idx)

print("max_val : ",max(cv_accuracy)*100)
train_idx = logmodel_idx[cv_accuracy.index(max(cv_accuracy))]

X_train, y_train = X_train2.loc[train_idx,], y_train2[train_idx]

rfc = RandomForestClassifier(n_estimators=500, random_state=2)
rfc.fit(X_train, y_train)
print("train acc : ", np.round(accuracy_score(rfc.predict(X_train), y_train),5)*100)
#0.9948942377826404
pred = rfc.predict_proba(X_test)
# 상위 10에 들면 True
count=0
for prednum in range(len(pred)):
    prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
    if y_test[prednum] in prediction10:
        count+=1
print("Accuracy : ", count/len(y_test)*100)
#0.6997084548104956




#rfc.score(X_test,y_test)
# 0.6997084548104956

# =============================================================================
# 
# ############################
# #Random Forest Classifier
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
# 
# ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
# cv_accuracy = []
# n_iter =0
# classifier_idx=[]
# for tr_idx, val_idx in ss.split(X_train2):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할
#     X_train, X_val = X_train2.loc[tr_idx,], X_train2.loc[val_idx,]
#     y_train, y_val = y_train2[tr_idx], y_train2[val_idx]
#     
#     classifier.fit(X_train, y_train)
#     pred = classifier.predict(X_val)
#     
#     accuracy = np.round(accuracy_score(y_val, pred), 5) # 소수점 4자리 반올림
#     train_size = X_train.shape[0]
#     val_size = X_val.shape[0]
#     
#     print('\n#{0} 교차 검증 정확도 : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
#           .format(n_iter, accuracy, train_size, val_size))
#     print('#{0} 검증 세트 인덱스 : {1}'.format(n_iter,val_idx))
#     n_iter += 1
#     cv_accuracy.append(accuracy)
#     classifier_idx.append(tr_idx)
# 
# print(cv_accuracy)
# train_idx = classifier_idx[0]
# 
# X_train, y_train = X_train2.loc[train_idx,], y_train2[train_idx]
# 
# classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train,y_train)
# print(np.round(accuracy_score(classifier.predict(X_train), y_train),5))
# 
# #predicting the values
# y_pred = classifier.predict(X_test)
# print(np.round(accuracy_score(y_pred, y_test),5))
# 
# 
# 
# #confusion matrix
# cm_RF = confusion_matrix(y_test, y_pred)
# print(cm_RF)
# 
# #Accuracy
# accuracy_score(y_test,y_pred)
# #0.8983333333333333
# 
# #Report
# report_RF = classification_report(y_test, y_pred)
# print(report_RF)
# 
# =============================================================================
###############################################################################
##KNN
###############################################################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='kd_tree', leaf_size=10, n_neighbors=15, p=1, weights='uniform')

## best hyperparameter search
# params = {'weights':['uniform','distance'],
#           'n_neighbors':[5,10,15,20,25,30],
#           'algorithm':['ball_tree','kd_tree','auto','brute'],
#           'p':[1,2],
#           'leaf_size':[10,20,30,40,50]}
# grid_model = GridSearchCV(knn, param_grid=params, scoring='accuracy', cv=10)
# grid_model.fit(X_train2, y_train2)
# print('최적 하이퍼 파라미터 : {}, 최적 평균 정확도 : {}'.format(grid_model.best_params_, grid_model.best_score_))
#최적 하이퍼 파라미터 : {'algorithm': 'kd_tree', 'leaf_size': 10, 'n_neighbors': 15, 'p': 1, 'weights': 'uniform'}, 최적 평균 정확도 : 0.22522335820208159


ss = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_accuracy = []
n_iter =0
knn_idx=[]
for tr_idx, val_idx in ss.split(X_train2, y_train2):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할
    X_train, X_val = X_train2.loc[tr_idx,], X_train2.loc[val_idx,]
    y_train, y_val = y_train2[tr_idx], y_train2[val_idx]
    
    knn.fit(X_train, y_train)
    pred = knn.predict_proba(X_val)
    
    count=0
    for prednum in range(len(pred)):
        prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
        if y_val[prednum] in prediction10:
            count+=1
    accuracy = count/len(y_val)
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, train_size, val_size))
    n_iter += 1
    cv_accuracy.append(accuracy)
    knn_idx.append(tr_idx)

print("max_val : ",max(cv_accuracy)*100)
train_idx = knn_idx[0]

X_train, y_train = X_train2.loc[train_idx,], y_train2[train_idx]

knn = KNeighborsClassifier(algorithm='kd_tree', leaf_size=10, n_neighbors=15, p=1, weights='uniform')
knn.fit(X_train,y_train)
print("train acc : ", np.round(accuracy_score(knn.predict(X_train), y_train),5)*100)
#0.7665937272064187
pred = knn.predict_proba(X_test)
# 상위 10에 들면 True
count=0
for prednum in range(len(pred)):
    prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
    if y_test[prednum] in prediction10:
        count+=1
print("Accuracy : ", count/len(y_test)*100)
#0.6909620991253644





# =============================================================================
# knn.score(X_test,y_test)
# #0.6909620991253644
# 
# ###Elbow method for optimum value of K
# #error_rate = []
# #for i in range(1,20):
# #    knn = KNeighborsClassifier(n_neighbors=i)
# #    knn.fit(X_train,y_train)
# #    pred_i = knn.predict(X_test)
# #    error_rate.append(np.mean(pred_i != y_test))
#     
# #plt.figure(figsize=(10,6))
# #plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',
# #         markerfacecolor='red', markersize=5)
# #plt.title('Error Rate vs. K Value')
# #plt.xlabel('K')
# #plt.ylabel('Error Rate')
# 
# pred = knn.predict(X_test)
# print(classification_report(y_test,pred))
# 
# matrix=confusion_matrix(y_test,pred)
# print(matrix)
# 
# 
# ############################
# #KNNeighbours
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p =2)
# classifier.fit(X_train, y_train)
# 
# #predicting the values
# y_pred = classifier.predict(X_test)
# 
# #confusion matrix
# cm_KNN = confusion_matrix(y_test, y_pred)
# print(cm_KNN)
# 
# #Accuracy
# accuracy_score(y_test,y_pred)
# #0.53
# 
# #Report
# report_KNN = classification_report(y_test, y_pred)
# print(report_KNN)
# =============================================================================

###############################################################################
##SVM
###############################################################################
#SVM
from sklearn.svm import SVC
classifier = SVC(random_state =2, probability=True)

## best hyperparameter search
params = {'C':[0.01, 0.1, 1, 5, 10],
          'kernel':['poly','rbf','sigmoid'],
          'gamma':['scale','auto'],
          'decision_function_shape':['ovo','ovr']}
grid_model = GridSearchCV(classifier, param_grid=params, scoring='accuracy', cv=10)
grid_model.fit(X_train2, y_train2)
print('최적 하이퍼 파라미터 : {}, 최적 평균 정확도 : {}'.format(grid_model.best_params_, grid_model.best_score_))
#


ss = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_accuracy = []
n_iter =0
classifier_idx=[]
for tr_idx, val_idx in ss.split(X_train2, y_train2):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할
    X_train, X_val = X_train2.loc[tr_idx,], X_train2.loc[val_idx,]
    y_train, y_val = y_train2[tr_idx], y_train2[val_idx]
    
    classifier.fit(X_train, y_train)
    pred = classifier.predict_proba(X_val)
    
    count=0
    for prednum in range(len(pred)):
        prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
        if y_val[prednum] in prediction10:
            count+=1
    accuracy = count/len(y_val)
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, train_size, val_size))
    n_iter += 1
    cv_accuracy.append(accuracy)
    classifier_idx.append(tr_idx)

print("max_val : ",max(cv_accuracy)*100)
train_idx = classifier_idx[7]

X_train, y_train = X_train2.loc[train_idx,], y_train2[train_idx]

classifier = SVC(kernel = 'rbf', random_state = 2, C = 1, probability=True)
classifier.fit(X_train,y_train)
print("train acc : ",np.round(accuracy_score(classifier.predict(X_train), y_train),5)*100)
#0.7148067104303428
pred = classifier.predict_proba(X_test)
# 상위 10에 들면 True
count=0
for prednum in range(len(pred)):
    prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
    if y_test[prednum] in prediction10:
        count+=1
print("Accuracy : ", count/len(y_test)*100)
#0.6676384839650146



# =============================================================================
# print(classifier.support_vectors_)
# #결정경계를 정의하는 서포트 벡터 확인
# 
# classifier.score(X_test,y_test)
# #
# 
# #predicting the values
# y_pred = classifier.predict(X_test)
# 
# #confusion matrix
# cm_SVC = confusion_matrix(y_test, y_pred)
# print(cm_SVC)
# 
# #Accuracy
# accuracy_score(y_test,y_pred)
# #0.965
# 
# #Report
# report_SVC = classification_report(y_test, y_pred)
# print(report_SVC)
# =============================================================================

###############################################################################
##NAIVE BAYES CLASSIFIER
###############################################################################
'''
from sklearn.naive_bayes import GaussianNB
일반적인 연속값 속성을 가지는 데이터에 사용

from sklearn.naive_bayes import MultinomialNB
클래스 여러개(다항)일 때 주로 사용

from sklearn.naive_bayes import BernoulliNB
클래스가 1,0같이 이진 속성일 경우 두개로 구분되는 binary일 때 주로 사용
'''

#NaiveBayes
"""
from sklearn.naive_bayes import BernoulliNB
naive_classifier = BernoulliNB()

naive_classifier.fit(X_train,y_train)
print(accuracy_score(naive_classifier.predict(X_train), y_train))
#0.6637490882567469
pred = naive_classifier.predict(X_test)
print(accuracy_score(pred, y_test))
#0.6209912536443148
"""
from sklearn.naive_bayes import MultinomialNB
naive_classifier = MultinomialNB(alpha=5)

## best hyperparameter search
params = {'alpha':[0.01, 0.1, 1, 5, 10]}
grid_model = GridSearchCV(naive_classifier, param_grid=params, scoring='accuracy', cv=10)
grid_model.fit(X_train2, y_train2)
print('최적 하이퍼 파라미터 : {}, 최적 평균 정확도 : {}'.format(grid_model.best_params_, grid_model.best_score_))
#최적 하이퍼 파라미터 : {'alpha': 5}


ss = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_accuracy = []
n_iter =0
naive_classifier_idx=[]
for tr_idx, val_idx in ss.split(X_train2, y_train2):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할
    X_train, X_val = X_train2.loc[tr_idx,], X_train2.loc[val_idx,]
    y_train, y_val = y_train2[tr_idx], y_train2[val_idx]
    
    naive_classifier.fit(X_train, y_train)
    pred = naive_classifier.predict_proba(X_val)
    
    count=0
    for prednum in range(len(pred)):
        prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
        if y_val[prednum] in prediction10:
            count+=1
    accuracy = count/len(y_val)
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, train_size, val_size))
    n_iter += 1
    cv_accuracy.append(accuracy)
    naive_classifier_idx.append(tr_idx)

print("max_val : ",max(cv_accuracy)*100)
train_idx = naive_classifier_idx[2]

X_train, y_train = X_train2.loc[train_idx,], y_train2[train_idx]

naive_classifier = MultinomialNB(alpha=5)
naive_classifier.fit(X_train,y_train)
print("train acc : ", np.round(accuracy_score(naive_classifier.predict(X_train), y_train),5)*100)
#0.6637490882567469
pred = naive_classifier.predict_proba(X_test)
# 상위 10에 들면 True
count=0
for prednum in range(len(pred)):
    prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
    if y_test[prednum] in prediction10:
        count+=1
print("Accuracy : ", count/len(y_test)*100)






# =============================================================================
# classifier.score(X_test,y_test)
# 
# #predicting the values
# y_pred = classifier.predict(X_test)
# 
# #confusion matrix
# cm_NB = confusion_matrix(y_test, y_pred)
# print(cm_NB)
# 
# #Accuracy
# accuracy_score(y_test,y_pred)
# #0.79
# 
# #Report
# report_NB = classification_report(y_test, y_pred)
# print(report_NB)
# =============================================================================

###############################################################################
##GRADIENT BOOSTING CLASSIFIER
###############################################################################
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
GB_classifier = GradientBoostingClassifier(random_state=1)

ss = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cv_accuracy = []
n_iter =0
GB_classifier_idx=[]
for tr_idx, val_idx in ss.split(X_train2, y_train2):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할
    X_train, X_val = X_train2.loc[tr_idx,], X_train2.loc[val_idx,]
    y_train, y_val = y_train2[tr_idx], y_train2[val_idx]
    
    GB_classifier.fit(X_train, y_train)
    pred = GB_classifier.predict_proba(X_val)
    
    count=0
    for prednum in range(len(pred)):
        prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
        if y_val[prednum] in prediction10:
            count+=1
    accuracy = count/len(y_val)
    
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    
    print('\n#{0} 교차 검증 정확도 : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
          .format(n_iter, accuracy, train_size, val_size))
    n_iter += 1
    cv_accuracy.append(accuracy)
    GB_classifier_idx.append(tr_idx)

print("max_val : ",max(cv_accuracy)*100)
train_idx = GB_classifier_idx[7]

X_train, y_train = X_train2.loc[train_idx,], y_train2[train_idx]

GB_classifier = GradientBoostingClassifier(random_state=1)
GB_classifier.fit(X_train,y_train)
print("train acc : ", np.round(accuracy_score(GB_classifier.predict(X_train), y_train),5)*100)
#0.8016046681254558
pred = GB_classifier.predict_proba(X_test)
# 상위 10에 들면 True
count=0
for prednum in range(len(pred)):
    prediction10 = sorted(range(len(pred[prednum])),key= lambda i: pred[prednum][i])[-10:]
    if y_test[prednum] in prediction10:
        count+=1
print("Accuracy : ", count/len(y_test)*100)
#0.6909620991253644






# =============================================================================
# #predicting the values
# y_pred = classifier.predict(X_test)
# 
# #confusion matrix
# cm_GBC = confusion_matrix(y_test, y_pred)
# print(cm_GBC)
# 
# #Accuracy
# accuracy_score(y_test,y_pred)
# #0.8816666666666667
# 
# #Report
# report_GBC = classification_report(y_test, y_pred)
# print(report_GBC)
# =============================================================================
