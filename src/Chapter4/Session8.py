import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

cust_df = pd.read_csv("../../dataset/santander/train.csv",encoding='latin-1')
print(cust_df.shape)
cust_df.info()
print(cust_df['TARGET'].value_counts())
unsatisfied_cnt = cust_df[cust_df['TARGET']==1].count()
total_cnt = cust_df.TARGET.count()

print(cust_df.describe())
print(cust_df['var3'].value_counts())

cust_df['var3'].replace(-999999,2,inplace=True)
cust_df.drop('ID',axis=1,inplace=True)

X_features = cust_df.iloc[:,:-1]
y_labels = cust_df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_features,y_labels,test_size=0.2)
train_cnt = y_train.count()
test_cnt = y_test.count()

print(train_cnt)
print(y_train.value_counts())

print("학습 세트 레이블 값 분포 비율 : \n",y_train.value_counts()/train_cnt)
print("테스트 세트 레이블 값 분포 비율 : \n",y_test.value_counts()/test_cnt)

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

xgb_clf = XGBClassifier(n_estimators=100)
params = {'max_depth':[5,7],'min_child_weight':[1,3],'colsample_bytree':[0.5,0.75]}
gridcv = GridSearchCV(xgb_clf,param_grid=params,cv=3)
gridcv.fit(X_train,y_train,early_stopping_rounds=30,eval_metric="auc",eval_set=[(X_train,y_train),(X_test,y_test)])
print(gridcv.best_params_)
xgb_roc_score=roc_auc_score(y_test,gridcv.predict_proba(X_test)[:,1],average='macro')
print(xgb_roc_score)

