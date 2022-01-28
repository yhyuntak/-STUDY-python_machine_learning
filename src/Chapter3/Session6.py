import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils_titanic as ut

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score,\
    confusion_matrix,precision_recall_curve,roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

diabetes_data = pd.read_csv("../../dataset/diabetes.csv")
print(diabetes_data.keys())
print(diabetes_data['Outcome'].value_counts())
print(diabetes_data.info())

y_data = diabetes_data['Outcome']
X_data = diabetes_data.drop(['Outcome'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.2,random_state=156,stratify=y_data)

lr_clf=LogisticRegression()
lr_clf.fit(X_train,y_train)
pred=lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:,1]
ut.get_clf_eval(y_test,pred,pred_proba)

# ut.precision_recall_curve_plot(y_test,pred_proba)

print(diabetes_data.describe())

# plt.figure()
# plt.hist(diabetes_data['Glucose'],bins=10)
# plt.show()

zero_features = ['Glucose', 'BloodPressure',  'SkinThickness',     'Insulin',         'BMI']
total_count = diabetes_data['Glucose'].count()
for feature in zero_features :
    zero_count = diabetes_data[diabetes_data[feature]==0][feature].count()
    print("{0}의 min이 0인 건수는 {1}, 퍼센트는 {2:.2f}%".format(feature,zero_count,zero_count/total_count*100))

mean_zero_features = diabetes_data[zero_features].mean()
print(mean_zero_features)
diabetes_data[zero_features] = diabetes_data[zero_features].replace(0,mean_zero_features)

new_y_data = diabetes_data['Outcome']
# new_y_data = pd.DataFrame(data=new_X_data,columns=['Outcome'])
new_X_data = diabetes_data.drop(['Outcome'],axis=1)

scaler = StandardScaler()
X_Scaled = scaler.fit_transform(new_X_data)
X_train,X_test,y_train,y_test = train_test_split(X_Scaled,new_y_data,test_size=0.2,random_state=156,stratify=new_y_data)

new_lr_clf = LogisticRegression()
new_lr_clf.fit(X_train,y_train)
pred=new_lr_clf.predict(X_test)
pred_proba=new_lr_clf.predict_proba(X_test)[:,1]

ut.get_clf_eval(y_test,pred,pred_proba)

thresholds = np.array([0.3,0.33,0.36,0.39,0.42,0.45,0.48,0.5])
pred_proba = new_lr_clf.predict_proba(X_test)
ut.get_eval_by_threshold(y_test,pred_proba[:,1].reshape(-1,1),thresholds)

#
# from sklearn.metrics import precision_recall_curve
# pred_proba_class1 = new_lr_clf.predict_proba(X_test)[:,1]
# prec,rec,thres = precision_recall_curve(y_test,pred_proba_class1)
# thr_idx = np.arange(0,thres.shape[0],15)
# print("10개의 임계값   : ",np.round(thres[thr_idx],2))
# print("임계값별 정밀도 : ", np.round(prec[thr_idx],2))
# print("임계값별 재현율 : ", np.round(rec[thr_idx],2))
