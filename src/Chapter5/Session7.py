import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
data_scaled = scaler.fit_transform(cancer.data)

X_train,X_test,y_train,y_test = train_test_split(data_scaled,cancer.target,test_size=0.3)

from sklearn.metrics import accuracy_score,roc_auc_score

lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
lr_preds = lr_clf.predict(X_test)

print("accuracy : {0:.3f}".format(accuracy_score(y_test,lr_preds)))
print("roc_auc : {0:.3f}".format(roc_auc_score(y_test,lr_preds)))


from sklearn.model_selection import GridSearchCV

params = {'penalty':['l2','l1'],
          'C':[0.01,0.1,1,1,5,10]}

grid_clf = GridSearchCV(lr_clf,param_grid=params,scoring='accuracy',cv=3)
grid_clf.fit(data_scaled,cancer.target)
print("최적 하이퍼 파라미터 : {0}, 최적 평균 정확도:{1:.3f}".format(grid_clf.best_params_,grid_clf.best_score_))
