import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer_data = load_breast_cancer()
X_data =cancer_data.data
y_label =cancer_data.target

X_train,X_test,y_train,y_test = train_test_split(X_data,y_label,test_size=0.2,random_state=0)

knn_clf = KNeighborsClassifier(n_neighbors=4)
rf_clf = RandomForestClassifier(n_estimators=100,random_state=0)
dt_clf = DecisionTreeClassifier()
ada_clf = AdaBoostClassifier(n_estimators=100)

lr_final = LogisticRegression(C=10)
knn_clf.fit(X_train,y_train)
rf_clf.fit(X_train,y_train)
dt_clf.fit(X_train,y_train)
ada_clf.fit(X_train,y_train)

knn_pred = knn_clf.predict((X_test))
rf_pred = rf_clf.predict((X_test))
dt_pred = dt_clf.predict((X_test))
ada_pred = ada_clf.predict((X_test))

print('KNN : ',accuracy_score(y_test,knn_pred))
print('랜덤 포레스트 : ',accuracy_score(y_test,rf_pred))
print('결정 트리 : ',accuracy_score(y_test,dt_pred))
print('에이다부스트 : ',accuracy_score(y_test,ada_pred))

pred= np.array([knn_pred,rf_pred,dt_pred,ada_pred])
pred=np.transpose(pred)
lr_final.fit(pred,y_test)
final = lr_final.predict(pred)
print(accuracy_score(y_test,final))