from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris= load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data,train_label)

pred=dt_clf.predict(train_data)
print(accuracy_score(train_label,pred))


from sklearn.model_selection import KFold
import numpy as np

features = iris.data
label = iris.target
dt_clf_kfold = DecisionTreeClassifier()
kfold = KFold(n_splits=5)
cv_accuracy = []

n_iter = 0
for train_index,test_index in kfold.split((features)):
    X_train,X_test = features[train_index],features[test_index]
    y_train,y_test = label[train_index],label[test_index]

    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    n_iter +=1

    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n{0} : {1} , {2} , {3} , {0}'.format(n_iter,accuracy,train_size,test_size))
    cv_accuracy.append(accuracy)

print(cv_accuracy)
print(np.mean(cv_accuracy))