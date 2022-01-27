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

## Stratified K fold ##
print("-"*30)
print(" ## Stratified K fold ##")
print("-"*30)
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df['label']=iris.target
# print(iris_df['label'].value_counts())

kfold = KFold(n_splits=3)
n_iter = 0

for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    # print(label_train.value_counts())
    # print(label_test.value_counts())

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
n_iter = 0

for train_index,test_index in skf.split(iris_df,iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print(label_train.value_counts())
    print(label_test.value_counts())

## cross_val_score() ##
print("-"*30)
print(" ## cross_val_score() ##")
print("-"*30)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

scores = cross_val_score(dt_clf,data,label,scoring='accuracy',cv=3)
print('교차 검증별 정확도:',np.round(scores,4))
print('평균 검증 정확도:',np.round(np.mean(scores),4))

## GridSearchCV ##
print("-"*30)
print(" ## GridSearchCV ##")
print("-"*30)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

iris_data = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=121)
dtree = DecisionTreeClassifier()

parameters = {'max_depth':[1,2,3],'min_samples_split':[2,3]}
grid_dtrees = GridSearchCV(estimator=dtree,param_grid=parameters,cv=3,refit=True)
grid_dtrees.fit(X_train,y_train)

scores_df = pd.DataFrame(grid_dtrees.cv_results_)
print(scores_df[['params','mean_test_score', 'std_test_score', 'rank_test_score',
       'split0_test_score', 'split1_test_score', 'split2_test_score']])
print("GRidSearchCV 최적 파라미터:",grid_dtrees.best_params_)
print("GRidSearchCV 최고 정확도:{0:.4f}:".format(grid_dtrees.best_score_))

best_estimator = grid_dtrees.best_estimator_

pred = best_estimator.predict(X_test)
print("테스트 데이터 세트 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

