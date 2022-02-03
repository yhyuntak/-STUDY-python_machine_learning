from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils.util as util
import warnings
warnings.filterwarnings('ignore')

dt_clf = DecisionTreeClassifier()
iris_data = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2)

dt_clf.fit(X_train,y_train)

# from sklearn.tree import export_graphviz
# export_graphviz(dt_clf,out_file="tree.dot",class_names=iris_data.target_names,\
#                 feature_names=iris_data.feature_names,impurity=True,filled=True)
#
# from IPython import display
# import graphviz
# with open("tree.dot") as f:
#     dot_graph = f.read()
# display(graphviz.Source(dot_graph))

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
print("Feature importances:\n{0}".format(np.round(dt_clf.feature_importances_,3)))

for name,value in zip(iris_data.feature_names,dt_clf.feature_importances_):
    print("{0} : {1:.3f}".format(name,value))

# plt.figure()
# sns.barplot(x=dt_clf.feature_importances_,y=iris_data.feature_names)
# plt.show()

## 결정 트리 과적합 ##
print("-"*30)
print("## 결정 트리 과적합 ##")
print("-"*30)

from sklearn.datasets import make_classification

# plt.title("3 CLass values with 2 Features sample data creation")

X_features,y_lables = make_classification(n_features=2,n_redundant=0,n_informative=2,
                                          n_classes=3,n_clusters_per_class=1,random_state=0)

# plt.scatter(X_features[:,0],X_features[:,1],marker='o',c=y_lables,s=25,edgecolors='k')


# Classifier의 Decision Boundary를 시각화 하는 함수
def visualize_boundary(model, X, y):
    fig, ax = plt.subplots()

    # 학습 데이타 scatter plot으로 나타내기
    ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start, xlim_end = ax.get_xlim()
    ylim_start, ylim_end = ax.get_ylim()

    # 호출 파라미터로 들어온 training 데이타로 model 학습 .
    model.fit(X, y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행.
    xx, yy = np.meshgrid(np.linspace(xlim_start, xlim_end, num=200), np.linspace(ylim_start, ylim_end, num=200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # contourf() 를 이용하여 class boundary 를 visualization 수행.
    n_classes = len(np.unique(y))

    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', clim=(y.min(), y.max()),
                           zorder=1)

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(min_samples_leaf=6).fit(X_features,y_lables)
# visualize_boundary(dt_clf,X_features,y_lables)
#
# plt.show()

## 결정 트리 실습 ##
print("-"*30)
print("## 결정 트리 실습 ##")
print("-"*30)

import pandas as pd

feature_name_df = pd.read_csv("../../dataset/human_activity/features.txt",sep='\s+',header=None,names=['column_index','column_name'])
feature_name = feature_name_df.iloc[:,1]
feature_name = feature_name.values.tolist()

feature_dup_df = feature_name_df.groupby('column_name').count()
print(feature_dup_df[feature_dup_df['column_index']>1].count())
print(feature_dup_df[feature_dup_df['column_index']>1].head())

X_train,X_test,y_train,y_test = util.get_human_dataset()

print("## 학습 피처 데이터 셋 info()")
print(X_train.info())
print(X_train.head(3))
print(y_test.head(3))
print(y_train['action'].value_counts())


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)
pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test,pred)
print("결정 트리 예측 정확도 : {0:.4f}".format(accuracy))

print("분류기 기본 하이퍼 파라미터 :\n",dt_clf.get_params())

from sklearn.model_selection import GridSearchCV
params = {    'max_depth' : [8,12],'min_samples_split':[16,24] }
grid_cv = GridSearchCV(dt_clf,param_grid=params,scoring='accuracy',cv=5,verbose=1)
grid_cv.fit(X_train,y_train)
print("GSCV 최고 평균 정확도 수치 : {0:.4f}".format(grid_cv.best_score_))
print("GSCV 최적 하이퍼 파라미터 : ",grid_cv.best_params_)

best_df_clf = grid_cv.best_estimator_
pred = best_df_clf.predict(X_test)
accuracy = accuracy_score(y_test,pred)
print("accuracy : ",accuracy)

ftr_importances_values = best_df_clf.feature_importances_
ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
plt.figure()
plt.title('Top 20')
sns.barplot(x=ftr_top20,y=ftr_top20.index)
plt.show()

