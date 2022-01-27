import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('../../dataset/titanic_train.csv')
print(titanic_df.info())
print('데이터 세트 Null 값 개수:',titanic_df.isnull().sum().sum())

titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)
titanic_df['Cabin'].fillna('N',inplace=True)
titanic_df['Embarked'].fillna('N',inplace=True)
print('데이터 세트 Null 값 개수:',titanic_df.isnull().sum().sum())

print("Sex 값 분포:",titanic_df['Sex'].value_counts())
print("Cabin 값 분포:",titanic_df['Cabin'].value_counts())
print("Embarked 값 분포:",titanic_df['Embarked'].value_counts())

titanic_df['Cabin']=titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))
print(titanic_df.groupby(['Sex','Survived'])['Survived'].count())

# plt.figure(1)
# sns.barplot(x='Sex',y='Survived',data=titanic_df)
# plt.figure(2)
# sns.barplot(x='Pclass',y='Survived',hue='Sex',data=titanic_df)
# plt.show()

def get_category(age):
    cat = ''
    if age<=1 -1: cat = 'Unknown'
    elif age<=5: cat = 'Baby'
    elif age<= 12: cat = 'Child'
    elif age <= 18 : cat = 'Teenager'
    elif age <= 25 : cat = 'Student'
    elif age <= 35 : cat = 'Young Adult'
    elif age <= 60 : cat = 'Adult'
    else : cat = 'Elderly'

    return cat

# plt.figure(figsize=(10,6))
# group_names = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Elderly']
# titanic_df['Age_cat']=titanic_df['Age'].apply(lambda x : get_category(x))
# sns.barplot(x='Age_cat',y='Survived',hue='Sex',data=titanic_df,order=group_names)
# titanic_df.drop('Age_cat',axis=1,inplace=True)
# plt.show()

from sklearn.preprocessing import LabelEncoder
def encode_feature(dataDF):
    features=['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
    return dataDF
titanic_df = encode_feature(titanic_df)
print(titanic_df.head())

y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop(['Survived','Name','PassengerId','Ticket'],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df,test_size=0.2,random_state=11)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
lr_clf = LogisticRegression()

dt_clf.fit(X_train,y_train)
dt_pred = dt_clf.predict(X_test)
print("DecisionTreeClassifier 정확도 평가 :{0:.4f}".format(accuracy_score(y_test,dt_pred)))

rf_clf.fit(X_train,y_train)
rf_pred = rf_clf.predict(X_test)
print("RandomForestClassifier 정확도 평가 :{0:.4f}".format(accuracy_score(y_test,rf_pred)))

lr_clf.fit(X_train,y_train)
lr_pred = lr_clf.predict(X_test)
print("RandomForestClassifier 정확도 평가 :{0:.4f}".format(accuracy_score(y_test,lr_pred)))


from sklearn.model_selection import KFold
def exec_kfold(clf,folds=5):
    kfold=KFold(n_splits=folds)
    scores=[]

    for iter_count,(train_idx,test_idx) in enumerate(kfold.split(X_titanic_df)):
        X_train,X_test = X_titanic_df.values[train_idx],X_titanic_df.values[test_idx]
        y_train,y_test = y_titanic_df.values[train_idx],y_titanic_df.values[test_idx]
        clf.fit(X_train,y_train)
        pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test,pred)
        scores.append(accuracy)
        print("교차 검증 {0} 정확도:{1:.4f}".format(iter_count,accuracy))

    mean_score = np.mean(scores)
    print("평균 정확도 :{0:.4f}".format(mean_score))
exec_kfold(dt_clf,folds=5)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_clf,X_titanic_df,y_titanic_df,cv=5)
for iter,accuracy in enumerate(scores):
    print("교차 검증 {0} 정확도 : {1:.4f}".format(iter,accuracy))
print("평균 정확도 :{0:.4f}".format(np.mean(scores)))

from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':[2,3,5,10],'min_samples_split':[2,3,5],'min_samples_leaf':[1,5,8]}
grid_dclf = GridSearchCV(dt_clf,param_grid=parameters,scoring='accuracy',cv=5)
grid_dclf.fit(X_train,y_train)

print('GridSearchCV 최적 하이퍼 파라미터 : ',grid_dclf.best_params_)
print('GridSearchCV 최고 정확도 :{0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_

dpredictions = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test,dpredictions)
print("베스트 결정트리 정확도 :{0:.4f}".format(accuracy))

