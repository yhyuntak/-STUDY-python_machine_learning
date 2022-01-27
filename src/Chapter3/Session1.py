from sklearn.base import BaseEstimator
import numpy as np
import utils_titanic as ut
class MyDummyClassifier(BaseEstimator):
    def fit(self,X,y=None):
        pass
    def predict(self,X):
        pred=np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            if X['Sex'].iloc[i]==1:
                pred[i]=0
            else :
                pred[i] =1

        return pred

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv("../../dataset/titanic_train.csv")
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)
X_titanic_df = ut.transform_features(X_titanic_df)
X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df,test_size=0.2)
mydcf = MyDummyClassifier()
mydcf.fit(X_train,y_train)
mypredictions = mydcf.predict(X_test)
print("Dummy Classifier의 정확도는 {0:.4f}".format(accuracy_score(y_test,mypredictions)))


from sklearn.datasets import load_digits

class MyFakeClassifier(BaseEstimator):
    def fit(self,X,y):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)

mnist = load_digits()
y = (mnist.target==7).astype(int)
X_train,X_test,y_train,y_test = train_test_split(mnist.data,y)

print("레이블 테스트 세트 크기:",y_test.shape)
print("테스트 세트 레이블 0과 1의 분포도")
print(pd.Series(y_test).value_counts())

fakeclf = MyFakeClassifier()
fakeclf.fit(X_train,y_train)
fakepred = fakeclf.predict(X_test)
print("정확도 : ",accuracy_score(y_test,fakepred))

