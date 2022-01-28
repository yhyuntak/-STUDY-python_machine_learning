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

## confusion matrix ##
print("-"*30)
print("## confusion matrix ##")
print("-"*30)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,fakepred))

## precision & recall ##
print("-"*30)
print("## precision & recall ##")
print("-"*30)

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

titanic_df = pd.read_csv("../../dataset/titanic_train.csv")
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived',axis=1)
X_titanic_df = ut.transform_features(X_titanic_df)

X_train,X_test,y_train,y_test = train_test_split(X_titanic_df,y_titanic_df,test_size=0.2,random_state=11)

lr_clf = LogisticRegression()
lr_clf.fit(X_train,y_train)
pred = lr_clf.predict(X_test)
ut.get_clf_eval(y_test,pred)

pred_proba = lr_clf.predict_proba(X_test)
print("pred_proba 결과 :",pred_proba[:3])
print("pred 결과 : ",pred[:3].reshape(-1,1))

from sklearn.preprocessing import Binarizer

custom_threshold = 0.2
pred_proba_1 = pred_proba[:1].reshape(-1,1)
binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict = binarizer.transform(pred_proba)
custom_predict_space = np.zeros((custom_predict.shape[0],1))
for idx,pred_val in enumerate(custom_predict):
    if pred_val[1] == 1 :
        custom_predict_space[idx] = 1
custom_predict = custom_predict_space.reshape(-1).astype(int)
# print(custom_predict)
# print(pred.shape)
ut.get_clf_eval(y_test,custom_predict)

from sklearn.metrics import precision_recall_curve

pred_proba_class1 = lr_clf.predict_proba(X_test)[:,1]
precisions,recalls,thresholds = precision_recall_curve(y_test,pred_proba_class1)
print("반환된 분류 결정 임곗값 배열의 shape:",thresholds.shape)

thr_index = np.arange(0,thresholds.shape[0],15)
print("샘플 추출을 위한 임계값 배열의 index 10개:",thr_index)
print("샘플용 10개의 임계값:",np.round(thresholds[thr_index],2))

print("샘플 임계값별 정밀도:",np.round(precisions[thr_index],2))
print("샘플 임계값별 재현율:",np.round(recalls[thr_index],2))

## F1 score ##
print("-"*30)
print("## F1 score ##")
print("-"*30)

from sklearn.metrics import f1_score
f1=f1_score(y_test,pred)
print("F1 스코어: {0:.4f}".format(f1))
