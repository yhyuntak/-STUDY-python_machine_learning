import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils.util as util

card_df = pd.read_csv('../../dataset/creditcard.csv')
print(card_df.head(3))

X_train,X_test,y_train,y_test =util.get_train_test_dataset(card_df)
print("학습 데이터 레이블 값 비율")
print(y_train.value_counts()/y_train.count())
print("테스트 데이터 레이블 값 비율")
print(y_test.value_counts()/y_test.count())

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_over,y_train_over = smote.fit_resample(X_train,y_train)

from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_clf.fit(X_train_over,y_train_over)
lr_pred = lr_clf.predict(X_test)
lr_pred_proba = lr_clf.predict_proba(X_test)[:,1]
util.get_clf_eval(y_test,lr_pred,lr_pred_proba)


from lightgbm import LGBMClassifier
lgbm_clf = LGBMClassifier(n_estimator=1000,num_leaves=64,n_job=-1,boost_from_average=False)
util.get_model_train_eval(lgbm_clf,ftr_train=X_train_over,ftr_test=X_test,tgt_train=y_train_over,tgt_test=y_test)



#
import seaborn as sns
# plt.figure()
# plt.xticks(range(0,30000,1000),rotation=60)
# sns.distplot(card_df['Amount'])
# plt.show()
#
#
# plt.figure()
# corr=card_df.corr()
# sns.heatmap(corr,cmap='RdBu')
# plt.show()


