import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
X_features = dataset.data
y_label = dataset.target
cancer_df = pd.DataFrame(data=X_features,columns=dataset.feature_names)
cancer_df['target'] = y_label

print(dataset.target_names)
print(cancer_df['target'].value_counts())

X_train,X_test,y_train,y_test = train_test_split(X_features,y_label,test_size=0.2)

dtrain =xgb.DMatrix(data=X_train,label=y_train)
dtest = xgb.DMatrix(data=X_test,label=y_test)

params = {'max_depth':3,
          'eta':0.1,
          'objective':'binary:logistic',
          'eval_metric':'logloss',
          'early_stoppings':100
          }
num_rounds = 400

wlist = [(dtrain,'train'),(dtest,'eval')]
xgb_model = xgb.train(params=params,dtrain=dtrain,num_boost_round = num_rounds, early_stopping_rounds = 100,evals =wlist)
pred_probs = xgb_model.predict(dtest)
print(pred_probs.shape)
print(np.round(pred_probs[:10],3))
preds = [1 if x>0.5 else 0 for x in pred_probs]
print(preds[:10])

import matplotlib.pyplot as plt

# fig,ax = plt.subplots(figsize=(10,12))
# plot_importance(xgb_model,ax=ax)
# plt.show()

from xgboost import XGBClassifier
xgb_wrapper = XGBClassifier(n_estimators=400,learning_rate=0.1,max_depth=3)
evals = [(X_test,y_test)]
xgb_wrapper.fit(X_train,y_train,early_stopping_rounds = 10,eval_metric="logloss",eval_set=evals,verbose=True)
w_preds = xgb_wrapper.predict(X_test)
w_pred_proba = xgb_wrapper.predict_proba(X_test)[:,1]
w_pred_label = [1 if x>0.5 else 0 for x in w_pred_proba]
print(w_pred_label[:10])
