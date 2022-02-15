from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

boston=load_boston()
boston_df = pd.DataFrame(data=boston.data,columns=boston.feature_names)
boston_df['PRICE'] = boston.target
y_target = boston_df['PRICE']
X_data = boston_df.drop(['PRICE'],axis=1,inplace=False)

# RF = RandomForestRegressor(n_estimators=1000)
# neg_mse_scores = cross_val_score(RF,X_data,y_target,scoring="neg_mean_squared_error",cv=5)
# rmse_scores = np.sqrt(-1*neg_mse_scores)
# avg_rmse = np.mean(rmse_scores)
#
# print("-" * 20)
# print("5 folds의 개별 Negative MSE scores: ", np.round(neg_mse_scores, 2))
# print("5 folds의 개별 RMSE scores: ", np.round(rmse_scores, 2))
# print("5 folds의 평균 RMSE: ", avg_rmse)
# print("-" * 20)

def get_model_cv_prediction(model,X_data,y_target):
    neg_mse_scores = cross_val_score(model,X_data,y_target,scoring="neg_mean_squared_error",cv=5)
    rmse_scores = np.sqrt(-1*neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print("####",model.__class__.__name__,"####")
    print("5교차 검증의 평균 RMSE : {0:.3f}".format(avg_rmse))

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

dt_reg = DecisionTreeRegressor(max_depth=4)
rf_reg = RandomForestRegressor(n_estimators=1000)
gb_reg = GradientBoostingRegressor(n_estimators=1000)
xgb_reg = XGBRegressor(n_estimators=1000)
lgb_reg = LGBMRegressor(n_estimators=1000)

models = [dt_reg,rf_reg,gb_reg,xgb_reg,lgb_reg]
for model in models :
    get_model_cv_prediction(model,X_data,y_target)

