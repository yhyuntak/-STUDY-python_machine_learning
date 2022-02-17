import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bike_df = pd.read_csv("../../dataset/bike-sharing-demand/train.csv")
print(bike_df.info())

bike_df['datetime'] = bike_df.datetime.apply(pd.to_datetime)
bike_df['year'] = bike_df.datetime.apply(lambda x : x.year)
bike_df['month'] = bike_df.datetime.apply(lambda x : x.month)
bike_df['day'] = bike_df.datetime.apply(lambda x : x.day)
bike_df['hour'] = bike_df.datetime.apply(lambda x : x.hour)
drop_columns = ['datetime','casual','registered']
bike_df.drop(drop_columns,axis=1,inplace=True)

from sklearn.metrics import mean_squared_error,mean_absolute_error

def rmsle(y,pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = np.square(log_y - log_pred)
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

def rmse(y,pred):
    return np.sqrt(mean_squared_error(y,pred))

def evaluate_regr(y,pred):
    rmsle_val = rmsle(y,pred)
    rmse_val = rmse(y,pred)
    mae_val = mean_absolute_error(y,pred)
    print("RMSLE : {0:.3f}, RMSE : {1:.3f}, MAE : {2:.3f}".format(rmsle_val,rmse_val,mae_val))


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso

y_target = bike_df['count']
X_features = bike_df.drop(['count'],axis=1,inplace=False)
X_train,X_test,y_train,y_test = train_test_split(X_features,y_target,test_size=0.3)

lr_reg = LinearRegression()
lr_reg.fit(X_train,y_train)
pred = lr_reg.predict(X_test)

evaluate_regr(y_test,pred)

def get_top_error_data(y_test,pred,n_tops = 5):
    result_df = pd.DataFrame(y_test.values,columns=['real_count'])
    result_df['predicted_count'] = np.round(pred)
    result_df['diff'] = np.abs(result_df['real_count']-result_df['predicted_count'])

    print(result_df.sort_values('diff',ascending=False)[:n_tops])
#
# get_top_error_data(y_test,pred,n_tops=5)

# y_target.hist()
# plt.show()
#
y_target_log = np.log1p(y_target)
X_train,X_test,y_train,y_test = train_test_split(X_features,y_target_log,test_size=0.3)
lr_reg.fit(X_train,y_train)
pred = lr_reg.predict(X_test)
evaluate_regr(np.expm1(y_test),np.expm1(pred))

# coef = pd.Series(lr_reg.coef_,index=X_features.columns)
# coef_sort = coef.sort_values(ascending=False)
# sns.barplot(x=coef_sort.values,y=coef_sort.index)
# plt.show()

print(X_features.head(10))
X_features_ohe = pd.get_dummies(X_features,columns=['year','month','day','hour','holiday','workingday','season','weather'])
print(X_features_ohe.head(10))

X_train,X_test,y_train,y_test = train_test_split(X_features_ohe,y_target_log,test_size=0.3)

def get_model_predict(model,X_train,X_test,y_train,y_test,is_expm1=False):
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    if is_expm1 :
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    print("#"*5,model.__class__.__name__,'#'*5)
    evaluate_regr(y_test,pred)

lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=10)
lasso_reg = Lasso(alpha=0.01)

for model in [lr_reg,ridge_reg,lasso_reg]:
    get_model_predict(model,X_train,X_test,y_train,y_test,is_expm1=True)

# coef = pd.Series(data=lr_reg.coef_,index=X_features_ohe.columns)
# coef_sort = coef.sort_values(ascending=False)[:20]
# sns.barplot(x=coef_sort,y=coef_sort.index)
# plt.show()

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

rf_reg = RandomForestRegressor(n_estimators=500)
gbm_reg = GradientBoostingRegressor(n_estimators=500)
xgb_reg = XGBRegressor(n_estimators=500)
lgbm_reg = LGBMRegressor(n_estimators=500)

for model in [rf_reg,gbm_reg,xgb_reg,lgbm_reg]:
    get_model_predict(model,X_train.values,X_test.values,y_train.values,y_test.values,is_expm1=True)
