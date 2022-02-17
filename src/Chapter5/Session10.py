import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

house_df_org = pd.read_csv("../../dataset/house-prices-advanced-regression-techniques/train.csv")
house_df = house_df_org.copy()
print(house_df.head(3))
print(house_df.info())
print(house_df.dtypes.value_counts())
isnull_series = house_df.isnull().sum()
print(isnull_series[isnull_series>0].sort_values(ascending=False))

# house_df['SalePrice'].hist()

log_sale_price = np.log1p(house_df['SalePrice'])
# sns.distplot(log_sale_price)
# plt.show()
#

drop_columns = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Id']

house_df.drop(drop_columns,axis=1,inplace=True)
house_df.fillna(house_df.mean(),inplace=True)
null_column_count = house_df.isnull().sum()[house_df.isnull().sum()>0]
print(house_df.dtypes[null_column_count.index])

house_df_ohe = pd.get_dummies(house_df)

from sklearn.metrics import mean_squared_error
def get_rmse(model,X_test,y_test):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test,pred)
    rmse = np.sqrt(mse)
    print(model.__class__.__name__,'로그 변환된 RMSE:',np.round(rmse,3))
    return rmse

def get_rmses(models,X_test,y_test):
    rmses = []
    for model in models:
        rmse = get_rmse(model,X_test,y_test)
        rmses.append(rmse)
    return rmses

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split

y_target = np.log1p(house_df['SalePrice'])
X_features = house_df_ohe.drop('SalePrice',axis=1,inplace=False)
X_train,X_test,y_train,y_test = train_test_split(X_features,y_target,test_size=0.2,random_state=156)

lr_reg = LinearRegression()
lr_reg.fit(X_train,y_train)
ridge_reg = Ridge()
ridge_reg.fit(X_train,y_train)
lasso_reg = Lasso()
lasso_reg.fit(X_train,y_train)

models = [lr_reg,ridge_reg,lasso_reg]
get_rmses(models,X_test,y_test)

from sklearn.model_selection import cross_val_score

def get_avg_rmse_cv(models):
    for model in models :
        rmse_list = np.sqrt(-cross_val_score(model,X_features,y_target,scoring="neg_mean_squared_error",cv=5))
        rmse_avg = np.mean(rmse_list)
        print("{0} CV RMSE 값 리스트 : {1}".format(model.__class__.__name__, np.round(rmse_list, 3)))
        print("{0} CV 평균 RMSE 값 : {1}".format(model.__class__.__name__, np.round(rmse_avg, 3)))

get_avg_rmse_cv(models)

from sklearn.model_selection import GridSearchCV
def print_best_params(model,params,X_features,y_target):
    grid_model = GridSearchCV(model,param_grid=params,scoring="neg_mean_squared_error",cv=5)
    grid_model.fit(X_features,y_target)
    rmse = np.sqrt(-1*grid_model.best_score_)
    print("{0} 5 CV 시 최적 평균 RMSE 값 : {1}, 최적 alpha : {2}".format(model.__class__.__name__,np.round(rmse,3),grid_model.best_params_))

ridge_params = {'alpha':[0.05,0.1,1,5,8,10,12,15,20]}
lasso_params = {'alpha':[0.001,0.005,0.008,0.05,0.03,0.1,0.5,1,5,10]}
print_best_params(ridge_reg,ridge_params,X_features,y_target)
print_best_params(lasso_reg,lasso_params,X_features,y_target)

lr_reg = LinearRegression()
lr_reg.fit(X_train,y_train)
ridge_reg = Ridge(alpha=12)
ridge_reg.fit(X_train,y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train,y_train)

models = [lr_reg,ridge_reg,lasso_reg]
get_rmses(models,X_test,y_test)

from scipy.stats import skew

features_index = house_df.dtypes[house_df.dtypes != 'object'].index
skew_features = house_df[features_index].apply(lambda x : skew(x))
skew_features_top = skew_features[skew_features>1]
print(skew_features_top.sort_values(ascending=False))


house_df[skew_features_top.index] = np.log1p(house_df[skew_features_top.index])

house_df_ohe = pd.get_dummies(house_df)
y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis=1,inplace=False)
X_train,X_test,y_train,y_test = train_test_split(X_features,y_target,test_size=0.2)
print_best_params(ridge_reg,ridge_params,X_features,y_target)
print_best_params(lasso_reg,lasso_params,X_features,y_target)

# plt.scatter(x=house_df_org['GrLivArea'],y=house_df_org['SalePrice'])
# plt.ylabel('SalePrice',fontsize=15)
# plt.xlabel('GrLivArea',fontsize=15)
# plt.show()

cond1 = house_df_ohe['GrLivArea'] > np.log1p(4000)
cond2 = house_df_ohe['SalePrice'] < np.log1p(500000)
outlier_index = house_df_ohe[cond1&cond2].index
house_df_ohe.drop(outlier_index,axis=0,inplace=True)


y_target = house_df_ohe['SalePrice']
X_features = house_df_ohe.drop('SalePrice',axis=1,inplace=False)
X_train,X_test,y_train,y_test = train_test_split(X_features,y_target,test_size=0.2)
print_best_params(ridge_reg,ridge_params,X_features,y_target)
print_best_params(lasso_reg,lasso_params,X_features,y_target)

# skew_features = house_df[features_index].apply(lambda x : skew(x))
# skew_features_top = skew_features[skew_features>1]
# print(skew_features_top.sort_values(ascending=False))

def get_rmse_pred(preds):
    for key in preds.keys():
        pred_value = preds[key]
        mse = mean_squared_error(y_test,pred_value)
        rmse = np.sqrt(mse)
        print("{0} 모델의 RMSE : {1}".format(key,rmse))
ridge_reg = Ridge(alpha=8)
ridge_reg.fit(X_train,y_train)
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(X_train,y_train)

ridge_pred = ridge_reg.predict(X_test)
lasso_pred = lasso_reg.predict(X_test)

pred = 0.6*ridge_pred+0.4*lasso_pred
preds = {'최종 혼합':pred,
         'Ridge':ridge_pred,
         'Lasso':lasso_pred}

get_rmse_pred(preds)
