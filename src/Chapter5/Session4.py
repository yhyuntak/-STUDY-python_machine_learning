import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_boston

boston = load_boston()
boston_df = pd.DataFrame(data=boston.data,columns=boston.feature_names)
boston_df['PRICE'] = boston.target

# print(boston_df.info())
#
# fig,axs = plt.subplots(figsize=(16,8),ncols=4,nrows=2)
# lm_features = ['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']
# for i,feature in enumerate(lm_features):
#     row = int(i/4)
#     col = i%4
#     sns.regplot(x=feature,y='PRICE',data=boston_df,ax=axs[row][col])
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y_target = boston_df['PRICE']
X_data = boston_df.drop(['PRICE'],axis=1,inplace=False)
X_train,X_test,y_train,y_test = train_test_split(X_data,y_target,test_size=0.3)
lr=LinearRegression()
lr.fit(X_train,y_train)
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test,y_preds)
rmse=np.sqrt(mse)

print("MSE : {0:.3f}, RMSE : {1:.3F}".format(mse,rmse))
print("Variance score : {0:.3f}".format(r2_score(y_test,y_preds)))

print("절편 값 : ",lr.intercept_)
print("회귀 계수 값 : ",np.round(lr.coef_,1))

coeff= pd.Series(data=lr.coef_,index=boston.feature_names)
coeff.sort_values(ascending=False,inplace=True)
print(coeff)

from sklearn.model_selection import cross_val_score
y_target = boston_df['PRICE']
X_data = boston_df.drop(['PRICE'],axis=1,inplace=False)
lr = LinearRegression()
neg_mse_scores = cross_val_score(lr,X_data,y_target,scoring="neg_mean_squared_error",cv=5)
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print("5 folds의 개별 Negative MSE scores: ",np.round(neg_mse_scores,2))
print("5 folds의 개별 RMSE scores: ",np.round(rmse_scores,2))
print("5 folds의 평균 RMSE: ",avg_rmse)

from sklearn.preprocessing import PolynomialFeatures

X=np.arange(4).reshape(2,2)
# poly = PolynomialFeatures(degree=2)
# poly.fit(X)
# poly_ftr=poly.transform(X)
# print("poly 전 : ",X)
# print("poly 후 : ",poly_ftr)

def polynomial_func(X):
    y=1+2*X[:,0]+3*X[:,0]**2+4*X[:,1]**3
    return y
y= polynomial_func(X)

poly_ftr = PolynomialFeatures(degree=3).fit_transform(X)
model = LinearRegression()
model.fit(poly_ftr,y)
# print("Polynomial 회귀 계수 \n",np.round(model.coef_,2))
# print("Polynomial 회귀 Shape :",model.coef_.shape)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def polynomial_func_2(X):
    return 1+2*X[:,0]+3*X[:,0]+4*X[:,1]**3
model = Pipeline([('poly',PolynomialFeatures(degree=3)),
                  ('linear',LinearRegression())])
X=np.arange(4).reshape(-1,2)
print(X)
y=polynomial_func_2(X)
print(y)
model = model.fit(X,y)
print("Polynomial pipeline 회귀 계수 \n",np.round(model.named_steps['linear'].coef_,2))

