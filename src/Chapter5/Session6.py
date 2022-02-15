from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston

boston = load_boston()
boston_df = pd.DataFrame(data=boston.data,columns=boston.feature_names)

boston_df['PRICE'] = boston.target
print(boston_df.head(3))

X_data = boston_df.drop(['PRICE'],axis=1,inplace=False)
y_target = boston.target

alphas = [0,0.1,1,10,100]

for alpha in alphas :

    ridge = Ridge(alpha=alpha)

    neg_mse_scores = cross_val_score(ridge,X_data,y_target,scoring="neg_mean_squared_error",cv=5)
    rmse_scores=np.sqrt(-1*neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print("-"*20)
    print("alpha : ",alpha)
    print("5 folds의 개별 Negative MSE scores: ",np.round(neg_mse_scores,2))
    print("5 folds의 개별 RMSE scores: ",np.round(rmse_scores,2))
    print("5 folds의 평균 RMSE: ",avg_rmse)
    print("-"*20)

# fig,axs = plt.subplots(figsize=(18,6),nrows=1,ncols=5)
# coeff_df=pd.DataFrame()
#
# for pos,alpha in enumerate(alphas):
#     ridge = Ridge(alpha=alpha)
#     ridge.fit(X_data,y_target)
#     coeff = pd.Series(data=ridge.coef_,index=X_data.columns)
#     colname = 'alpha:'+str(alpha)
#     coeff_df[colname]=coeff
#
#     coeff = coeff.sort_values(ascending=False)
#     axs[pos].set_title(colname)
#     axs[pos].set_xlim(-3,6)
#     sns.barplot(x=coeff.values,y=coeff.index,ax=axs[pos])
# plt.show()
#


from sklearn.linear_model import Lasso,ElasticNet

def get_linear_reg_eval(model_name,params=None,X_data_n=None,y_target_n=None,verbose=True,return_coeff=True):
    coeff_df = pd.DataFrame()
    if verbose : print("#"*6,model_name,"#"*6)
    for param in params:
        if model_name == "Ridge" : model = Ridge(alpha=param)
        elif model_name == "Lasso" : model = Lasso(alpha=param)
        elif model_name == "ElasticNet" : model = ElasticNet(alpha=param,l1_ratio=0.7)

        neg_mse_scores = cross_val_score(model,X_data_n,y_target_n,scoring="neg_mean_squared_error",cv=5)
        avg_rmse = np.mean(np.sqrt(-1*neg_mse_scores))
        print("alpha {0}일 때 5 폴드 세트의 평균 RMSE : {1:.3f}".format(param,avg_rmse))
        model.fit(X_data_n,y_target_n)
        if return_coeff:
            coeff = pd.Series(data=model.coef_)
            colname='alpha'+str(param)
            coeff_df[colname]=coeff
    return coeff_df

lasso_alphas = [0.07,0.1,0.5,1,3]
coeff_lasso_df = get_linear_reg_eval("Lasso",params=lasso_alphas,X_data_n=X_data,y_target_n=y_target)

sort_column = 'alpha'+str(lasso_alphas[0])
coeff_lasso_df.sort_values(by=sort_column,ascending=False,inplace=True)
print(coeff_lasso_df)
#

elastic_alphas = [0.07,0.1,0.5,1,3]
coeff_elastic_df = get_linear_reg_eval("ElasticNet",params=elastic_alphas,X_data_n=X_data,y_target_n=y_target)
sort_column = 'alpha'+str(elastic_alphas[0])
coeff_elastic_df.sort_values(by=sort_column,ascending=False,inplace=True)
print(coeff_elastic_df)

print("#!#!# page 330 #!#!#")
print("#!#!# page 330 #!#!#")
print("#!#!# page 330 #!#!#")
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures

def get_scaled_data(method='None',p_degree=None,input_data=None):
    if method == 'Standard' : scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax' : scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log' : scaled_data = np.log1p(input_data)
    else : scaled_data = input_data
    if p_degree != None : scaled_data = PolynomialFeatures(degree=p_degree,include_bias=False).fit_transform(scaled_data)

    return scaled_data

ridge_alphas = [0.1,1,10,100]
scale_methods = [('Standard',None),('Standard',2),('MinMax',None),('MinMax',2),('Log',None)]

for scaled_method in scale_methods :
    X_data_scaled = get_scaled_data(method=scaled_method[0],p_degree=scaled_method[1],input_data=X_data)
    coeff_ridge_df = get_linear_reg_eval('Ridge',params=ridge_alphas,X_data_n=X_data_scaled,y_target_n=y_target)

    # print(coeff_ridge_df.head())

