from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
columns = ['sepal_length','sepal_width','petal_length','petal_width']
iris_df = pd.DataFrame(iris.data,columns=columns)
iris_df['target']=iris.target

# markers = ['^','s','o']
# for i,marker in enumerate(markers):
#     x_axis_data = iris_df[iris_df['target']==i]['sepal_length']
#     y_axis_data = iris_df[iris_df['target']==i]['sepal_width']
#     plt.scatter(x_axis_data,y_axis_data,marker=marker,label=iris.target_names[i])
#
# plt.legend()
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.show()

from sklearn.preprocessing import StandardScaler
iris_scaled = StandardScaler().fit_transform(iris_df.iloc[:,:-1])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)

pca_columns = ['pca_component_1','pca_component_2']
iris_df_pca = pd.DataFrame(iris_pca,columns=pca_columns)
iris_df_pca['target']=iris_df['target']
print(iris_df_pca.head(3))


# markers = ['^','s','o']
# for i,marker in enumerate(markers):
#     x_axis_data = iris_df_pca[iris_df_pca['target']==i]['pca_component_1']
#     y_axis_data = iris_df_pca[iris_df_pca['target']==i]['pca_component_2']
#     plt.scatter(x_axis_data,y_axis_data,marker=marker,label=iris.target_names[i])
#
# plt.legend()
# plt.xlabel('pca_component_1')
# plt.ylabel('pca_component_2')
# plt.show()

print(pca.explained_variance_ratio_)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rcf = RandomForestClassifier()
scores = cross_val_score(rcf,iris.data,iris.target,scoring='accuracy',cv=3)
print("원본 데이터 교차 검증 개별 정확도:",scores)
print("원본 데이터 평균 정확도:",np.mean(scores))

pca_scores = cross_val_score(rcf,iris_df_pca.iloc[:,:-1],iris.target,scoring='accuracy',cv=3)
print("pca 데이터 교차 검증 개별 정확도:",pca_scores)
print("pca 데이터 평균 정확도:",np.mean(pca_scores))

df = pd.read_csv('../../dataset/UCI_Credit_Card.csv')
print(df.shape)
print(df.head(3))
df.rename(columns={'PAY_0':'PAY_1','default.payment.next.month':'default'},inplace=True)
y_target = df['default']
X_features = df.drop(['ID','default'],axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

corr = X_features.corr()
# plt.figure(figsize=(14,14))
# sns.heatmap(corr,annot=True,fmt='.1g')
# plt.show()

cols_bill = ['BILL_AMT'+str(i) for i in range(1,7)]
scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(X_features[cols_bill])
pca = PCA(n_components=2)
pca.fit(df_cols_scaled)
print("PCA별 Components별 변동성:",pca.explained_variance_ratio_)

rcf = RandomForestClassifier(n_estimators=300)
scores = cross_val_score(rcf,X_features,y_target,scoring='accuracy',cv=3)

print("CV=3 인 경우의 개별 Fold 세트별 정확도 :",scores)
print("평균 정확도:{0:.4f}".format(np.mean(scores)))

