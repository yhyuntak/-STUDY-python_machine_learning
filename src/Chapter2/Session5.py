from sklearn.preprocessing import LabelEncoder

## label encoding ##
print("-"*30)
print("## label encoding ##")
print("-"*30)

items=['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

print("인코딩 변환값:",labels)
print("인코딩 클래스:",encoder.classes_)
print("디코딩 원본값:",encoder.inverse_transform([4,5,2,0,1,1,3,3]))

## one-hot encoding ##
print("-"*30)
print("## one-hot encoding ##")
print("-"*30)

from sklearn.preprocessing import OneHotEncoder
import numpy as np

label_enc = LabelEncoder()
label_enc.fit(items)
labels = label_enc.transform(items)
labels = labels.reshape(-1,1)

onehot_enc = OneHotEncoder()
onehot_enc.fit(labels)
onehot_labels = onehot_enc.transform(labels)
print("원핫 인코딩 결과:",onehot_labels.toarray())

import pandas as pd
df = pd.DataFrame(data=items,columns=['items'])
onehot_labels_used_pandas = pd.get_dummies(df)
print(onehot_labels_used_pandas)

## 피처 스케일링과 정규화 ##
print("-"*30)
print("## 피처 스케일링과 정규화 ##")
print("-"*30)

print("## 피처 스케일링 : StandardScaler ##")
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names)
print("feature들의 평균 값:",iris_df.mean())
print("feature들의 분산 값:",iris_df.var())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
print("feature들의 평균 값:",iris_df_scaled.mean())
print("feature들의 분산 값:",iris_df_scaled.var())

print("## 피처 스케일링 : MinMaxScaler ##")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)
iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
print("feature들의 최소 값:",iris_df_scaled.min())
print("feature들의 최대 값:",iris_df_scaled.max())