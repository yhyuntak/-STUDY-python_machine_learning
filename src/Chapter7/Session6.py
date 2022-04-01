import pandas as pd
import datetime as dt
import math
import numpy as np
import matplotlib.pyplot as plt

retail_df = pd.read_excel(io='Online Retail.xlsx')
print(retail_df.head(3))
print(retail_df.info())

retail_df = retail_df[retail_df['Quantity']>0]
retail_df = retail_df[retail_df['UnitPrice']>0]

print(retail_df.isnull())
print(retail_df.isnull().sum())

print("*"*50)
print("column of CustomerID has lots of null data")
retail_df.dropna(axis=0,inplace=True)
print("Check the number of null data of CustomerID")
print(retail_df.isnull().sum())
print("*"*50)


print(retail_df['Country'].value_counts())

retail_df = retail_df[retail_df['Country']=='United Kingdom']

##############

retail_df['sale_amount'] = retail_df['UnitPrice']*retail_df['Quantity']
retail_df['CustomerID'] = retail_df['CustomerID'].astype(int)
print(retail_df['CustomerID'].value_counts().head(5))
print(retail_df.groupby('CustomerID'))

aggregations = {
    'InvoiceDate':'max',
    'InvoiceNo':'count',
    'sale_amount':'sum'
}
custom_df = retail_df.groupby('CustomerID').agg(aggregations)
custom_df = custom_df.rename(columns = {
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'sale_amount': 'Monetary'
})

custom_df = custom_df.reset_index()
print(custom_df)

custom_df['Recency'] = dt.datetime(2011,12,10) - custom_df['Recency']
custom_df['Recency'] = custom_df['Recency'].apply(lambda x : x.days+1)

# fig,(ax1,ax2,ax3) = plt.subplots(figsize=(12,4),nrows=1,ncols=3)
# ax1.set_title('Recency Hist')
# ax1.hist(custom_df['Recency'])
# ax2.set_title('Frequency Hist')
# ax2.hist(custom_df['Frequency'])
# ax3.set_title('Monetary Hist')
# ax3.hist(custom_df['Monetary'])
# plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples

X_features = custom_df[['Recency','Frequency','Monetary']].values
X_features_scaled = StandardScaler().fit_transform(X_features)

kmeans = KMeans(n_clusters=3,random_state=0)
labels = kmeans.fit_predict(X_features_scaled)
custom_df['cluster_label'] = labels

print("실루엣 스코어는 : {0.:3f}".format(silhouette_score(X_features_scaled,labels)))

# 왜곡을 줄이기위해 log를 취한다
custom_df['Recency_log'] = np.log1p(custom_df['Recency'])
custom_df['Frequency_log'] = np.log1p(custom_df['Frequency'])
custom_df['Monetary_log'] = np.log1p(custom_df['Monetary'])

X_features = custom_df[['Recency_log,Frequency_log,Monetary_log']].values
X_features_scaled = StandardScaler().fit_transform(X_features)
labels = kmeans.fit_predict(X_features_scaled)
custom_df['cluster_label']=labels

print("실루엣 스코어는 : {0.:3f}".format(silhouette_score(X_features_scaled,labels)))


