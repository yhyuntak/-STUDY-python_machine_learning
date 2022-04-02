import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
iris = load_iris()
feature_names = ['sepal_length','sepal_width','petal_length','petal_width']

iris_df = pd.DataFrame(data=iris.data,columns=feature_names)
iris_df['target'] = iris.target

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3,random_state=0).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

iris_df['gmm_cluster']=gmm_cluster_labels
# iris_df['target']=iris.target

iris_result = iris_df.groupby('target')['gmm_cluster'].value_counts()
print(iris_result)


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.6,min_samples=8,metric='euclidean')
dbscan_labels = dbscan.fit_predict(iris.data)
iris_df['dbscan_cluster']=dbscan_labels
iris_df['target']=iris.target

iris_result = iris_df.groupby('target')['dbscan_cluster'].value_counts()
print(iris_result)


####################################################3333
