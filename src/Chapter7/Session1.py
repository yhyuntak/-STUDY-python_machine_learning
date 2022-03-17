from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris= load_iris()
iris_df=pd.DataFrame(data=iris.data,columns=['sepal_length','sepal_width','petal_length','petal_width'])

kmeans = KMeans(n_clusters=3,max_iter=300,random_state=0)
kmeans.fit(iris_df)

iris_df['target'] = iris.target
iris_df['cluster'] = kmeans.labels_
iris_result = iris_df.groupby(['target','cluster'])['sepal_length'].count()
print(iris_result)

from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)

iris_df['pca_x']=pca_transformed[:,0]
iris_df['pca_y']=pca_transformed[:,1]
marker0_index = iris_df[iris_df['cluster']==0].index
marker1_index = iris_df[iris_df['cluster']==1].index
marker2_index = iris_df[iris_df['cluster']==2].index
# plt.scatter(x=iris_df.loc[marker0_index,'pca_x'],y=iris_df.loc[marker0_index,'pca_y'],marker='o')
# plt.scatter(x=iris_df.loc[marker1_index,'pca_x'],y=iris_df.loc[marker1_index,'pca_y'],marker='^')
# plt.scatter(x=iris_df.loc[marker2_index,'pca_x'],y=iris_df.loc[marker2_index,'pca_y'],marker='s')
# plt.show()

from sklearn.datasets import make_blobs
blobs_features,blobs_target = make_blobs(n_samples=300,n_features=2,centers=3,cluster_std=[0.1,0.5,1],random_state=0)
blobs_df = pd.DataFrame(data=blobs_features,columns=['feature1','feature2'])
blobs_df['target']=blobs_target
kmeans = KMeans(n_clusters=3)
kmeans.fit(blobs_features)
blobs_df['kmeans'] = kmeans.labels_

marker0_index = blobs_df[blobs_df['kmeans']==0].index
marker1_index = blobs_df[blobs_df['kmeans']==1].index
marker2_index = blobs_df[blobs_df['kmeans']==2].index

print(blobs_df.loc[marker0_index,'kmeans'])
# plt.scatter(x=blobs_df.loc[marker0_index,'feature1'],y=blobs_df.loc[marker0_index,'feature2'],marker='o')
# plt.scatter(x=blobs_df.loc[marker1_index,'feature1'],y=blobs_df.loc[marker1_index,'feature2'],marker='s')
# plt.scatter(x=blobs_df.loc[marker2_index,'feature1'],y=blobs_df.loc[marker2_index,'feature2'],marker='^')
# plt.show()

from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score,silhouette_samples

score_samples = silhouette_samples(iris.data,iris_df['cluster'])
print('silhouette_samples() return 값의 shape',score_samples.shape)
iris_df['silhouette_coeff']=score_samples
average_Score = silhouette_score(iris.data,iris_df['cluster'])
print('붓꽃 데이터 세트 silhouette analisis score : {0:.3f}'.format(average_Score))
print(iris_df.head(10))

print(iris_df.groupby('cluster')['silhouette_coeff'].mean())


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift

X,y = make_blobs(n_samples=200,n_features=2,centers=3,cluster_std=0.7,random_state=0)
meanshift = MeanShift(bandwidth=1)
cluster_labels = meanshift.fit_predict(X)
print("cluster_labels 유형 :",np.unique(cluster_labels))

from sklearn.cluster import estimate_bandwidth
bandwidth = estimate_bandwidth(X)
print("bandwidth 값:",round(bandwidth,3))

cluster_df = pd.DataFrame(data=X,columns=['ftr1','ftr2'])
cluster_df['target'] = y
best_bandwidth = estimate_bandwidth(X)
meanshift = MeanShift(bandwidth=best_bandwidth)
cluster_labels = meanshift.fit_predict(X)
print("cluster_labels 유형 :",np.unique(cluster_labels))
cluster_df['meanshift_label'] = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ['o','s','^','x','*']
for label in unique_labels :
    label_cluster = cluster_df[cluster_df['meanshift_label']==label]
    center_x_y = centers[label]

    plt.scatter(x=label_cluster['ftr1'],y=label_cluster['ftr2'],marker=markers[label])

    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color='gray', alpha=0.9, marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', marker='$%d$' % label )
plt.show()


