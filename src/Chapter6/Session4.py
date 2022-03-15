import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd

np.random.seed(121)
matrix = np.random.random((6,6))
print('원본 행렬:\n',matrix)
U,Sigma,Vt = svd(matrix,full_matrices=False)
print('\nSVD 행렬 차원:',U.shape,Sigma.shape,Vt.shape)
print('\nSigma 값 행렬:',np.diag(Sigma))

num_componets = 4
U_tr, Sigma_tr, Vt_tr = svds(matrix,k=num_componets)
print("Truncated SVD 행렬 차원",U_tr.shape,Sigma_tr.shape,Vt_tr.shape)
print('Truncated SVD Sigma 값 행렬:',np.diag(Sigma_tr))
matrix_tr = np.dot(np.dot(U_tr,np.diag(Sigma_tr)),Vt_tr)

print("Truncated SVD로 복원된 행렬:\n",matrix_tr)


from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
iris_ftrs = iris.data
tsvd = TruncatedSVD(n_components=2)
iris_tsvd = tsvd.fit_transform(iris_ftrs)
print(iris_tsvd.shape)

plt.scatter(x=iris_tsvd[:,0],y=iris_tsvd[:,1],c=iris.target)
plt.show()
