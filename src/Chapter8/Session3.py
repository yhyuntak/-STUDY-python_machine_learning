import numpy as np
from scipy import sparse

data = np.array([3,1,2])
row_pos = np.array([0,0,1])
col_pos = np.array([0,2,1])
sparse_coo = sparse.coo_matrix((data,(row_pos,col_pos)))

print(sparse_coo.toarray())

dense3 = np.array([[0,0,1,0,0,5],[1,4,0,3,2,5],[0,6,0,3,0,0],[2,0,0,0,0,0]])
coo = sparse.coo_matrix(dense3)
csr = sparse.csr_matrix(dense3)
print("coo : ",coo)
print("csr : ",csr)
