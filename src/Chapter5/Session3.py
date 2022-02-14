import numpy as np
import matplotlib.pyplot as plt

X = 2*np.random.rand(100,1)
y = 6+4*X+np.random.randn(100,1)

# plt.scatter(X,y)
# plt.show()

def get_cost(y,y_pred):
    N= len(y)
    cost = np.sum(np.square(y-y_pred))/N
    return cost

def get_weight_updates(w1,w0,X,y,learning_rate=0.01):
    N=len(y)
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    y_pred = np.dot(X,w1.T)+w0
    diff = y-y_pred

    w0_factors=np.ones((N,1))

    w1_update = -(2/N)*learning_rate*(np.dot(X.T,diff))
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T,diff))

    return w1_update,w0_update

def gradient_desent_steps(X,y,iters=10000):
    w0=np.zeros((1,1))
    w1=np.zeros((1,1))

    for ind in range(iters):
        w1_update,w0_update = get_weight_updates(w1,w0,X,y)
        w1=w1-w1_update
        w0=w0-w0_update

    return w1,w0

w1,w0=gradient_desent_steps(X,y,iters=1000)
print("GD w1:{0:.3f} w0:{1:.3f}".format(w1[0,0],w0[0,0]))
y_pred = w1[0,0]*X+w0
print("Gradient descent cost : {0:.4f}",get_cost(y,y_pred))
# plt.scatter(X,y)
# plt.plot(X,y_pred)
# plt.show()

def stochastic_gradient_descent_Steps(X,y,batch_size=10,iters=1000):
    w0=np.zeros((1,1))
    w1=np.zeros((1,1))
    prev_cost = 100000
    iter_index = 0

    for ind in range(iters):
        np.random.seed(ind)
        stocastic_random_index = np.random.permutation(X.shape[0])
        sample_X=X[stocastic_random_index[0:batch_size]]
        sample_y=y[stocastic_random_index[0:batch_size]]
        w1_update,w0_update = get_weight_updates(w1,w0,sample_X,sample_y)
        w1=w1-w1_update
        w0=w0-w0_update

    return w1,w0

w1,w0 = stochastic_gradient_descent_Steps(X,y,iters=1000)
print("SGD w1:{0:.3f} w0:{1:.3f}".format(w1[0,0],w0[0,0]))
y_pred = w1[0,0]*X+w0
print("SGD cost :{0:.4f}".format(get_cost(y,y_pred)))

