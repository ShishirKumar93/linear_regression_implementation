import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def normalize(X): # creating standard variables here (u-x)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / s

def MSE(X,y,B,lmbda):
    return np.sum(np.subtract(y , np.matmul(X,B))**2)

def loss_gradient(X, y, B, lmbda):
    return -1 * np.matmul(X.T, np.subtract(y, np.matmul(X,B)))

def loss_ridge(X, y, B, lmbda):
    return np.sum(np.square(np.subtract(y ,np.matmul(X,B)))) + lmbda * np.sum(np.multiply(B,B))

def loss_gradient_ridge(X, y, B, lmbda):
    return -1 * np.matmul(X.T, y - np.matmul(X,B)) + lmbda * B

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, B,lmbda):
    x_b = np.matmul(X,B)
    return -1 * np.sum(np.subtract(np.matmul(y.T,x_b) , np.log(np.add(1 ,np.exp(x_b)))))

def log_likelihood_gradient(X, y, B, lmbda):
    return -1 * np.matmul(X.T, np.subtract(y , sigmoid(np.matmul(X,B))))

# To be implemented
def L1_log_likelihood(X, y, B, lmbda):
    pass

# To be implemented
def L1_log_likelihood_gradient(X, y, B, lmbda):
    pass

def minimize(X, y, loss, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=0.00000001):
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")
    if addB0:
        B0 = np.ones(shape=(n, 1))
        x = np.hstack([B0, X])
        B = np.random.random_sample(size=(p+1, 1)) * 2 - 1
    if not addB0:
        x = X
        B = np.random.random_sample(size=(p, 1)) * 2 - 1
    step = np.zeros(B.shape)
    eps = 1e-5 # prevent division by 0
    iters = 0
    loss_now = loss(x,y,B,lmbda)
    while True:
        iters = iters + 1
        prev_B = B
        loss_prev = loss_now
        cost = loss_gradient(x, y, B, lmbda)
        step = np.add(step,np.multiply(cost,cost))
        B = np.subtract(prev_B, (eta * cost / (np.sqrt(step) + eps)))
        loss_now = loss(x,y,B,lmbda)
        if np.abs(loss_now - loss_prev) < precision or iters > max_iter:  break
    if addB0: return B
    if not addB0: return  np.vstack([y.mean(), B])

class LinearRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                           MSE,
                           loss_gradient,
                           self.eta,
                           self.lmbda,
                           self.max_iter)

class RidgeRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=80,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                           loss_ridge,
                           loss_gradient_ridge,
                           self.eta,
                           self.lmbda,
                           self.max_iter,
                           addB0=False)

class LogisticRegression621:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        sig = sigmoid(np.dot(X, self.B))
        return np.where(sig<0.5,0,1)
    
    def predict_proba(self,X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                           log_likelihood,
                           log_likelihood_gradient,
                           self.eta,
                           self.lmbda,
                           self.max_iter)


# To be implemented
class LassoLogistic621:
    pass