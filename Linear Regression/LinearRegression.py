# from sklearn.metrics import mean_squared_error
import numpy as np


class LinearRegression:
    def __init__(self, lr: int = 0.01, n_iters: int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape     # X shape [N, f]
        self.weights = np.random.rand(num_features)  # W shape [f, 1]
        self.bias = 0

        for i in range(self.n_iters):

            # y_pred shape should be N, 1
            y_pred = np.dot(X, self.weights) + self.bias

            # if i % 100 == 0:
            #     print(f"Iter: {i} || MSE: {mean_squared_error(y, y_pred)}")

            diff = y_pred - y

            # X -> [N,f]
            # y_pred -> [N]
            # dw -> [f]
            dw = (1 / num_samples) * np.dot(X.T, y_pred - y)
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
