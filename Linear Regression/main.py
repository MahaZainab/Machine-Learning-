from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from models import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = datasets.make_regression(
        n_samples=500, n_features=1, noise=15, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    plt.grid()
    plt.title('Linear Regression Model')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.scatter(X[:, 1], y)
    plt.show()
    # print(X.shape, y.shape)

    regressor = LinearRegression(n_iters=1000)
    plt.grid()
    plt.title('Linear Regression Model')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.scatter(X[:, 1], y)
    plt.plot(X, LinearRegression(n_iters=0).fit(X, y).predict(
        X), linewidth=2, color='black', label='prediction')
    plt.legend()
    plt.show()

    regressor.fit(X_train, y_train)

    predictions = regressor.predict(X_test)
    print(f'MSE: {mean_squared_error(predictions, y_test)}')

    # print(X.shape, y.shape)

    plt.grid()
    plt.title('Linear Regression Model')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.scatter(X[:, 1], y)
    plt.plot(X, regressor.predict(X), linewidth=2,
             color='black', label='prediction')
    plt.legend()
    # plt.grid()
    plt.show()
