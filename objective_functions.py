import numpy as np

bird_function_bounds = [(-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi)]


def bird_function(X):
    """Bird function"""
    X = np.atleast_2d(X)

    x = X[:, 0]
    y = X[:, 1]
    F = np.sin(y) * np.exp((1-np.cos(x)) ** 2) \
        + np.cos(x) * np.exp((1-np.sin(y)) ** 2) \
        + (x-y) ** 2
    return -F.reshape((-1, 1))
