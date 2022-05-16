import numpy as np

# Langermann's function.
LANGERMANN_FUNCTION_BOUNDS = [(3, 5), (3, 5)]
LANGERMANN_FUNCTION_THRESHOLD = -0.5
LANGERMANN_FUNCTION_NAME = "langermann_function"

# Test functions for optimization needs - Marcin Molga, Czesław Smutnicki
def langermann_function(X):
    """Langermann's function.
    The Langermann function is a multimodal test function. The local minima are unevenly distributed.
    Max find by observation, f*=1.8 at (x,y)=(3.3, 3.06)
    """
    m = 5
    a = [3, 5, 2, 1, 7]
    b = [5, 2, 1, 4, 9]
    c = [1, 2, 5, 2, 3]

    X = np.atleast_2d(X)
    x = X[:, 0]
    y = X[:, 1]
    noise = np.random.normal(0, 0.25 ** 2)
    F = np.sum(
        [
            c[i]
            * np.exp(-1 / np.pi * ((x - a[i]) ** 2 + (y - b[i]) ** 2))
            * np.cos(np.pi * ((x - a[i]) ** 2 + (y - b[i]) ** 2))
            for i in range(m)
        ],
        axis=0,
    )
    F += noise
    return F.reshape((-1, 1))
