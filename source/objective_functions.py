import numpy as np
from typing import Final

# domain of the function
BIRD_FUNCTION_BOUNDS: Final = [(-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi)]
# user defined threshold for function
BIRD_FUNCTION_THRESHOLD: Final = -35.0
BIRD_FUNCTION_NAME: Final = "bird_function"

# Test functions for optimization needs - Marcin Molga, Czesław Smutnicki
def bird_function(X):
    """Bird function.
    -f* = 106.764537 at
    (x, y) = (4.70104, 3.15294) and
    (x, y) = (-1.58214, -3.13024)
    """
    X = np.atleast_2d(X)
    x = X[:, 0]
    y = X[:, 1]
    noise = np.random.normal(0, 0.25 ** 2)
    F = (
        np.sin(y) * np.exp((1 - np.cos(x)) ** 2)
        + np.cos(x) * np.exp((1 - np.sin(y)) ** 2)
        + (x - y) ** 2
        + noise
    )
    # optimization involves finding maximum value of function
    return -F.reshape((-1, 1))


# Langermann's function.
LANGERMANN_FUNCTION_BOUNDS: Final = [(3, 5), (3, 5)]
LANGERMANN_FUNCTION_THRESHOLD: Final = -0.3
LANGERMANN_FUNCTION_NAME: Final = "langermann_function"

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


# Levy05 function
LEVY05_FUNCTION_BOUNDS = [(-2, 2), (-2, 2)]
LEVY05_FUNCTION_THRESHOLD = -20.0
LEVY05_FUNCTION_NAME = "levy05_function"

# http://infinity77.net/global_optimization/test_functions_nd_L.html
def levy05_function(X):
    """Levy05 function.
    This is a multimodal optimization problem.
    -f*=176.1375 at (x, y)=(-1.3068, -1.4248)
    """
    X = np.atleast_2d(X)
    x = X[:, 0]
    y = X[:, 1]
    noise = np.random.normal(0, 0.25 ** 2)
    F = (
        np.sum([i * np.cos((i - 1) * x + i) for i in range(1, 6)], axis=0)
        * np.sum([j * np.cos((j + 1) * y + j) for j in range(1, 6)], axis=0)
        + (x + 1.42513) ** 2
        + (y + 0.80032) ** 2
    )
    F += noise
    return -F.reshape((-1, 1))


# Michalewicz's function
MICHALEWICZ_FUNCTION_BOUNDS = [
    (0, np.pi),
    (0, np.pi),
    (0, np.pi),
    (0, np.pi),
    (0, np.pi),
]
MICHALEWICZ_FUNCTION_THRESHOLD = 0.001  # mean is 0.46
MICHALEWICZ_FUNCTION_NAME = "michalewicz_function"


def michalewicz_function(X):
    """
    The Michalewicz function is a multimodal test function (owns n! local optima).
    The parameter m defines the “steepness” of the valleys or edges. Larger m leads
    to more difficult search. For very large m the function behaves like a needle in
    the haystack (the function values for points in the space outside the narrow peaks
    give very little information on the location of the global optimum).

    It is usually set m = 10. Test area is usually restricted to hyphercube 0 <= x_i <= pi,
    i = 1, ..., n. The global minimum value has been approximated by f(x)=-4.687 for n=5
    and by f(x)=-9.66 for n=10.

    Respective optimal solutions are not given.
    """
    X = np.atleast_2d(X)
    m = 10
    n = 5
    noise = np.random.normal(0, 0.05 ** 2)
    y = []
    for x in X:
        i = np.arange(1, n + 1)
        F = np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m))
        # for i in range(n):
        #     F += np.sin(x[i]) * ((np.sin(i * x[i] ** 2 / np.pi)) ** (2 * m))
        F += noise
        y.append(F)

    y = np.array(y)
    return y.reshape((-1, 1))
