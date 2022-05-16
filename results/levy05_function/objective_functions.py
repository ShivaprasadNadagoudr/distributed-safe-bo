import numpy as np

# Levy05 function
LEVY05_FUNCTION_BOUNDS = [(-2, 0), (-2, 0)]
LEVY05_FUNCTION_THRESHOLD = -50.0
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
    F = -(
        np.sum([i * np.cos((i - 1) * x + i) for i in range(1, 6)], axis=0)
        * np.sum([j * np.cos((j + 1) * y + j) for j in range(1, 6)], axis=0)
        + (x + 1.42513) ** 2
        + (y + 0.80032) ** 2
    )
    F += noise
    return -F.reshape((-1, 1))
