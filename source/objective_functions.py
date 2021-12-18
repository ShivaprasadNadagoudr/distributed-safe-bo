import numpy as np
from typing import Final

# domain of the function
BIRD_FUNCTION_BOUNDS: Final = [(-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi)]

# user defined threshold for function
BIRD_FUNCTION_THRESHOLD: Final = -35.0


def bird_function(X):
    """Bird function.
    -f* = 106.764537 at
    (x, y) = (4.70104, 3.15294) and
    (x, y) = (-1.58214, -3.13024)
    """
    X = np.atleast_2d(X)

    x = X[:, 0]
    y = X[:, 1]
    noise = np.random.normal(0, 0.5 ** 2)
    F = (
        np.sin(y) * np.exp((1 - np.cos(x)) ** 2)
        + np.cos(x) * np.exp((1 - np.sin(y)) ** 2)
        + (x - y) ** 2
        + noise
    )
    # optimization involves finding maximum value of function
    return -F.reshape((-1, 1))
