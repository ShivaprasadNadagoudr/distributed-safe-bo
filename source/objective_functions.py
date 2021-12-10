import numpy as np
from typing import Final

BIRD_FUNCTION_BOUNDS: Final = [
    (-2 * np.pi, 2 * np.pi),
    (-2 * np.pi, 2 * np.pi)
]
BIRD_FUNCTION_THRESHOLD: Final = -35.0


def bird_function(X):
    """Bird function"""
    X = np.atleast_2d(X)

    x = X[:, 0]
    y = X[:, 1]
    F = np.sin(y) * np.exp((1-np.cos(x)) ** 2) \
        + np.cos(x) * np.exp((1-np.sin(y)) ** 2) \
        + (x-y) ** 2
    return -F.reshape((-1, 1))
