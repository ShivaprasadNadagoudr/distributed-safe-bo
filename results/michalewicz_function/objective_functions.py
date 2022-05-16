import numpy as np

# Michalewicz's function
MICHALEWICZ_FUNCTION_BOUNDS = [
    (0, np.pi),
    (0, np.pi),
    (0, np.pi),
    (0, np.pi),
    (0, np.pi),
]
MICHALEWICZ_FUNCTION_THRESHOLD = 0.8  # mean is 0.46
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
    m = 5
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
