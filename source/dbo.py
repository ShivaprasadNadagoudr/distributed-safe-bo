import GPy
import numpy as np
from hyperspaces import create_hyperspaces, initial_deploy
from objective_functions import (
    bird_function,
    BIRD_FUNCTION_BOUNDS,
    BIRD_FUNCTION_THRESHOLD,
)

if __name__ == "__main__":
    bounds = BIRD_FUNCTION_BOUNDS
    no_subspaces = 4
    bounds_indices = [(0, no_subspaces - 1) for _ in range(len(bounds))]
    safe_threshold = BIRD_FUNCTION_THRESHOLD
    # parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)
    hyperspaces_list = create_hyperspaces(bounds, no_subspaces)

    # Measurement noise
    noise_var = 0.25 ** 2

    # Initial safe set
    x0 = np.zeros((1, len(bounds)))  # safe point at zero
    # x0 = np.array([[-2.5]]) # 1D single safe point
    # x0 = np.array([[-2.5, 3.4]]) # 2D single safe point
    # x0 = np.array([[-2.5], [6.5]]) # 1D multiple safe points

    # Define Kernel
    kernel = GPy.kern.RBF(
        input_dim=len(bounds), variance=4.0, lengthscale=1.0, ARD=True
    )

    # true function
    objective_function = bird_function

    initial_deploy(
        x0,
        bounds,
        bounds_indices,
        kernel,
        objective_function,
        safe_threshold,
        noise_var,
        hyperspaces_list,
    )
