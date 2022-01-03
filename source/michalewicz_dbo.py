import GPy
import numpy as np
from hyperspaces import create_hyperspaces, initial_deploy
import objective_functions as obj_fun

if __name__ == "__main__":
    bounds = obj_fun.MICHALEWICZ_FUNCTION_BOUNDS
    no_subspaces = 2
    bounds_indices = [[0, no_subspaces - 1] for _ in range(len(bounds))]
    safe_threshold = obj_fun.MICHALEWICZ_FUNCTION_THRESHOLD
    # parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)
    hyperspaces_list = create_hyperspaces(bounds, no_subspaces)

    # Measurement noise
    noise_var = 0.05 ** 2

    # Initial safe set
    # x0 = np.zeros((1, len(bounds)))  # safe point at zero
    # x0 = np.array([[-2.5]]) # 1D single safe point
    # x0 = np.array([[4, 4]])  # 2D single safe point
    # x0 = np.array([[-2.5], [6.5]]) # 1D multiple safe points
    x0 = np.array([[2, 3, 1, 2, 3]])  # 5D single safe point

    # Define Kernel
    kernel = GPy.kern.RBF(
        input_dim=len(bounds), variance=2.0, lengthscale=1.0, ARD=True
    )

    # true function
    objective_function = obj_fun.michalewicz_function

    initial_deploy(
        x0,
        bounds,
        bounds_indices,
        kernel,
        objective_function,
        safe_threshold,
        noise_var,
        hyperspaces_list,
        obj_fun.MICHALEWICZ_FUNCTION_NAME,
    )
