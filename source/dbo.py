import GPy
import numpy as np
from hyperspaces import create_hyperspaces, initial_deploy
import objective_functions as obj_fun
from sklearn_functions import svc_crossval

if __name__ == "__main__":
    # bounds = obj_fun.LANGERMANN_FUNCTION_BOUNDS
    bounds = [(-3, 2), (-4, -1)]
    no_subspaces = 4
    bounds_indices = [[0, no_subspaces - 1] for _ in range(len(bounds))]
    # safe_threshold = obj_fun.LANGERMANN_FUNCTION_THRESHOLD
    safe_threshold = 0.90
    # parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)
    hyperspaces_list = create_hyperspaces(bounds, no_subspaces)

    # Measurement noise
    noise_var = 0.05 ** 2

    # Initial safe set
    # x0 = np.zeros((1, len(bounds)))  # safe point at zero
    # x0 = np.array([[-2.5]]) # 1D single safe point
    x0 = np.array([[0.9865, -2.948]])  # 2D single safe point
    # x0 = np.array([[-2.5], [6.5]]) # 1D multiple safe points

    # Define Kernel
    kernel = GPy.kern.RBF(
        input_dim=len(bounds), variance=1.0, lengthscale=1.0, ARD=True
    )

    # true function
    # objective_function = obj_fun.langermann_function
    objective_function = svc_crossval

    initial_deploy(
        x0,
        bounds,
        bounds_indices,
        kernel,
        objective_function,
        safe_threshold,
        noise_var,
        hyperspaces_list,
        obj_fun.LANGERMANN_FUNCTION_NAME,
    )
