import GPy
import numpy as np
from hyperspaces import create_hyperspaces, initial_deploy
import sklearn_functions as skf
import safeopt
import time

if __name__ == "__main__":
    bounds = skf.RFC_BOUNDS
    no_subspaces = 4
    bounds_indices = [[0, no_subspaces - 1] for _ in range(len(bounds))]
    safe_threshold = skf.RFC_THRESHOLD
    parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)
    hyperspaces_list = create_hyperspaces(bounds, no_subspaces)

    print("Here")
    # exit()
    # Measurement noise
    noise_var = 0.05 ** 2

    # Initial safe set
    # x0 = np.zeros((1, len(bounds)))  # safe point at zero
    # x0 = np.array([[-2.5]]) # 1D single safe point
    x0 = np.array([[98.86, 18.39, 0.7144]])  # 2D single safe point
    # x0 = np.array([[-2.5], [6.5]]) # 1D multiple safe points

    # Define Kernel
    kernel = GPy.kern.RBF(
        input_dim=len(bounds), variance=1.0, lengthscale=1.0, ARD=True
    )

    # true function
    # objective_function = obj_fun.langermann_function
    objective_function = skf.rfc_crossval

    # initial_deploy(
    #     x0,
    #     bounds,
    #     bounds_indices,
    #     kernel,
    #     objective_function,
    #     safe_threshold,
    #     noise_var,
    #     hyperspaces_list,
    #     skf.RFC_NAME,
    # )

    start_time = time.time()
    # The statistical model of our objective function
    gp = GPy.models.GPRegression(
        x0, objective_function(x0), kernel, noise_var=noise_var
    )

    # The optimization routine
    opt = safeopt.SafeOptSwarm(gp, safe_threshold, bounds=bounds, threshold=0.2)
    # opt = safeopt.SafeOpt(
    #     gp, parameter_set, safe_threshold, lipschitz=None, threshold=0.2
    # )

    for i in range(2):
        # Obtain next query point
        x_next = opt.optimize()
        # Get a measurement from the real system
        y_meas = objective_function(x_next)
        # Add this to the GP model
        opt.add_new_data_point(x_next, y_meas)

    X = opt.x
    Y = opt.y
    max_val = Y.max()
    idx = Y.idxmax()
    max_point = X[idx]

    print(X)
    print(Y)
    print(max_val)
    print(idx)
    print(max_point)
