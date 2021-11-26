
import GPy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import safeopt
from typing import List, Tuple

from hyperspaces import create_hyperspaces
from objective_functions import bird_function, bird_function_bounds

if __name__ == "__main__":
    # Measurement noise
    noise_var = 0.05 ** 2

    # Bounds on the inputs variable
    bounds = bird_function_bounds
    hyperspaces = create_hyperspaces(bounds, 2)
    parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)

    # Define Kernel
    kernel = GPy.kern.RBF(
        input_dim=len(bounds),
        variance=2.,
        lengthscale=1.0,
        ARD=True
    )

    # Initial safe set
    x0 = np.zeros((1, len(bounds)))  # safe point at zero
    # x0 = np.array([[-2.5]]) # 1D single safe point
    # x0 = np.array([[-2.5, 3.4]]) # 2D single safe point
    # x0 = np.array([[-2.5], [6.5]]) # 1D multiple safe points

    # noisy true function
    objective_function = bird_function

    """
    IF initial safe set contains more than single point.
        group points corresponding to hyperspace,
        then deploy each hyperspace into a separate process 
    """

    # The statistical model of our objective function
    gp = GPy.models.GPRegression(
        x0, objective_function(x0),
        kernel, noise_var=noise_var
    )

    # The optimization routine
    # opt = safeopt.SafeOptSwarm(gp, -np.inf, bounds=bounds, threshold=0.2)
    opt = safeopt.SafeOpt(
        gp,
        parameter_set,
        -35.,
        lipschitz=None,
        threshold=0.2
    )

    evaluation_constraint = 100
    for i in range(evaluation_constraint):
        # obtain new query point
        x_next = opt.optimize()
        # here I should check whether x_next point belongs to same hyperspace
        # or not

        # Get a measurement from the real system
        y_meas = objective_function(x_next)
        # Add this to the GP model
        opt.add_new_data_point(x_next, y_meas)

        opt.plot(100, plot_3d=False)
