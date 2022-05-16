import GPy
import numpy as np
import pandas as pd
import safeopt
import time
from sklearn.preprocessing import MinMaxScaler

# GP_regression noise
noise_var = 0.001
noise_var_to_sample_function = 0.000001  # keep to low upto 10^-5

# Bounds on the inputs variable
bounds = [(-10.0, 10.0), (-10.0, 10.0)]
safe_threshold = 0.5
parameter_set = safeopt.linearly_spaced_combinations(bounds, 100)
# Define Kernel
kernel = GPy.kern.RBF(input_dim=len(bounds), variance=1.0, lengthscale=1.0, ARD=True)


def sample_safe_fun():
    i = 0
    while True:
        fun = safeopt.sample_gp_function(
            kernel, bounds, noise_var_to_sample_function, 50
        )
        y_val = fun([0.0, 0.0], noise=False)
        print("sample_function", i, y_val)
        i += 1
        if y_val > safe_threshold:
            break
    return fun


# Initial safe set
x0 = np.zeros((1, len(bounds)))  # safe point at zero
# x0 = np.array([[-2.5]]) # 1D single safe point
# x0 = np.array([[-2.5, 3.4]]) # 2D single safe point
# x0 = np.array([[-2.5], [6.5]]) # 1D multiple safe points

scaled_data = []
unsafe_evals = []

# for no of functions
for j in range(200):
    print("function", j)
    # noisy true function
    objective_function = sample_safe_fun()

    # The statistical model of our objective function
    # gp = GPy.models.GPRegression(x0, objective_function(x0), kernel, noise_var=noise_var)
    gp = GPy.models.GPRegression(
        x0, objective_function(x0), kernel, noise_var=noise_var
    )

    # The optimization routine
    # opt = safeopt.SafeOptSwarm(gp, safe_threshold, bounds=bounds, threshold=0.2)
    opt = safeopt.SafeOpt(
        gp, parameter_set, safe_threshold, lipschitz=None, threshold=-np.inf
    )

    try:
        for i in range(99):
            # print("iteration", i)
            # Obtain next query point
            x_next = opt.optimize()
            # print(x_next)
            # Get a measurement from the real system
            y_meas = objective_function(x_next)
            # Add this to the GP model
            opt.add_new_data_point(x_next, y_meas)
    except Exception as e:
        print(e)
        continue

    y_val = objective_function(parameter_set)
    scaler = MinMaxScaler()
    scaler.fit(y_val)
    scaled_data.append(pd.Series(np.squeeze(scaler.transform(opt.y))))
    unsafe_evals.append(
        pd.Series(np.squeeze(opt.y)).apply(lambda x: 1 if x < safe_threshold else 0)
    )

scaled_data_df = pd.concat(scaled_data, axis=1)
unsafe_evals_df = pd.concat(unsafe_evals, axis=1)

scaled_data_df.to_csv(
    "./results/gp_sample_function/"
    + time.strftime("%d_%b_%Y_%H_%M_%S_", time.localtime())
    + "gp_sample_sbo_achieved_max.csv",
    index=False,
)

unsafe_evals_df.to_csv(
    "./results/gp_sample_function/"
    + time.strftime("%d_%b_%Y_%H_%M_%S_", time.localtime())
    + "gp_sample_sbo_cumu_unsafe_evals.csv",
    index=False,
)
