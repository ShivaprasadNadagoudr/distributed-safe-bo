import GPy
import numpy as np
import pandas as pd
import safeopt
import time
import objective_functions as obj_fun

# Measurement noise
noise_var = 0.5 ** 2

# Bounds on the inputs variable
bounds = obj_fun.LEVY05_FUNCTION_BOUNDS
safe_threshold = obj_fun.LEVY05_FUNCTION_THRESHOLD
parameter_set = safeopt.linearly_spaced_combinations(bounds, 100)

# Define Kernel
kernel = GPy.kern.RBF(
    input_dim=len(bounds), variance=1.0, lengthscale=[0.5, 0.25], ARD=True
)
# kernel = GPy.kern.Matern52(input_dim=len(bounds), variance=2, ARD=True)

# Initial safe set
x0 = np.array([[-0.8, -1.0]])  # 2D single safe point
# x0 = np.array([[-1.5, -1.03]])  # 2D single safe point -- this is near to optima

# noisy true function
objective_function = obj_fun.levy05_function

# The statistical model of our objective function
gp = GPy.models.GPRegression(x0, objective_function(x0), kernel, noise_var=noise_var)

start_time = time.time()
# The optimization routine
# opt = safeopt.SafeOptSwarm(gp, safe_threshold, bounds=bounds, threshold=0.2)
opt = safeopt.SafeOpt(gp, parameter_set, safe_threshold, threshold=0)

for i in range(50):
    print("iteration", i)
    # Obtain next query point
    x_next = opt.optimize()
    # print(x_next)
    # Get a measurement from the real system
    y_meas = objective_function(x_next)
    # Add this to the GP model
    opt.add_new_data_point(x_next, y_meas)

total_run_time = time.time() - start_time
X = opt.x
Y = opt.y
max_val = Y.max()
idx = Y.argmax()
max_point = X[idx]

objective_function_name = obj_fun.LEVY05_FUNCTION_NAME

arr = []
for x, y in zip(X, Y):
    arr.append(np.append(x, y))

arr = np.array(arr)
dimension = X.shape[1]
label = []
label.extend(["x" + str(i) for i in range(dimension)])
label.append("y")

points_df = df = pd.DataFrame(arr, columns=label)
time_str = time.strftime("%d_%b_%H_%M_%S_", time.localtime())
points_df.to_csv(
    "./results-data/" + objective_function_name + "_sbo_" + time_str + "log.csv",
    index=False,
)
report = "Total run time : %f\n" % total_run_time
no_points_evaluated = points_df.shape[0]
report += "Number of points evaluated : %d\n" % no_points_evaluated
no_unsafe_evaluation = points_df.y[points_df.y < safe_threshold].count()
report += "Number of unsafe evaluations : %d\n" % no_unsafe_evaluation
optimum_value = points_df["y"].max()
optimum_value_at = points_df.iloc[points_df["y"].idxmax()][0 : points_df.shape[1] - 1]
report += "Optimization results\ny = %f\nat\n%s" % (
    optimum_value,
    optimum_value_at.to_string(),
)
with open(
    "./results-data/" + objective_function_name + "_sbo_" + time_str + "log.txt", "w",
) as res_file:
    res_file.write(report)

print(report)

