import GPy
import numpy as np
import pandas as pd
import safeopt
import time
import hyperspaces as hs
import objective_functions as obj_fun

# Measurement noise
noise_var = 0.05 ** 2

# Bounds on the inputs variable
bounds = obj_fun.BIRD_FUNCTION_BOUNDS
safe_threshold = obj_fun.BIRD_FUNCTION_THRESHOLD
hyperspaces = hs.create_hyperspaces(bounds, 4)
parameter_set = safeopt.linearly_spaced_combinations(bounds, 100)

# Define Kernel
kernel = GPy.kern.RBF(input_dim=len(bounds), variance=2.0, lengthscale=1.0, ARD=True)

# Initial safe set
x0 = np.zeros((1, len(bounds)))  # safe point at zero
# x0 = np.array([[-2.5]]) # 1D single safe point
# x0 = np.array([[-2.5, 3.4]]) # 2D single safe point
# x0 = np.array([[-2.5], [6.5]]) # 1D multiple safe points

# noisy true function
objective_function = obj_fun.bird_function

# The statistical model of our objective function
gp = GPy.models.GPRegression(x0, objective_function(x0), kernel, noise_var=noise_var)

start_time = time.time()
# The optimization routine
opt = safeopt.SafeOptSwarm(gp, safe_threshold, bounds=bounds, threshold=0.2)
# opt = safeopt.SafeOpt(
#     gp, parameter_set, safe_threshold, lipschitz=None, threshold=0.2
# )

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

objective_function_name = obj_fun.BIRD_FUNCTION_NAME

arr = []
for x, y in zip(X, Y):
    arr.append(np.append(x, y))

arr = np.array(arr)
dimension = X.shape[1]
label = []
label.extend(["x" + str(i) for i in range(dimension)])
label.append("y")

points_df = df = pd.DataFrame(arr, columns=label)

points_df.to_csv(
    "./results/"
    + objective_function_name
    + "_sbo"
    + time.strftime("_%d_%b_%Y_%H_%M_%S", time.localtime())
    + ".csv",
    index=False,
)
report = "Total run time : %f\n" % total_run_time
no_points_evaluated = points_df.shape[0]
report += "Number of points evaluated : %d\n" % no_points_evaluated
no_unsafe_evaluation = points_df.y[points_df.y < safe_threshold].count()
report += "Number of unsafe evaluations : %d\n" % no_unsafe_evaluation
optimum_value = points_df["y"].max()
optimum_value_at = points_df.iloc[points_df["y"].idxmax()][0 : points_df.shape[1]]
report += "Optimization results\ny = %f\nat\n%s" % (
    optimum_value,
    optimum_value_at.to_string(),
)
with open(
    "./results/"
    + objective_function_name
    + "_sbo"
    + time.strftime("_%d_%b_%Y_%H_%M_%S", time.localtime())
    + ".txt",
    "w",
) as res_file:
    res_file.write(report)
