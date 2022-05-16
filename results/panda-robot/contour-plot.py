import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import GPy
from safeopt import SafeOpt, linearly_spaced_combinations

df_sbo = pd.read_csv(
    "./results/panda-robot/data/panda_robot_sbo_16_May_10_42_35_log.csv"
)
df_dbo = pd.read_csv(
    "./results/panda-robot/data/panda_robot_dbo_16_May_14_58_54_log.csv"
)
df_ovr = pd.read_csv(
    "./results/panda-robot/data/panda-robot_ovr_16_May_15_21_03_log.csv"
)


# SafeOpt
df_sbo = df_sbo.to_numpy()
x = df_sbo[:, :2]
y = df_sbo[:, 2].reshape(-1, 1)
g = df_sbo[:, 3].reshape(-1, 1)

lengthscale = 0.7
L = [lengthscale / 6, lengthscale / 6]
KERNEL_f = GPy.kern.sde_Matern32(
    input_dim=x.shape[1], lengthscale=L, ARD=True, variance=1
)
KERNEL_g = GPy.kern.sde_Matern32(
    input_dim=x.shape[1], lengthscale=L, ARD=True, variance=1
)
fun_gp = GPy.models.GPRegression(x, y, noise_var=0.1 ** 2, kernel=KERNEL_f)
cons_gp = GPy.models.GPRegression(x, g, noise_var=0.1 ** 2, kernel=KERNEL_g)

bounds = [[-1, 1], [-1, 1]]
parameter_set = linearly_spaced_combinations(bounds, num_samples=100)
opt = SafeOpt([fun_gp, cons_gp], parameter_set, fmin=[-np.inf, 0], beta=3.5)

q = np.linspace(-1, 1, 25)
r_cost = np.linspace(-1, 1, 25)
a = np.asarray(np.meshgrid(q, r_cost)).T.reshape(-1, 2)
input = a
mean, var = opt.gps[1].predict(input)  # predicting `g1` for input
std = np.sqrt(var)
l_x0 = mean - opt.beta(opt.t) * std  # lower bound of constraint function value
safe_idx = np.where(l_x0 >= 0)[0]  # taking function value g1>=0 as safe
values = np.zeros(a.shape[0])  # start by taking all values as 0 to denote unsafe
values[safe_idx] = 1  # update safe values as 1

mean, var = opt.gps[0].predict(input)  # predicting `f` for input
l_f = mean - opt.beta(opt.t) * std  # lower bound of objective function value

safe_l_f = l_f[safe_idx]  # safe function values
safe_max = np.where(l_f == safe_l_f.max())[0]  # safe maximum
optimum_params = a[safe_max, :]  # take corresponding params
optimum_params = optimum_params.squeeze()
q = np.reshape(a[:, 0], [25, 25])  # reshaping 0th (q) column to 25x25
r_cost = np.reshape(a[:, 1], [25, 25])  # reshaping 1st (r) column to 25x25
values = values.reshape([25, 25])  # reshaping safe_sate values to 25x25
colours = ["red", "green"]
fig = plt.figure(figsize=(10, 10))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
ax.set_xlabel("q")
ax.set_ylabel("r")
cs = ax.contourf(q * 6, r_cost * 3, values)
ax.scatter(
    q * 6, r_cost * 3, c=values, cmap=matplotlib.colors.ListedColormap(colours),
)
ax.scatter(
    optimum_params[0] * 6,
    optimum_params[1] * 3,
    marker="<",
    color="b",
    s=np.asarray([200]),
)
ax.set_title("SafeSet belief after 200 iterations for SafeOpt")
ax.set_ylim([-3.1, 3.1])
ax.set_xlim([-6.1, 6.1])
plt.savefig("./results/panda-robot/data/SafeOpt_SafeSet_200.pdf", format="pdf", dpi=300)

# DistrSafeOpt
df_dbo = df_dbo.to_numpy()
x = df_dbo[:, :2]
y = df_dbo[:, 2].reshape(-1, 1)
g = df_dbo[:, 3].reshape(-1, 1)

lengthscale = 0.7
L = [lengthscale / 6, lengthscale / 6]
KERNEL_f = GPy.kern.sde_Matern32(
    input_dim=x.shape[1], lengthscale=L, ARD=True, variance=1
)
KERNEL_g = GPy.kern.sde_Matern32(
    input_dim=x.shape[1], lengthscale=L, ARD=True, variance=1
)
fun_gp = GPy.models.GPRegression(x, y, noise_var=0.1 ** 2, kernel=KERNEL_f)
cons_gp = GPy.models.GPRegression(x, g, noise_var=0.1 ** 2, kernel=KERNEL_g)

bounds = [[-1, 1], [-1, 1]]
parameter_set = linearly_spaced_combinations(bounds, num_samples=100)
opt = SafeOpt([fun_gp, cons_gp], parameter_set, fmin=[-np.inf, 0], beta=3.5)

q = np.linspace(-1, 1, 25)
r_cost = np.linspace(-1, 1, 25)
a = np.asarray(np.meshgrid(q, r_cost)).T.reshape(-1, 2)
input = a
mean, var = opt.gps[1].predict(input)  # predicting `g1` for input
std = np.sqrt(var)
l_x0 = mean - opt.beta(opt.t) * std  # lower bound of constraint function value
safe_idx = np.where(l_x0 >= 0)[0]  # taking function value g1>=0 as safe
values = np.zeros(a.shape[0])  # start by taking all values as 0 to denote unsafe
values[safe_idx] = 1  # update safe values as 1

mean, var = opt.gps[0].predict(input)  # predicting `f` for input
l_f = mean - opt.beta(opt.t) * std  # lower bound of objective function value

safe_l_f = l_f[safe_idx]  # safe function values
safe_max = np.where(l_f == safe_l_f.max())[0]  # safe maximum
optimum_params = a[safe_max, :]  # take corresponding params
optimum_params = optimum_params.squeeze()
q = np.reshape(a[:, 0], [25, 25])  # reshaping 0th (q) column to 25x25
r_cost = np.reshape(a[:, 1], [25, 25])  # reshaping 1st (r) column to 25x25
values = values.reshape([25, 25])  # reshaping safe_sate values to 25x25
colours = ["red", "green"]
fig = plt.figure(figsize=(10, 10))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
ax.set_xlabel("q")
ax.set_ylabel("r")
cs = ax.contourf(q * 6, r_cost * 3, values)
ax.scatter(
    q * 6, r_cost * 3, c=values, cmap=matplotlib.colors.ListedColormap(colours),
)
ax.scatter(
    optimum_params[0] * 6,
    optimum_params[1] * 3,
    marker="<",
    color="b",
    s=np.asarray([200]),
)
ax.set_title("SafeSet belief after 500 iterations for DistrSafeOpt")
ax.set_ylim([-3.1, 3.1])
ax.set_xlim([-6.1, 6.1])
plt.savefig("./results/panda-robot/data/DBO_SafeSet_500.pdf", format="pdf", dpi=300)


# OvrDistrSafeOpt
df_ovr = df_ovr.to_numpy()
x = df_ovr[:, :2]
y = df_ovr[:, 2].reshape(-1, 1)
g = df_ovr[:, 3].reshape(-1, 1)

lengthscale = 0.7
L = [lengthscale / 6, lengthscale / 6]
KERNEL_f = GPy.kern.sde_Matern32(
    input_dim=x.shape[1], lengthscale=L, ARD=True, variance=1
)
KERNEL_g = GPy.kern.sde_Matern32(
    input_dim=x.shape[1], lengthscale=L, ARD=True, variance=1
)
fun_gp = GPy.models.GPRegression(x, y, noise_var=0.1 ** 2, kernel=KERNEL_f)
cons_gp = GPy.models.GPRegression(x, g, noise_var=0.1 ** 2, kernel=KERNEL_g)

bounds = [[-1, 1], [-1, 1]]
parameter_set = linearly_spaced_combinations(bounds, num_samples=100)
opt = SafeOpt([fun_gp, cons_gp], parameter_set, fmin=[-np.inf, 0], beta=3.5)

q = np.linspace(-1, 1, 25)
r_cost = np.linspace(-1, 1, 25)
a = np.asarray(np.meshgrid(q, r_cost)).T.reshape(-1, 2)
input = a
mean, var = opt.gps[1].predict(input)  # predicting `g1` for input
std = np.sqrt(var)
l_x0 = mean - opt.beta(opt.t) * std  # lower bound of constraint function value
safe_idx = np.where(l_x0 >= 0)[0]  # taking function value g1>=0 as safe
values = np.zeros(a.shape[0])  # start by taking all values as 0 to denote unsafe
values[safe_idx] = 1  # update safe values as 1

mean, var = opt.gps[0].predict(input)  # predicting `f` for input
l_f = mean - opt.beta(opt.t) * std  # lower bound of objective function value

safe_l_f = l_f[safe_idx]  # safe function values
safe_max = np.where(l_f == safe_l_f.max())[0]  # safe maximum
optimum_params = a[safe_max, :]  # take corresponding params
optimum_params = optimum_params.squeeze()
q = np.reshape(a[:, 0], [25, 25])  # reshaping 0th (q) column to 25x25
r_cost = np.reshape(a[:, 1], [25, 25])  # reshaping 1st (r) column to 25x25
values = values.reshape([25, 25])  # reshaping safe_sate values to 25x25
colours = ["red", "green"]
fig = plt.figure(figsize=(10, 10))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
ax.set_xlabel("q")
ax.set_ylabel("r")
cs = ax.contourf(q * 6, r_cost * 3, values)
ax.scatter(
    q * 6, r_cost * 3, c=values, cmap=matplotlib.colors.ListedColormap(colours),
)
ax.scatter(
    optimum_params[0] * 6,
    optimum_params[1] * 3,
    marker="<",
    color="b",
    s=np.asarray([200]),
)
ax.set_title("SafeSet belief after 500 iterations for OvrDistrSafeOpt")
ax.set_ylim([-3.1, 3.1])
ax.set_xlim([-6.1, 6.1])
plt.savefig("./results/panda-robot/data/OvrDBO_SafeSet_500.pdf", format="pdf", dpi=300)
