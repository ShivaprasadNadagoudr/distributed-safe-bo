import GPy
import numpy as np
import pandas as pd
import safeopt
import ray
import time
import asyncio
import logging
from typing import List, Tuple, Dict, Callable
import sys
import os
from safeopt import linearly_spaced_combinations
from safeopt import SafeOptSwarm, SafeOpt
import gym
import pandaenv  # Library defined for the panda environment
import mujoco_py
import scipy
from pandaenv.utils import inverse_dynamics_control
import random


class System(object):
    def __init__(
        self, position_bound, velocity_bound, rollout_limit=0, upper_eigenvalue=0
    ):
        self.env = gym.make("PandaEnvBasic-v0")
        # Define Q,R, A, B, C matrices
        self.Q = np.eye(6)
        self.R = np.eye(3) / 100
        self.env.seed(0)
        self.obs = self.env.reset()
        self.A = np.zeros([6, 6])
        # A=np.zeros([18,18])
        self.A[:3, 3:] = np.eye(3)
        self.B = np.zeros([6, 3])
        self.B[3:, :] = np.eye(3)
        self.T = 2000
        # Set up inverse dynamics controller
        self.ID = inverse_dynamics_control(
            env=self.env, njoints=9, target=self.env.goal
        )
        self.id = self.env.sim.model.site_name2id("panda:grip")
        self.rollout_limit = rollout_limit
        self.at_boundary = False
        self.Fail = False
        self.approx = True
        self.position_bound = position_bound
        self.velocity_bound = velocity_bound
        self.upper_eigenvalue = upper_eigenvalue
        # Define weighting matrix to set torques 4,6,7,.. (not required for movement) to 0.
        T1 = np.zeros(9)
        T1[4] = 1
        T1[6:] = 1
        T = np.diag(T1)
        N = np.eye(9) - np.dot(np.linalg.pinv(T, rcond=1e-4), T)
        self.N_bar = np.dot(N, np.linalg.pinv(np.dot(np.eye(9), N), rcond=1e-4))

    def simulate(self, params=None, render=False, opt=None, update=False):
        # Simulate the robot/run experiments
        x0 = None
        # Update parameters
        if params is not None:

            if update:
                param_a = self.set_params(params)
                self.Q = np.diag(param_a)
            else:
                self.Q = np.diag(params)

            # param is between -1 and 1
            self.R = np.eye(3) / 100 * np.power(10, 3 * params[1])
            # If want to change inition condition update, IC
            if opt is not None:
                if opt.criterion in ["S2"]:
                    x0 = params[opt.state_idx]
                    x0[3:] = np.zeros(3)

        # Get controller
        P = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R))
        K = scipy.linalg.inv(self.R) * (self.B.T * P)

        K = np.asarray(K)
        eigen_value = np.linalg.eig(self.A - np.dot(self.B, K))
        eigen_value = np.max(np.asarray(eigen_value[0]).real)

        Kp = K[:, :3]
        Kd = K[:, 3:]
        Objective = 0
        self.reset(x0)
        state = []
        constraint2 = 0
        if x0 is not None:
            x = np.hstack([params[:2].reshape(1, -1), x0.reshape(1, -1)])
            state.append(x)

        else:
            obs = self.obs["observation"].copy()
            obs[:3] = obs[:3] - self.env.goal
            obs[:3] /= self.position_bound
            obs[3:] /= self.velocity_bound
            x = np.hstack([params[:2].reshape(1, -1), obs.reshape(1, -1)])
            state.append(x)

        if opt is not None:
            if eigen_value > self.upper_eigenvalue and opt.criterion == "S3":
                self.at_boundary = True
                opt.s3_steps = np.maximum(0, opt.s3_steps - 1)
                return 0, 0, 0, state

        elif eigen_value > self.upper_eigenvalue:
            self.at_boundary = True
            self.Fail = True
            print("Eigenvalues too high ", end="")
            return 0, 0, 0, state

        if render:
            init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
            # init_dist=self.init_dist
            for i in range(self.T):

                bias = self.ID.g()

                J = self.ID.Jp(self.id)

                wM_des = np.dot(
                    Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"])
                ) - np.dot(
                    Kd,
                    self.obs["observation"][3:]
                    - np.ones(3) * 1 / self.env.Tmax * (i < self.env.Tmax),
                )
                u = -bias

                u += np.dot(J.T, wM_des)
                u = np.dot(self.N_bar, u)
                self.obs, reward, done, info = self.env.step(u)
                Objective += reward
                constraint2 = np.maximum(
                    np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
                    - init_dist,
                    constraint2,
                )
                # constraint_2 = np.maximum(np.max(np.abs(self.obs["observation"][3:])), constraint_2)
                self.env.render()

        else:
            # Simulate the arm
            init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
            # init_dist = self.init_dist
            for i in range(self.T):
                if opt is not None and not self.at_boundary:
                    obs = self.obs["observation"].copy()
                    obs[:3] -= self.env.goal
                    obs[:3] /= self.position_bound
                    obs[3:] /= self.velocity_bound
                    # obs[3:]=opt.x_0[:,3:]
                    # Evaluate Boundary condition/not necessary here as we use SafeOpt
                    if i % 10 == 0:
                        self.at_boundary, self.Fail, params = opt.check_rollout(
                            state=obs.reshape(1, -1), action=params
                        )

                    if self.Fail:
                        print("FAILED                  ", i, end=" ")
                        return 0, 0, 0, state
                    elif self.at_boundary:
                        params = params.squeeze()
                        print(" Changed action to", i, params, end="")
                        param_a = self.set_params(params.squeeze())
                        self.Q = np.diag(param_a)

                        self.R = np.eye(3) / 100 * np.power(10, 3 * params[1])
                        P = np.matrix(
                            scipy.linalg.solve_continuous_are(
                                self.A, self.B, self.Q, self.R
                            )
                        )
                        K = scipy.linalg.inv(self.R) * (self.B.T * P)

                        K = np.asarray(K)

                        Kp = K[:, :3]
                        Kd = K[:, 3:]

                # Collect rollouts (not necessary here as we run safeopt)
                if i < self.rollout_limit:
                    obs = self.obs["observation"].copy()
                    obs[:3] = obs[:3] - self.env.goal
                    obs[:3] /= self.position_bound
                    obs[3:] /= self.velocity_bound
                    x = np.hstack([params[:2].reshape(1, -1), obs.reshape(1, 1)])
                    state.append(x)
                bias = self.ID.g()

                J = self.ID.Jp(self.id)

                wM_des = np.dot(
                    Kp, (self.obs["desired_goal"] - self.obs["achieved_goal"])
                ) - np.dot(
                    Kd,
                    self.obs["observation"][3:]
                    - np.ones(3) * 1 / self.env.Tmax * (i < self.env.Tmax),
                )
                u = -bias
                u += np.dot(J.T, wM_des)
                u = np.dot(self.N_bar, u)
                self.obs, reward, done, info = self.env.step(u)
                Objective += reward
                constraint2 = np.maximum(
                    np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
                    - init_dist,
                    constraint2,
                )

                # constraint2 = np.maximum(np.max(np.abs(self.obs["observation"][3:])), constraint2)

        return Objective / self.T, constraint2 / init_dist, eigen_value, state

    def reset(self, x0=None):
        """
        Reset environmnent for next experiment
        """
        self.obs = self.env.reset()
        # self.init_dist = np.linalg.norm(self.env.goal - self.obs["achieved_goal"])
        self.Fail = False
        self.at_boundary = False

        if x0 is not None:
            x0 *= self.position_bound
            self.env.goal = self.obs["observation"][:3] - x0[:3]

    def set_params(self, params):
        """
        Update parameters for controller
        """
        q1 = np.repeat(np.power(10, 6 * params[0]), 3)  # param is between -1 and 1
        q2 = np.sqrt(q1) * 0.1
        updated_params = np.hstack((q1.squeeze(), q2.squeeze()))
        return updated_params


@ray.remote(resources={"resource2": 1})
class SharedData:
    """
    This will be a Ray Actor, which holds the global state / data shared among all workers.
    Why an Actor, and not simple queue or ray.put() methods?
    -- The data shared by ray.queue() and ray.put() is read only. So we cannot update the shared data.
    In our case we need all the points evaluated across whole search space as a prior to GP while
    instantiating new hyperspace into a worker node.
    """

    def __init__(
        self, bounds: List[Tuple], no_subspaces: int, no_evaluations: int = 50,
    ) -> None:
        """Creates a SharedData actor on a worker, also initializes shared data structures.

        Args:
            bounds:
                List of tuple containing two items i.e: lower bound and upper bound for each search parameter.
            no_subspaces:
                How many number of subspaces to create for a search parameter.
        """
        time_str = time.strftime("%d_%b_%H_%M_%S_", time.localtime())
        logging.basicConfig(
            level=logging.DEBUG,
            filename="logs/" + time_str + "SharedData.log",
            filemode="w",
            format="%(process)d-%(levelname)s-%(funcName)s-%(message)s",
        )
        self.no_evaluations = no_evaluations
        self.bounds = bounds
        self.no_subspaces = no_subspaces
        self.all_points_evaluated = []  # [X, y]
        self.all_parameter_subspaces = []
        self.subspace_indices_for_hyperspace = []
        self.subspaces_deployment_status = []
        self.hyperspaces_list = []  # contains all possible combinations of subspaces
        self.create_hyperspaces()
        self.bounds_indices = [[0, no_subspaces - 1] for _ in range(len(bounds))]
        self.worker_count = 0
        self.event = asyncio.Event()

    async def work_done(self):
        await self.event.wait()

    def update_worker_count(self, presence: bool, worker_name: str):
        if presence:
            self.worker_count += 1
            # print(worker_name, "created")
        else:
            self.worker_count -= 1
            # print(worker_name, "shutdown")
        # print("worker_count=%d" % self.worker_count)
        if self.worker_count == 0:
            # print("all processes died")
            # time.sleep(5)
            self.event.set()

    def get_no_evaluations(self):
        evals = self.no_evaluations
        self.no_evaluations -= 1
        return evals

    def get_current_no_evaluations(self):
        return self.no_evaluations

    def get_current_worker_count(self):
        return self.worker_count

    def append_point_evaluated(self, point) -> None:
        """Appends a newly evaluated point in a hyperspace into SharedData all_points_evaluated list"""
        self.all_points_evaluated.append(point)
        # logging.info(self.all_points_evaluated)

    def get_all_points_evaluated(self):
        # x, y = [], []
        # for evaluation in self.all_points_evaluated:
        #     x.append(evaluation[:-1])
        #     y.append(evaluation[-1])
        return self.all_points_evaluated

    def get_hyperspaces_list(self):
        return self.hyperspaces_list

    def get_bounds(self):
        return self.bounds

    def get_bounds_indices(self):
        return self.bounds_indices

    def get_subspace_indices_for_hyperspace(self, hyperspace_no):
        return self.subspace_indices_for_hyperspace[hyperspace_no]

    def create_hyperspaces(self):
        """Creates hyperspaces for the given parameters by dividing each of them into given number of subspaces."""

        # to divide each parameter space into given number of subspaces
        for parameter_space in self.bounds:
            low, high = parameter_space
            subspace_length = abs(high - low) / self.no_subspaces
            parameter_subspaces = []
            for i in range(self.no_subspaces):
                end = low + subspace_length
                parameter_subspaces.append((round(low, 8), round(end, 8) - 0.00000001))
                low = end
            self.all_parameter_subspaces.append(parameter_subspaces)

        rows = len(self.all_parameter_subspaces)  # no_parameters
        columns = len(self.all_parameter_subspaces[0])  # no_subspaces

        # initializing deployed status for each subsapce
        for i in range(rows):
            each_parameter_subspaces = []
            for _ in range(columns):
                # for each subspace maintain these 3 states to determine its deployment status
                # [is this subspace deployed (True/False), associated with any subspace (True/False),
                #  hyperspace number (None-not deployed with any hyperspace/int-deployed with that hyperspace)]
                each_parameter_subspaces.append(False)
            self.subspaces_deployment_status.append(each_parameter_subspaces)

        # no_hyperspaces = no_subspaces ** no_parameters
        for _ in range(columns ** rows):
            self.hyperspaces_list.append([])
            self.subspace_indices_for_hyperspace.append([])

        for row in range(rows):
            repeat = columns ** (rows - row - 1)
            for column in range(columns):
                item = self.all_parameter_subspaces[row][column]
                start = column * repeat
                for times in range(columns ** row):
                    for l in range(repeat):
                        self.hyperspaces_list[start + l].append(item)
                        self.subspace_indices_for_hyperspace[start + l].append(column)
                    start += columns * repeat
        logging.info("Hyperspaces are created successfully")
        logging.info(self.hyperspaces_list)

        # set bounds from hyperspaces created
        lower_bounds = [l for l, h in self.hyperspaces_list[0]]
        higher_bounds = [h for l, h in self.hyperspaces_list[-1]]
        self.bounds = list(zip(lower_bounds, higher_bounds))
        # return self.hyperspaces_list

    def which_hyperspace(self, x: List[List]) -> Dict:
        """Creates a dictionary with hyperspace number as key and list of points belong to that hyperspace as
        corresponding value.

        Args:
            x:
                List of points.
            hyperspaces:
                List containing all hyperspaces.

        Returns:
            safe_hyperspaces:
                Dictionary with hyperspace number and list of points as key-value pair.
        """
        safe_hyperspaces = {}
        # logging.debug("hyperspace_list lenght: %s", len(hyperspaces_list))
        # logging.debug(x)
        for point in x:
            # logging.debug("point: %s", point)
            for i, hyperspace in enumerate(self.hyperspaces_list):
                # logging.debug("hyperspace : %s = %s", i, hyperspace)
                belongs_flag = True
                for dimension, value in enumerate(point):
                    # logging.debug("\ndimension:%s\nvalue:%s", dimension, value)
                    if (
                        value >= hyperspace[dimension][0]
                        and value <= hyperspace[dimension][1]
                    ):
                        # logging.info("inside if condition, so just continue")
                        continue
                    else:
                        belongs_flag = False
                        # logging.info("not belongs to this hyperspace")
                        break
                if belongs_flag:
                    if i not in safe_hyperspaces:
                        safe_hyperspaces[i] = [point]
                    else:
                        safe_hyperspaces[i].append(point)
                    break
        # logging.debug(safe_hyperspaces)
        return safe_hyperspaces

    def split_search_space(self, ss, hs1, hs2) -> Tuple:
        """Splits the given search space `ss` between hyperspaces `hs1` and `hs2`,
        and returns the new search space bounds for both `hs1` and `hs2`.

        Args:
            ss:
            hs1:
            hs2:

        Returns:
            ss
        """
        logging.debug("Search space : %s\nhs1: %s\nhs2 : %s", ss, hs1, hs2)
        ss1 = []
        ss2 = []
        reverse_flag = False
        for param_index, (i, j) in enumerate(zip(hs1, hs2)):
            start, end = ss[param_index]
            if i == j:
                ss1.append(ss[param_index])
                ss2.append(ss[param_index])
            else:
                if i > j:
                    i, j = j, i
                    reverse_flag = True
                ds = self.subspaces_deployment_status[param_index]
                ds[i] = True
                ds[j] = True
                in_between = j - i - 1
                hs1_right = in_between // 2
                hs2_left = in_between - hs1_right
                hs1_low = i
                hs1_high = i + hs1_right
                hs2_low = j - hs2_left
                hs2_high = j

                for k in range(i - 1, start - 1, -1):
                    if not ds[k]:
                        hs1_low = k
                    else:
                        break

                for k in range(j + 1, end + 1):
                    if not ds[k]:
                        hs2_high = k
                    else:
                        break

                ss1.append([hs1_low, hs1_high])
                ss2.append([hs2_low, hs2_high])
                break
        if param_index < len(ss) - 1:
            for i in range(param_index + 1, len(ss)):
                ss1.append(ss[i])
                ss2.append(ss[i])

        logging.debug("Splitted seaech space\nss1 : %s\nss2 : %s", ss1, ss2)
        if reverse_flag:
            return (ss2, ss1)
        else:
            return (ss1, ss2)

    def get_bounds_from_index(self, hyperspace_bounds: List[Tuple]) -> List[Tuple]:
        """Returns the actual search bounds for the hyperspace defined in terms of subspace indices."""
        bounds = []
        for parameter_no, bound in enumerate(hyperspace_bounds):
            parameter_subspaces = self.all_parameter_subspaces[parameter_no]
            low = parameter_subspaces[bound[0]][0]
            high = parameter_subspaces[bound[1]][1]
            bounds.append((low, high))
        return bounds


@ray.remote(resources={"resource1": 1})
class DeployHyperspace:
    """
    This Actor work on dividing the hyperspace as per the algorithm and deploy new worker
    """

    def __init__(
        self,
        objective_function: Callable,
        safe_threshold: float,
        kernel_dict: Dict,
        noise_var: float,
        shared_data: SharedData,
        bounds=None,
        bounds_indices=None,
        worker_name: str = None,
    ) -> None:
        """Initializes with common data for optimization"""
        time_str = time.strftime("%d_%b_%H_%M_%S_", time.localtime())
        logging.basicConfig(
            level=logging.DEBUG,
            filename="logs/" + time_str + worker_name + ".log",
            filemode="w",
            format="%(process)d-%(levelname)s-%(funcName)s-%(message)s",
        )

        # self.safe_set = x
        self.objective_function = objective_function
        self.safe_threshold = safe_threshold
        self.noise_var = noise_var
        self.shared_data = shared_data  # SharedData.remote(bounds, no_subspaces)
        self.bounds = (
            bounds if bounds else ray.get(self.shared_data.get_bounds.remote())
        )
        self.bounds_indices = (
            bounds_indices
            if bounds_indices
            else ray.get(self.shared_data.get_bounds_indices.remote())
        )
        #  define the kernel from dictionary
        self.kernel_dict = kernel_dict
        if self.kernel_dict["name"] == "RBF":
            self.kernel = GPy.kern.RBF(
                input_dim=self.kernel_dict["input_dim"],
                variance=self.kernel_dict["variance"],
                lengthscale=self.kernel_dict["lengthscale"],
                ARD=self.kernel_dict["ARD"],
            )
        elif self.kernel_dict["name"] == "Matern32":
            self.kernel = GPy.kern.Matern32(
                input_dim=self.kernel_dict["input_dim"],
                variance=self.kernel_dict["variance"],
                lengthscale=self.kernel_dict["lengthscale"],
                ARD=self.kernel_dict["ARD"],
            )
        else:
            self.kernel = GPy.kern.RBF(
                input_dim=len(self.bounds), variance=2.0, lengthscale=1.0, ARD=True
            )

        self.worker_name = worker_name
        self.shared_data.update_worker_count.remote(True, self.worker_name)

        ########################

        # upper_overshoot = 0.08
        # upper_eigenvalue = -10
        lengthscale = 0.7
        self.upper_eigenvalue = -10
        self.upper_overshoot = 0.08
        self.failure_overshoot = 0
        self.mean_reward = -0.33
        self.std_reward = 0.14
        self.eigen_value_std = 21
        # Set up system and optimizer
        self.sys = System(
            rollout_limit=0,
            position_bound=0.5,
            velocity_bound=7,
            upper_eigenvalue=self.upper_eigenvalue,
        )

        L = [lengthscale / 6, lengthscale / 6]
        self.KERNEL_f = GPy.kern.sde_Matern32(
            input_dim=len(self.bounds), lengthscale=L, ARD=True, variance=1
        )
        self.KERNEL_g = GPy.kern.sde_Matern32(
            input_dim=len(self.bounds), lengthscale=L, ARD=True, variance=1
        )

    def set_bounds(self, bounds, bounds_indices):
        logging.info("Updating bounds")
        self.bounds = bounds
        self.bounds_indices = bounds_indices

    def optimization(
        self, x: List[List], y: List[List], safe_hyperspace_no: int,
    ) -> None:
        """ """
        logging.info("optimization started")

        current_hyperspace = safe_hyperspace_no
        new_hyperspace = current_hyperspace

        if x is None:
            all_points = ray.get(self.shared_data.get_all_points_evaluated.remote())
            x, y = [], []
            for point in all_points:
                x.append(point[0])
                y.append(point[1])
            x = np.array(x)
            y = np.array(y)
        else:
            x = np.array(x)

        # The statistical model of our objective function
        if y is None:
            y = []
            for params in x:
                f_, g1_, g2, state = self.sys.simulate(params, update=True)
                g2 = self.upper_eigenvalue - g2
                g2 /= self.eigen_value_std
                g1_ -= self.upper_overshoot
                g1_ = -g1_ / self.upper_overshoot
                f_ -= self.mean_reward
                f_ /= self.std_reward
                f = np.asarray([[f_]])
                f = f.reshape(-1, 1)
                g1 = np.asarray([[g1_]])
                g1 = g1.reshape(-1, 1)
                y_val = np.array([[f], [g1]])
                y_val = y_val.squeeze()
                constraint_satisified = g1 >= 0 and g2 >= 0
                print_str = f"params= {params}  | f={f_:.4f}  | g1={g1_:.2f}  | g2={g2:.2f}  | safe={constraint_satisified}"
                print(print_str)
                self.shared_data.append_point_evaluated.remote(
                    [params, y_val, [constraint_satisified]]
                )
                y.append(y_val)
            y = np.array(y)
            fun_gp = GPy.models.GPRegression(
                x, y[:, 0].reshape(-1, 1), self.KERNEL_f, noise_var=self.noise_var
            )
            cons_gp = GPy.models.GPRegression(
                x, y[:, 1].reshape(-1, 1), self.KERNEL_g, noise_var=self.noise_var
            )
        else:
            y = np.array(y)
            fun_gp = GPy.models.GPRegression(
                x, y[:, 0].reshape(-1, 1), self.KERNEL_f, noise_var=self.noise_var
            )
            cons_gp = GPy.models.GPRegression(
                x, y[:, 1].reshape(-1, 1), self.KERNEL_g, noise_var=self.noise_var
            )

        # The optimization routine
        # opt = safeopt.SafeOptSwarm(
        #     gp, self.safe_threshold, bounds=self.bounds, threshold=0.2
        # )
        parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 100)
        opt = safeopt.SafeOpt(
            [fun_gp, cons_gp], parameter_set, fmin=self.safe_threshold, beta=3.5
        )
        logging.debug(opt.bounds)

        logging.debug(
            "optimization for \nhyperspace_no: %s\nbounds: %s",
            current_hyperspace,
            self.bounds,
        )

        # found_new_hyperspace_flag = False
        try:
            evaluation_constraint = ray.get(
                self.shared_data.get_no_evaluations.remote()
            )
            # logging.info(
            #     "Before while - Evaluation constraint %d", evaluation_constraint
            # )
            while evaluation_constraint > 0:
                # logging.info("iteration: %s", i)
                logging.info("Evaluation constraint %d", evaluation_constraint)

                # obtain new query point
                params = opt.optimize()

                # Get a measurement from the real system
                f, g1, g2, state = self.sys.simulate(params, update=True)
                # Update and add collected data to GP
                g2 = self.upper_eigenvalue - g2
                g2 /= self.eigen_value_std
                g1 -= self.upper_overshoot
                g1 = -g1 / self.upper_overshoot
                f -= self.mean_reward
                f /= self.std_reward

                y_meas = np.array([[f], [g1]])
                y_meas = y_meas.squeeze()

                constraint_satisified = g1 >= 0 and g2 >= 0
                if not constraint_satisified:
                    self.failure_overshoot += (
                        -(g1 * self.upper_overshoot) + self.upper_overshoot
                    )

                print_str = f"iter={evaluation_constraint:3d}  | params= {params}  | f={f:.4f}  | g1={g1:.2f}  | g2={g2:.2f}  | safe={constraint_satisified}"
                print(print_str)

                self.shared_data.append_point_evaluated.remote(
                    [params, y_meas, [constraint_satisified]]
                )

                # CHECK : whether to compare `y_meas` with `threshold` to confirm for safety and add to corresponding
                # hyperspace. Since non-safe point also provides some information about objective function.

                # if y_meas >= self.safe_threshold:
                if constraint_satisified:
                    new_hyperspace = list(
                        ray.get(self.shared_data.which_hyperspace.remote([params]))
                    )[0]
                    logging.debug("new_hyperspace: %s", new_hyperspace)
                    # found_new_hyperspace_flag = True
                    if new_hyperspace != current_hyperspace:
                        break

                # Add this to the GP model
                opt.add_new_data_point(params, y_meas)
                time.sleep(1)

                # get the evaluation_constraint
                evaluation_constraint = ray.get(
                    self.shared_data.get_no_evaluations.remote()
                )
        except Exception as e:
            logging.debug("Exception occured")
            logging.exception(e)
        finally:
            logging.info("Finally - Evaluation constraint %d", evaluation_constraint)
            if evaluation_constraint <= 0:
                logging.info("No more evaluations to do")
            # if found_new_hyperspace_flag:
            #     return (current_hyperspace, new_hyperspace)
            # else:
            #     return (current_hyperspace, current_hyperspace)
            return (current_hyperspace, new_hyperspace)

    def deploy_hyperspace(self, hyperspace_no) -> None:
        logging.info("starting")
        logging.debug("deploy new hyperspace : %s", hyperspace_no)

        # check for leaf node
        is_leaf_flag = True
        for bound in self.bounds_indices:
            if bound[0] != bound[1]:
                is_leaf_flag = False
        if is_leaf_flag:
            logging.info("LEAF node deployment")
            logging.debug("search space bounds : %s", self.bounds)
            logging.info("calling for optimization")
            current_hyperspace, new_hyperspace = self.optimization(
                None, None, hyperspace_no,
            )

            logging.debug(
                "current_hyperspace : %s \nnew_hyperspace : %s",
                current_hyperspace,
                new_hyperspace,
            )
            self.shared_data.update_worker_count.remote(False, self.worker_name)
            ray.actor.exit_actor()
            return

        logging.debug("search space bounds : %s", self.bounds)
        logging.info("calling for optimization")
        (current_hyperspace, new_hyperspace) = self.optimization(
            None, None, hyperspace_no,
        )

        logging.debug(
            "current_hyperspace : %s \tnew_hyperspace : %s",
            current_hyperspace,
            new_hyperspace,
        )

        if current_hyperspace == new_hyperspace:
            logging.info("current_hyperspace == new_hyperspace")
            self.shared_data.update_worker_count.remote(False, self.worker_name)
            ray.actor.exit_actor()
            return

        hs1 = ray.get(
            self.shared_data.get_subspace_indices_for_hyperspace.remote(
                current_hyperspace
            )
        )
        hs2 = ray.get(
            self.shared_data.get_subspace_indices_for_hyperspace.remote(new_hyperspace)
        )
        logging.debug("\nhyperspace_1 : %s \nhyperspace_2 : %s", hs1, hs2)
        logging.info("Splitting the search space for both hyperspaces")
        ss1, ss2 = ray.get(
            self.shared_data.split_search_space.remote(self.bounds_indices, hs1, hs2)
        )
        logging.debug(
            "\nbounds_indices : %s \nsearch_space_1 : %s \nsearch_space_2 : %s",
            self.bounds_indices,
            ss1,
            ss2,
        )

        logging.info("forking deploy_hyperspace for search_space_2")
        no_current_workers = ray.get(self.shared_data.get_current_worker_count.remote())
        no_evaluations_remaining = ray.get(
            self.shared_data.get_current_no_evaluations.remote()
        )
        logging.info(
            "no_evaluations_remaining: %d | no_current_workers: %d",
            no_evaluations_remaining,
            no_current_workers,
        )
        if no_evaluations_remaining > no_current_workers:
            worker_name = "HyperSpace_" + str(new_hyperspace)
            new_node: DeployHyperspace = DeployHyperspace.options(
                name=worker_name, lifetime="detached"
            ).remote(
                self.objective_function,
                self.safe_threshold,
                self.kernel_dict,
                self.noise_var,
                self.shared_data,
                bounds=ray.get(self.shared_data.get_bounds_from_index.remote(ss2)),
                bounds_indices=ss2,
                worker_name=worker_name,
            )

            new_node.deploy_hyperspace.remote(new_hyperspace)

            self.bounds = ray.get(self.shared_data.get_bounds_from_index.remote(ss1))
            self.bounds_indices = ss1
            self.deploy_hyperspace(current_hyperspace)
        else:
            logging.info("no_evaluations_remaining <= no_current_workers")
            self.shared_data.update_worker_count.remote(False, self.worker_name)
            ray.actor.exit_actor()
            return


def initial_deploy(
    safe_set,
    shared_data: SharedData,
    objective_function,
    safe_threshold,
    noise_var,
    kernel_dict: Dict,
) -> None:
    bounds_indices = ray.get(shared_data.get_bounds_indices.remote())

    if safe_set.shape[0] == 3:
        # if safe set contains only one point, then we have to deploy whole space to a process.

        logging.info("starting for single point")

        safe_hyperspace_no = list(
            ray.get(shared_data.which_hyperspace.remote(safe_set))
        )[0]

        logging.info("calling optimization for hyperspace %d", safe_hyperspace_no)

        # bounds = ray.get(shared_data.get_bounds_from_index.remote(bounds_indices))

        worker_name = "HyperSpace_" + str(safe_hyperspace_no)
        initial_worker: DeployHyperspace = DeployHyperspace.options(
            name=worker_name, lifetime="detached"
        ).remote(
            objective_function,
            safe_threshold,
            kernel_dict,
            noise_var,
            shared_data,
            worker_name=worker_name,
        )

        current_hyperspace, new_hyperspace = ray.get(
            initial_worker.optimization.remote(safe_set, None, safe_hyperspace_no)
        )

        logging.debug(
            "\ncurrent_hyperspace : %s \nnew_hyperspace : %s",
            current_hyperspace,
            new_hyperspace,
        )

        if current_hyperspace == new_hyperspace:
            # if no new safe points in other hyperspace,
            # then ask user to whether continue or go for more number of evaluations
            shared_data.update_worker_count.remote(False, worker_name)
            ray.kill(initial_worker)
            return

        # call split search space method
        hs1 = ray.get(
            shared_data.get_subspace_indices_for_hyperspace.remote(current_hyperspace)
        )
        hs2 = ray.get(
            shared_data.get_subspace_indices_for_hyperspace.remote(new_hyperspace)
        )

        logging.debug("\nhyperspace_1 : %s \nhyperspace_2 : %s", hs1, hs2)
        logging.info("Splitting the search space for both hyperspaces")

        ss1, ss2 = ray.get(
            shared_data.split_search_space.remote(bounds_indices, hs1, hs2)
        )

        logging.debug(
            "\nbounds_indices : %s \nsearch_space_1 : %s \nsearch_space_2 : %s",
            bounds_indices,
            ss1,
            ss2,
        )

        logging.info("forking new process -- deploy_hyperspace for search_space_2")
        no_current_workers = ray.get(shared_data.get_current_worker_count.remote())
        no_evaluations_remaining = ray.get(
            shared_data.get_current_no_evaluations.remote()
        )
        logging.info(
            "no_evaluations_remaining: %d | no_current_workers: %d",
            no_evaluations_remaining,
            no_current_workers,
        )
        if no_evaluations_remaining > no_current_workers:
            worker_name = "HyperSpace_" + str(new_hyperspace)
            new_worker: DeployHyperspace = DeployHyperspace.options(
                name=worker_name, lifetime="detached"
            ).remote(
                objective_function,
                safe_threshold,
                kernel_dict,
                noise_var,
                shared_data,
                bounds=ray.get(shared_data.get_bounds_from_index.remote(ss2)),
                bounds_indices=ss2,
                worker_name=worker_name,
            )
            new_worker.deploy_hyperspace.remote(new_hyperspace)

            bounds = ray.get(shared_data.get_bounds_from_index.remote(ss1))
            bounds_indices = ss1
            initial_worker.set_bounds.remote(bounds, bounds_indices)
            initial_worker.deploy_hyperspace.remote(current_hyperspace)
        else:
            logging.info("no_evaluations_remaining <= no_current_workers")
            shared_data.update_worker_count.remote(False, worker_name)
            ray.kill(initial_worker)
            return

    elif safe_set.shape[0] != 0:
        # if safe set is not empty, deploy corresponding hyperspaces for given points in safe set.
        safe_hyperspaces = shared_data.which_hyperspace.remote(safe_set)
        # deploy_hyperspace(safe_hyperspaces, hyperspaces)


async def main():
    ray.init(
        address="172.20.46.18:8888", _redis_password="5241590000000000", namespace="dbo"
    )

    time_str = time.strftime("%d_%b_%H_%M_%S_", time.localtime())
    logging.basicConfig(
        level=logging.DEBUG,
        filename="logs/" + time_str + "driver.log",
        filemode="w",
        format="%(process)d-%(levelname)s-%(funcName)s-%(message)s",
    )

    objective_function_name = "panda_robot"
    bounds = [(1 / 3, 1), (-1, 1)]
    safe_threshold = [-np.inf, 0]
    # GP_regression noise
    noise_var = 0.1 ** 2
    kernel_dict = {
        "name": "RBF",
        "input_dim": len(bounds),
        "variance": 2.0,
        "lengthscale": 1.0,
        "ARD": True,
    }
    objective_function = None
    # parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)
    no_subspaces = 2
    evaluation_constraint = 10
    shared_data: SharedData = SharedData.options(name="SharedData").remote(
        bounds, no_subspaces, evaluation_constraint
    )

    # Initial safe set
    x0 = np.array([[4 / 6, -1], [5 / 6, -0.9], [1, -2 / 3]])  # 2D multiple safe point

    initial_deploy(
        x0, shared_data, objective_function, safe_threshold, noise_var, kernel_dict
    )
    await shared_data.work_done.remote()
    time.sleep(5)

    all_points = ray.get(shared_data.get_all_points_evaluated.remote())
    x, y, safe = [], [], []
    for point in all_points:
        x.append(point[0])
        y.append(point[1])
        safe.append(point[2])
    x = np.array(x)
    y = np.array(y)
    safe = np.array(safe)

    a = pd.DataFrame(x)
    b = pd.DataFrame(y)
    c = pd.DataFrame(safe)
    points_df = pd.concat((a, b, c), axis=1)
    time_str = time.strftime("%d_%b_%H_%M_%S_", time.localtime())

    no_points_evaluated = points_df.shape[0]
    dimension = points_df.shape[1]

    labels = ["x" + str(i) for i in range(dimension - 3)]
    labels.append("y")
    labels.append("g")
    labels.append("safe")
    points_df.set_axis(labels=labels, axis="columns", inplace=True)
    points_df.to_csv(
        "./results-data/" + objective_function_name + "_dbo_" + time_str + "log.csv",
        index=False,
    )

    report = "Number of points evaluated : %d\n" % no_points_evaluated
    no_unsafe_evaluation = points_df["safe"].apply(lambda v: 0 if v else 1).sum()
    report += "Number of unsafe evaluations : %d\n" % no_unsafe_evaluation
    optimum_value = points_df["y"].max()
    optimum_value_at = points_df.iloc[points_df["y"].idxmax()][
        0 : points_df.shape[1] - 3
    ]
    report += "Optimization results\ny = %f\nat\n%s" % (
        optimum_value,
        optimum_value_at.to_string(),
    )
    with open(
        "./results-data/" + objective_function_name + "_dbo_" + time_str + "log.txt",
        "w",
    ) as res_file:
        res_file.write(report)

    print(report)

    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
