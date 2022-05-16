import GPy
import numpy as np
import pandas as pd
import safeopt
import ray
import time
import sys
import asyncio
import logging
from typing import List, Tuple, Dict, Callable


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
                ARD=self.kernel_dict["ARD"],
            )
        else:
            self.kernel = GPy.kern.RBF(
                input_dim=len(self.bounds), variance=2.0, lengthscale=1.0, ARD=True
            )

        self.worker_name = worker_name
        self.shared_data.update_worker_count.remote(True, self.worker_name)

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
            y = self.objective_function(x)
            gp = GPy.models.GPRegression(x, y, self.kernel, noise_var=self.noise_var)
            self.shared_data.append_point_evaluated.remote([x[0], y[0]])
        else:
            y = np.array(y)
            gp = GPy.models.GPRegression(x, y, self.kernel, noise_var=self.noise_var)

        # The optimization routine
        # opt = safeopt.SafeOptSwarm(
        #     gp, self.safe_threshold, bounds=self.bounds, threshold=0.2
        # )
        parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 200)
        opt = safeopt.SafeOpt(
            gp, parameter_set, self.safe_threshold, lipschitz=None, threshold=-5.0
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
                x_next = opt.optimize()
                # Get a measurement from the real system
                y_meas = self.objective_function(x_next)

                logging.debug(
                    "variables state\nx_next: %s\ny_next: %s",
                    np.array_str(x_next),
                    np.array_str(y_meas),
                )

                self.shared_data.append_point_evaluated.remote([x_next, y_meas[0]])

                # CHECK : whether to compare `y_meas` with `threshold` to confirm for safety and add to corresponding
                # hyperspace. Since non-safe point also provides some information about objective function.

                if y_meas >= self.safe_threshold:
                    new_hyperspace = list(
                        ray.get(self.shared_data.which_hyperspace.remote([x_next]))
                    )[0]
                    logging.debug("new_hyperspace: %s", new_hyperspace)
                    # found_new_hyperspace_flag = True
                    if new_hyperspace != current_hyperspace:
                        break

                # Add this to the GP model
                opt.add_new_data_point(x_next, y_meas)
                time.sleep(2)

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
                logging.info("No evaluations")
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

    if safe_set.shape[0] == 1:
        # if safe set contains only one point, then we have to deploy whole space to a process.

        logging.info("starting")

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


# Test functions for optimization needs - Marcin Molga, Czesław Smutnicki
def bird_function(X):
    """Bird function.
    -f* = 106.764537 at
    (x, y) = (4.70104, 3.15294) and
    (x, y) = (-1.58214, -3.13024)
    """
    X = np.atleast_2d(X)
    x = X[:, 0]
    y = X[:, 1]
    noise = np.random.normal(0, 0.25 ** 2)
    F = (
        np.sin(y) * np.exp((1 - np.cos(x)) ** 2)
        + np.cos(x) * np.exp((1 - np.sin(y)) ** 2)
        + (x - y) ** 2
        + noise
    )
    # optimization involves finding maximum value of function
    return -F.reshape((-1, 1))


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

    objective_function_name = "bird_function"
    bounds = [(-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi)]
    safe_threshold = -25.0
    # GP_regression noise
    noise_var = 0.05 ** 2
    kernel_dict = {
        "name": "RBF",
        "input_dim": len(bounds),
        "variance": 2.0,
        "lengthscale": 1.0,
        "ARD": True,
    }
    objective_function = bird_function
    # parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)
    no_subspaces = 2
    evaluation_constraint = 100
    shared_data: SharedData = SharedData.options(name="SharedData").remote(
        bounds, no_subspaces, evaluation_constraint
    )

    # Initial safe set
    x0 = np.zeros((1, len(bounds)))  # safe point at zero
    # x0 = np.array([[-2.5]]) # 1D single safe point
    # x0 = np.array([[4, 4]])  # 2D single safe point
    # x0 = np.array([[-2.5], [6.5]]) # 1D multiple safe points

    initial_deploy(
        x0, shared_data, objective_function, safe_threshold, noise_var, kernel_dict
    )
    await shared_data.work_done.remote()
    time.sleep(5)

    all_points = ray.get(shared_data.get_all_points_evaluated.remote())
    x, y = [], []
    for point in all_points:
        x.append(point[0])
        y.append(point[1])
    x = np.array(x)
    y = np.array(y)

    a = pd.DataFrame(x)
    b = pd.DataFrame(y)
    points_df = pd.concat((a, b), axis=1)
    time_str = time.strftime("%d_%b_%H_%M_%S_", time.localtime())

    no_points_evaluated = points_df.shape[0]
    dimension = points_df.shape[1]

    labels = ["x" + str(i) for i in range(dimension - 1)]
    labels.append("y")
    points_df.set_axis(labels=labels, axis="columns", inplace=True)
    points_df.to_csv(
        "./results-data/" + objective_function_name + "_dbo_" + time_str + "log.csv",
        index=False,
    )

    report = "Number of points evaluated : %d\n" % no_points_evaluated
    no_unsafe_evaluation = points_df.y[points_df.y < safe_threshold].count()
    report += "Number of unsafe evaluations : %d\n" % no_unsafe_evaluation
    optimum_value = points_df["y"].max()
    optimum_value_at = points_df.iloc[points_df["y"].idxmax()][0 : points_df.shape[1]]
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
