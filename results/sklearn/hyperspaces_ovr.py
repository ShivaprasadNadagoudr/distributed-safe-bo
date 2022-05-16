import GPy
import numpy as np
import safeopt
import ray
import time
import logging
import pandas as pd
import random
import sys
import asyncio
from typing import List, Tuple, Dict, Callable


@ray.remote
class SharedData:
    """
    This will be a Ray Actor, which holds the global state / data shared among all workers.
    Why an Actor, and not simple queue or ray.put() methods?
    -- The data shared by ray.queue() and ray.put() is read only. So we cannot update the shared data.
    In our case we need all the points evaluated across whole search space as a prior to GP while
    instantiating new hyperspace into a worker node.
    """

    def __init__(
        self,
        bounds: List[Tuple],
        no_subspaces: int,
        no_evaluations: int = 50,
        overlap: float = 0.15,
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
        self.overlap = overlap
        self.create_hyperspaces()
        self.bounds_indices = [[0, no_subspaces - 1] for _ in range(len(bounds))]
        self.shared_points_to_hyperspaces = [
            [] for _ in range(len(self.hyperspaces_list))
        ]
        self.worker_count = 0
        self.event = asyncio.Event()

    async def work_done(self):
        await self.event.wait()

    def update_worker_count(self, presence: bool, worker_name: str):
        if presence:
            print(worker_name, "created")
            self.worker_count += 1
        else:
            print(worker_name, "shutdown")
            self.worker_count -= 1
        print("worker_count=%d" % self.worker_count)
        if self.worker_count == 0:
            print("all processes died")
            time.sleep(2)
            self.event.set()

    def get_no_evaluations(self):
        evals = self.no_evaluations
        self.no_evaluations -= 1
        return evals

    def append_point_evaluated(self, point: List, share_to_hyperspaces: List) -> None:
        """Appends a newly evaluated point in a hyperspace into SharedData all_points_evaluated list"""
        self.all_points_evaluated.append(point)
        # logging.info(self.all_points_evaluated)
        if len(share_to_hyperspaces) != 0:
            # self.send_shared_points_to_hyperspaces(point, share_to_hyperspaces)
            for hyperspace in share_to_hyperspaces:
                self.shared_points_to_hyperspaces[hyperspace].append(point)

    def clear_shared_points(self, hyperspaces_no: List):
        """Clears the shared points for a hyperspace because they are alredy inserted into model from all_points_evaluated"""
        for hyperspace in hyperspaces_no:
            self.shared_points_to_hyperspaces[hyperspace] = []

    def get_shared_points(self, hyperspaces_no: List):
        """Returns the shared points for requested hyperspaces list (all hyperspaces in current search space)"""
        points = []
        for hyperspace in hyperspaces_no:
            if len(self.shared_points_to_hyperspaces[hyperspace]) == 0:
                pass
            else:
                for point in self.shared_points_to_hyperspaces[hyperspace]:
                    points.append(point)
                self.shared_points_to_hyperspaces[hyperspace] = []
        return points

    def get_all_points_evaluated(self):
        """Return all points evaluated in whole searc space"""
        # not converting to numpy array here, because SharedData worker can get busy in serving data to other workers.
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
            overlap_lenght = subspace_length * self.overlap / 2
            end = low + subspace_length + overlap_lenght
            parameter_subspaces = []
            parameter_subspaces.append((round(low, 8), round(end, 8)))
            low = low + subspace_length
            for _ in range(1, self.no_subspaces - 1):
                end = low + subspace_length + overlap_lenght
                parameter_subspaces.append(
                    (round(low - overlap_lenght, 8), round(end, 8))
                )  # remove rounding, as there is overlapping subspaces is going on. no need to precise look for hyperspace
                low = low + subspace_length
            end = low + subspace_length
            parameter_subspaces.append((round(low - overlap_lenght, 8), round(end, 8)))
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
                    # to break the loop after first hyperspace for which point belongs, so commenting `break` will
                    # produce a dictionary where multiple hyperspaces are included as key for a point (case of overlapped hyperspaces)
                    # break
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

    def get_all_hyperspaces_in_current_search_space(
        self, bounds_indices: List[Tuple]
    ) -> List:
        """Return all hyperspaces for given search space bound indices"""
        result = []
        for hs_no, hs_indices in enumerate(self.subspace_indices_for_hyperspace):
            flag = True
            for dim, hs_indice in enumerate(hs_indices):
                if (
                    hs_indice >= bounds_indices[dim][0]
                    and hs_indice <= bounds_indices[dim][1]
                ):
                    continue
                else:
                    flag = False
                    break
            if flag:
                result.append(hs_no)
        return result


@ray.remote
class DeployHyperspace:
    """
    This Actor work on dividing the hyperspace as per the algorithm and deploy new worker
    """

    def __init__(
        self,
        objective_function: Callable,
        safe_threshold: float,
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
        self.all_hyperspaces_current_search_space = ray.get(
            self.shared_data.get_all_hyperspaces_in_current_search_space.remote(
                self.bounds_indices
            )
        )
        logging.info(
            "all_hyperspaces_current_search_space : %s",
            self.all_hyperspaces_current_search_space,
        )

        self.kernel = GPy.kern.RBF(
            input_dim=len(self.bounds), variance=2.0, lengthscale=1.0, ARD=True
        )
        # if x is None:
        #     logging.info("x is none")  # self.deploy_hyperspace()
        # else:
        #     logging.info("Starting initial_deploy")
        #     self.initial_deploy()
        self.worker_name = worker_name
        self.shared_data.update_worker_count.remote(True, self.worker_name)

    def set_bounds(self, bounds, bounds_indices):
        logging.info("Updating bounds")
        self.bounds = bounds
        self.bounds_indices = bounds_indices
        # update the hyperspaces_list in current search space
        self.all_hyperspaces_current_search_space = ray.get(
            self.shared_data.get_all_hyperspaces_in_current_search_space.remote(
                self.bounds_indices
            )
        )
        logging.info(
            "all_hyperspaces_current_search_space : %s",
            self.all_hyperspaces_current_search_space,
        )

    def optimization(
        self, x: List[List], y: List[List], safe_hyperspace_no: int,
    ) -> None:
        """ """
        logging.info("optimization started")

        current_hyperspace = safe_hyperspace_no
        new_hyperspace = current_hyperspace

        if x is None:
            # clear shared data points, because they are already contained in all_points
            # so, no duplicate adding
            self.shared_data.clear_shared_points.remote(
                self.all_hyperspaces_current_search_space
            )
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
            # logging.debug(x)
            # logging.debug(y)
            gp = GPy.models.GPRegression(x, y, self.kernel, noise_var=self.noise_var)
            self.shared_data.append_point_evaluated.remote([x[0], y[0]], [])
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

        try:
            evaluation_constraint = ray.get(
                self.shared_data.get_no_evaluations.remote()
            )
            while evaluation_constraint > 0:
                # logging.info("iteration: %s", i)
                logging.info("Evaluation constraint %d", evaluation_constraint)

                # getting and adding shared points to the model
                shared_points = ray.get(
                    self.shared_data.get_shared_points.remote(
                        self.all_hyperspaces_current_search_space
                    )
                )
                logging.info("Got %d shared points", len(shared_points))
                logging.debug(shared_points)
                for point in shared_points:
                    logging.debug(point)
                    x = np.array(point[0])
                    y = np.array(point[1])
                    opt.add_new_data_point(x, y)

                # obtain new query point
                x_next = opt.optimize()
                # Get a measurement from the real system
                y_meas = self.objective_function(x_next)

                logging.debug(
                    "variables state\nx_next: %s\ny_next: %s",
                    np.array_str(x_next),
                    np.array_str(y_meas),
                )

                point_hyperspaces = list(
                    ray.get(self.shared_data.which_hyperspace.remote([x_next]))
                )
                logging.debug("point_hyperspaces : %s", point_hyperspaces)

                # maintain list which contains all the hyperspaces in current search space
                # since which_hyperspace() will return all the hyperspaces in which the point belongs
                # then remove all the hyperspaces that are not belong to current search space
                legit_hyperspaces = set(
                    self.all_hyperspaces_current_search_space
                ).intersection(set(point_hyperspaces))
                logging.debug("legit_hyperspaces : %s", legit_hyperspaces)

                # share point to hyperspaces (hyperspaces for which the point belongs to but not in current search space)
                share_to_hyperspaces = set(point_hyperspaces) - set(
                    self.all_hyperspaces_current_search_space
                )

                logging.info(
                    "Sharing point to hyperspaces %s", list(share_to_hyperspaces)
                )
                self.shared_data.append_point_evaluated.remote(
                    [x_next, y_meas[0]], list(share_to_hyperspaces)
                )

                # CHECK : whether to compare `y_meas` with `threshold` to confirm for safety and add to corresponding
                # hyperspace. Since non-safe point also provides some information about objective function

                if current_hyperspace in point_hyperspaces:
                    logging.info("new_hyperspace = current_hyperspace")
                    new_hyperspace = current_hyperspace
                else:
                    logging.info("new_hyperspace != current_hyperspace")
                    new_hyperspace = random.choice(list(legit_hyperspaces))
                    if y_meas >= self.safe_threshold:
                        break
                    else:
                        new_hyperspace = current_hyperspace

                logging.debug("new_hyperspace: %s", new_hyperspace)

                # Add this to the GP model
                opt.add_new_data_point(x_next, y_meas)

                # get the evaluation_constraint
                evaluation_constraint = ray.get(
                    self.shared_data.get_no_evaluations.remote()
                )
        except:
            logging.debug("Exception occured")
            logging.debug(sys.exc_info()[0])
        finally:
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
        worker_name = "HyperSpace_" + str(new_hyperspace)
        new_node: DeployHyperspace = DeployHyperspace.options(
            name=worker_name, lifetime="detached"
        ).remote(
            self.objective_function,
            self.safe_threshold,
            self.noise_var,
            self.shared_data,
            bounds=ray.get(self.shared_data.get_bounds_from_index.remote(ss2)),
            bounds_indices=ss2,
            worker_name=worker_name,
        )

        new_node.deploy_hyperspace.remote(new_hyperspace)

        bounds = ray.get(self.shared_data.get_bounds_from_index.remote(ss1))
        bounds_indices = ss1
        self.set_bounds(bounds, bounds_indices)
        self.deploy_hyperspace(current_hyperspace)


def initial_deploy(
    safe_set, shared_data: SharedData, objective_function, safe_threshold
) -> None:
    bounds_indices = ray.get(shared_data.get_bounds_indices.remote())
    # Measurement noise
    noise_var = 0.05 ** 2

    if safe_set.shape[0] == 1:
        # if safe set contains only one point, then we have to deploy whole space to a process.

        logging.info("starting")

        # Here we may get more than one hyperspace as the point may belong to overlapped hyperspace.
        # choose only one hyperspace to start the optimization process
        safe_hyperspace_no = random.choice(
            list(ray.get(shared_data.which_hyperspace.remote(safe_set)))
        )

        logging.info("calling optimization for hyperspace %d", safe_hyperspace_no)

        # bounds = ray.get(shared_data.get_bounds_from_index.remote(bounds_indices))

        worker_name = "HyperSpace_" + str(safe_hyperspace_no)
        initial_worker: DeployHyperspace = DeployHyperspace.options(
            name=worker_name, lifetime="detached"
        ).remote(
            objective_function,
            safe_threshold,
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
        worker_name = "HyperSpace_" + str(new_hyperspace)
        new_worker: DeployHyperspace = DeployHyperspace.options(
            name=worker_name, lifetime="detached"
        ).remote(
            objective_function,
            safe_threshold,
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
        address="172.20.46.18:8888",
        _redis_password="5241590000000000",
        namespace="dbo_overlapped",
    )

    time_str = time.strftime("%d_%b_%H_%M_%S_", time.localtime())
    logging.basicConfig(
        level=logging.DEBUG,
        filename="logs/" + time_str + "driver.log",
        filemode="w",
        format="%(process)d-%(levelname)s-%(funcName)s-%(message)s",
    )

    objective_function_name = "bird_function"
    objective_function = bird_function
    bounds = [(-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi)]
    safe_threshold = -35.0
    no_subspaces = 4
    overlap = 0.15
    evaluation_constraint = 20
    # parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)
    shared_data = SharedData.options(name="SharedData").remote(
        bounds, no_subspaces, evaluation_constraint, overlap
    )

    # Initial safe set
    x0 = np.zeros((1, len(bounds)))  # safe point at zero
    # x0 = np.array([[-2.5]]) # 1D single safe point
    # x0 = np.array([[4, 4]])  # 2D single safe point
    # x0 = np.array([[-2.5], [6.5]]) # 1D multiple safe points

    initial_deploy(x0, shared_data, objective_function, safe_threshold)
    await shared_data.work_done.remote()

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
        "./logs/" + time_str + objective_function_name + "_overlapped_log.csv",
        index=False,
        header=False,
    )

    report = "Number of points evaluated : %d\n" % no_points_evaluated
    no_unsafe_evaluation = points_df.y[points_df.y < -35].count()
    report += "Number of unsafe evaluations : %d\n" % no_unsafe_evaluation
    optimum_value = points_df["y"].max()
    optimum_value_at = points_df.iloc[points_df["y"].idxmax()][0 : points_df.shape[1]]
    report += "Optimization results\ny = %f\nat\n%s" % (
        optimum_value,
        optimum_value_at.to_string(),
    )
    with open(
        "./logs/" + time_str + objective_function_name + "_overlapped_log.txt", "w",
    ) as res_file:
        res_file.write(report)

    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
