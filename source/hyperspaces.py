import GPy
import numpy as np
import pandas as pd
import safeopt
import os
import logging
from multiprocessing import Process, Queue
from typing import List, Tuple, Dict, Final, Callable
from objective_functions import (
    bird_function,
    BIRD_FUNCTION_BOUNDS,
    BIRD_FUNCTION_THRESHOLD,
)

# logging.basicConfig(filename="app.log", filemode="w", level=logging.DEBUG)
# logging.basicConfig(format="%(process)d-%(levelname)s-%(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create handlers
f_handler = logging.FileHandler("file.log", "w")
# f_handler.setLevel(logging.INFO)
# Create formatters and add it to handlers
f_format = logging.Formatter(
    "%(process)d - %(levelname)6s - %(funcName)20s() : %(message)s"
)
f_handler.setFormatter(f_format)
# Add handlers to the logger
logger.addHandler(f_handler)

all_parameter_subspaces = []
subspace_indices_for_hyperspace = []
subspaces_deployment_status = []
points_evaluated_in_hyperspace = {}
evaluation_constraint = 50
# lock = Lock()


def create_hyperspaces(
    parameter_spaces: List[Tuple], no_subspaces: int
) -> List[List[Tuple]]:
    """Creates hyperspaces for the given parameters by dividing each of them into given number of subspaces.

    Args:
        parameter_spaces:
            List of tuple containing two items i.e: lower bound and upper bound for each search parameter.
        no_subspaces:
            How many number of subspaces to create for a search parameter.

    Returns:
        hyperspaces:
            Set of all possible combinations of subspaces.
    """
    global subspaces_deployment_status
    global all_parameter_subspaces
    global subspace_indices_for_hyperspace
    # to divide each parameter space into given number of subspaces
    for parameter_space in parameter_spaces:
        low, high = parameter_space
        subspace_length = abs(high - low) / no_subspaces
        parameter_subspaces = []
        for i in range(no_subspaces):
            end = low + subspace_length
            parameter_subspaces.append((round(low, 8), round(end, 8) - 0.00000001))
            low = end
        all_parameter_subspaces.append(parameter_subspaces)

    rows = len(all_parameter_subspaces)  # no_parameters
    columns = len(all_parameter_subspaces[0])  # no_subspaces

    # initializing deployed status for each subsapce
    for i in range(rows):
        each_parameter_subspaces = []
        for _ in range(columns):
            # for each subspace maintain these 3 states to determine its deployment status
            # [is this subspace deployed (True/False), associated with any subspace (True/False),
            #  hyperspace number (None-not deployed with any hyperspace/int-deployed with that hyperspace)]
            each_parameter_subspaces.append(False)
        subspaces_deployment_status.append(each_parameter_subspaces)

    # no_hyperspaces = no_subspaces ** no_parameters
    hyperspaces_list = []  # contains all possible combinations of subspaces
    for _ in range(columns ** rows):
        hyperspaces_list.append([])
        subspace_indices_for_hyperspace.append([])

    for row in range(rows):
        repeat = columns ** (rows - row - 1)
        for column in range(columns):
            item = all_parameter_subspaces[row][column]
            start = column * repeat
            for times in range(columns ** row):
                for l in range(repeat):
                    hyperspaces_list[start + l].append(item)
                    subspace_indices_for_hyperspace[start + l].append(column)
                start += columns * repeat
    return hyperspaces_list


def which_hyperspace(x: List[List], hyperspaces_list: List[List[Tuple]]) -> Dict:
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
    # logger.debug("hyperspace_list lenght: %s", len(hyperspaces_list))

    for point in x:
        # logger.debug("point: %s", point)
        for i, hyperspace in enumerate(hyperspaces_list):
            # logger.debug("hyperspace : %s = %s", i, hyperspace)
            belongs_flag = True
            for dimension, value in enumerate(point):
                # logger.debug("\ndimension:%s\nvalue:%s", dimension, value)
                if (
                    value >= hyperspace[dimension][0]
                    and value <= hyperspace[dimension][1]
                ):
                    # logger.info("inside if condition, so just continue")
                    continue
                else:
                    belongs_flag = False
                    # logger.info("not belongs to this hyperspace")
                    break
            if belongs_flag:
                if i not in safe_hyperspaces:
                    safe_hyperspaces[i] = [point]
                else:
                    safe_hyperspaces[i].append(point)
                break
    return safe_hyperspaces


def split_search_space(ss, hs1, hs2):
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
            ds = subspaces_deployment_status[param_index]
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

    if reverse_flag:
        return ss2, ss1
    else:
        return ss1, ss2


def get_bounds_from_index(hyperspace_bounds: List[Tuple]) -> List[Tuple]:
    """ """
    bounds = []
    for parameter_no, bound in enumerate(hyperspace_bounds):
        parameter_subspaces = all_parameter_subspaces[parameter_no]
        low = parameter_subspaces[bound[0]][0]
        high = parameter_subspaces[bound[1]][1]
        bounds.append((low, high))
    return bounds


def all_points_evaluated_df(
    evaluated_points_queue: Queue, dimension: int
) -> pd.DataFrame:
    all_points_evaluated = []
    while not evaluated_points_queue.empty():
        hs_no, x_i, y_i = evaluated_points_queue.get()
        row = [hs_no]
        row.extend(x_i)
        row.append(y_i[0])
        all_points_evaluated.append(row)

    label = ["hyperspace"]
    label.extend(["x" + str(i) for i in range(dimension)])
    label.append("y")
    return pd.DataFrame(all_points_evaluated, columns=label)


def optimization(
    x: List[List],
    y: List[List],
    bounds: List[Tuple],
    kernel: GPy.kern,
    objective_function: Callable,
    safe_threshold: float,
    noise_var: float,
    safe_hyperspace_no: int,
    hyperspaces_list: List[List[Tuple]],
    evaluated_points_queue: Queue,
) -> None:
    """ """
    # global evaluation_constraint
    logger.info("started")

    current_hyperspace = safe_hyperspace_no
    new_hyperspace = current_hyperspace

    x = np.array(x)
    # The statistical model of our objective function
    if y is None:
        gp = GPy.models.GPRegression(
            x, objective_function(x), kernel, noise_var=noise_var
        )
    else:
        y = np.array(y)
        gp = GPy.models.GPRegression(x, y, kernel, noise_var=noise_var)

    # The optimization routine
    opt = safeopt.SafeOptSwarm(
        gp, BIRD_FUNCTION_THRESHOLD, bounds=bounds, threshold=0.2
    )
    # opt = safeopt.SafeOpt(
    #     gp,
    #     parameter_set,
    #     safe_threshold,
    #     lipschitz=None,
    #     threshold=0.2
    # )

    # {key: hyperspace_no, value: evaluated unsfae points}
    # converting to `np.arry()` or `list` because `opt.x` is of `ObsAr` type.
    if y is None:
        evaluated_points = {
            current_hyperspace: [[np.array(x_i) for x_i in opt.x], list(opt.y)]
        }
        for x_i, y_i in zip(
            evaluated_points[current_hyperspace][0],
            evaluated_points[current_hyperspace][1],
        ):
            # row = [current_hyperspace]
            # row.extend(x_i)
            # row.append(y_i[0])
            evaluated_points_queue.put([current_hyperspace, x_i, y_i])
        y = []
    else:
        evaluated_points = {}

    logger.debug(
        "optimization for \nhyperspace_no: %s\nbounds: %s\nx: %s\ny: %s\nevaluated_points: %s\n",
        current_hyperspace,
        bounds,
        list(x),
        y,
        evaluated_points,
    )

    try:
        for i in range(10):
            logger.info("iteration: %s", i)

            # if evaluation_constraint <= 0:
            #     break

            # obtain new query point
            x_next = opt.optimize()
            # Get a measurement from the real system
            y_meas = objective_function(x_next)

            logger.debug(
                "variables state\nx_next: %s\ny_next: %s",
                np.array_str(x_next),
                np.array_str(y_meas),
            )

            new_hyperspace_dict = which_hyperspace([x_next], hyperspaces_list)
            # logger.debug("new_hyperspace_dict: %s\n", new_hyperspace_dict)
            new_hyperspace = list(new_hyperspace_dict)[0]
            logger.debug("new_hyperspace: %s\n", new_hyperspace)
            evaluated_points_queue.put([new_hyperspace, x_next, y_meas[0]])

            # CHECK : whether to compare `y_meas` with `threshold` to confirm for safety and add to corresponding
            # hyperspace. Since non-safe point also provides some information about objective function.
            if new_hyperspace in evaluated_points.keys():
                evaluated_points[new_hyperspace][0].append(x_next)
                evaluated_points[new_hyperspace][1].append(y_meas[0])
            else:
                evaluated_points[new_hyperspace] = [[x_next], [y_meas[0]]]

            if y_meas >= safe_threshold and new_hyperspace != current_hyperspace:
                break
                # return current_hyperspace, new_hyperspace, evaluated_points

            # Add this to the GP model
            opt.add_new_data_point(x_next, y_meas)

            # opt.plot(100, plot_3d=False)
    finally:
        return current_hyperspace, new_hyperspace, evaluated_points


def deploy_hyperspace(
    hyperspace_no: int,
    hyperspace_bounds: List[Tuple],
    kernel: GPy.kern,
    objective_function: Callable,
    safe_threshold: float,
    noise_var: float,
    hyperspaces_list: List[List[Tuple]],
    evaluated_points_queue: Queue,
) -> None:
    """ """
    global points_evaluated_in_hyperspace
    logger.info("starting")
    logger.debug("deploy new hyperspace : %s", hyperspace_no)

    if hyperspace_no not in points_evaluated_in_hyperspace.keys():
        logger.error("No safe points given for hyperspace: %s", hyperspace_no)
        return

    # check for leaf node
    is_leaf_flag = True
    for bound in hyperspace_bounds:
        if bound[0] != bound[1]:
            is_leaf_flag = False
    if is_leaf_flag:
        logger.info("LEAF node deployment")
        return

    x, y = points_evaluated_in_hyperspace[hyperspace_no]
    logger.debug("points_evaluated : \nx : %s \ny : %s", x, y)

    logger.info("getting bounds from indices")
    bounds = get_bounds_from_index(hyperspace_bounds)
    logger.debug("search space bounds : %s", bounds)

    logger.info("calling for optimization")
    current_hyperspace, new_hyperspace, evaluated_points = optimization(
        x,
        y,
        bounds,
        kernel,
        objective_function,
        safe_threshold,
        noise_var,
        hyperspace_no,
        hyperspaces_list,
        evaluated_points_queue,
    )

    logger.debug(
        "data recieved from forked process \ncurrent_hyperspace : %s \nnew_hyperspace : %s \nevaluated_points : \n%s",
        current_hyperspace,
        new_hyperspace,
        evaluated_points,
    )

    logger.info("adding evaluated points global dictionary")
    for k, v in evaluated_points.items():
        if k in points_evaluated_in_hyperspace.keys():
            for i, value in enumerate(v):
                for point in value:
                    points_evaluated_in_hyperspace[k][i].append(point)
        else:
            points_evaluated_in_hyperspace[k] = v

    if current_hyperspace == new_hyperspace:
        logger.info("current_hyperspace == new_hyperspace")
        return

    # logger.info("NOT waiting for forked process to join")
    hs1 = subspace_indices_for_hyperspace[current_hyperspace]
    hs2 = subspace_indices_for_hyperspace[new_hyperspace]
    logger.debug("\nhyperspace_1 : %s \nhyperspace_2 : %s", hs1, hs2)
    logger.info("Splitting the search space for both hyperspaces")
    ss1, ss2 = split_search_space(hyperspace_bounds, hs1, hs2)
    logger.debug(
        "\nbounds_indices : %s \nsearch_space_1 : %s \nsearch_space_2 : %s",
        hyperspace_bounds,
        ss1,
        ss2,
    )

    logger.info("forking deploy_hyperspace for search_space_2")
    p = Process(
        target=deploy_hyperspace,
        args=(
            new_hyperspace,
            ss2,
            kernel,
            objective_function,
            safe_threshold,
            noise_var,
            hyperspaces_list,
            evaluated_points_queue,
        ),
    )
    p.start()

    logger.info("calling deploy_hyperspace for search_space_1")
    deploy_hyperspace(
        current_hyperspace,
        ss1,
        kernel,
        objective_function,
        safe_threshold,
        noise_var,
        hyperspaces_list,
        evaluated_points_queue,
    )
    p.join()


def initial_deploy(
    x: List[List],
    bounds: List[Tuple],
    bounds_indices: List[Tuple],
    kernel: GPy.kern,
    objective_function: Callable,
    safe_threshold: float,
    noise_var: float,
    hyperspaces_list: List[List[Tuple]],
) -> None:
    """ """
    global points_evaluated_in_hyperspace
    evaluated_points_queue = Queue()
    dimension = x.shape[1]
    if x.shape[0] == 1:
        # if safe set contains only one point, then we have to deploy whole space to a process.
        logger.info("starting")
        safe_hyperspace_no = list(which_hyperspace(x, hyperspaces_list))[0]

        logger.info("calling for optimization")
        bounds = get_bounds_from_index(bounds_indices)
        current_hyperspace, new_hyperspace, evaluated_points = optimization(
            x,
            None,
            bounds,
            kernel,
            objective_function,
            safe_threshold,
            noise_var,
            safe_hyperspace_no,
            hyperspaces_list,
            evaluated_points_queue,
        )
        logger.debug(
            "data recieved from forked process \ncurrent_hyperspace : %s \
                \nnew_hyperspace : %s \nevaluated_points : \n%s",
            current_hyperspace,
            new_hyperspace,
            evaluated_points,
        )
        logger.info("adding evaluated points global dictionary")
        for k, v in evaluated_points.items():
            points_evaluated_in_hyperspace[k] = v

        # call split search space method
        hs1 = subspace_indices_for_hyperspace[current_hyperspace]
        hs2 = subspace_indices_for_hyperspace[new_hyperspace]
        logger.debug("\nhyperspace_1 : %s \nhyperspace_2 : %s", hs1, hs2)
        logger.info("Splitting the search space for both hyperspaces")
        ss1, ss2 = split_search_space(bounds_indices, hs1, hs2)
        logger.debug(
            "\nbounds_indices : %s \nsearch_space_1 : %s \nsearch_space_2 : %s",
            bounds_indices,
            ss1,
            ss2,
        )

        logger.info("forking new process -- deploy_hyperspace for search_space_2")
        p = Process(
            target=deploy_hyperspace,
            args=(
                new_hyperspace,
                ss2,
                kernel,
                objective_function,
                safe_threshold,
                noise_var,
                hyperspaces_list,
                evaluated_points_queue,
            ),
        )
        p.start()

        logger.info("calling deploy_hyperspace for search_space_1")
        deploy_hyperspace(
            current_hyperspace,
            ss1,
            kernel,
            objective_function,
            safe_threshold,
            noise_var,
            hyperspaces_list,
            evaluated_points_queue,
        )
        p.join()
        points_df = all_points_evaluated_df(evaluated_points_queue, dimension)
        points_df.to_csv("results.csv", index=False)
    elif x.shape[0] != 0:
        # if safe set is not empty, deploy corresponding hyperspaces for given points in safe set.
        safe_hyperspaces = which_hyperspace(x, hyperspaces_list)
        # deploy_hyperspace(safe_hyperspaces, hyperspaces)
