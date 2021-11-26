
from typing import List, Tuple


def create_hyperspaces(
    parameter_spaces: List[Tuple],
    no_subspaces: int
) -> List[List[Tuple]]:
    """Creates hyperspaces for the given parameters by dividing each of them 
    into given number of subspaces.

    Args:
        parameter_spaces:
            List of tuple containing two items i.e: lower bound and upper bound 
            for each search parameter.
        no_subspaces:
            How many number of subspaces to create for a search parameter.

    Returns:
        None
    """
    all_parameter_subspaces = []
    # to divide each parameter space into given number of subspaces
    for parameter_space in parameter_spaces:
        low, high = parameter_space
        subspace_length = abs(high - low)/no_subspaces
        parameter_subspaces = []
        for i in range(no_subspaces):
            parameter_subspaces.append((low, low + subspace_length))
            low = low + subspace_length
        # print(parameter_subspaces)
        all_parameter_subspaces.append(parameter_subspaces)
    # print(all_parameter_subspaces)

    rows = len(all_parameter_subspaces)  # no_parameters
    columns = len(all_parameter_subspaces[0])  # no_subspaces
    # no_hyperspaces = no_subspaces ** no_parameters
    hyperspaces = []  # contains all possible combinations of subspaces
    for _ in range(columns**rows):
        hyperspaces.append([])

    for row in range(rows):
        repeat = columns ** (rows-row-1)
        for column in range(columns):
            item = all_parameter_subspaces[row][column]
            start = column * repeat
            for times in range(columns**row):
                for l in range(repeat):
                    hyperspaces[start+l].append(item)
                start += columns * repeat
    return hyperspaces
