import itertools
import lzma
import pickle

import pandas
import scipy
import winsound

import numpy as np
from tqdm import tqdm


def beep():
    """Produces a pleasing beep"""
    for x in range(3):
        winsound.Beep(2000 + 100 * x, 100)
    winsound.Beep(2300, 700)


def run_scan(fun, var_axes, filename=None, trials=1, beep_on_done=True):
    """
    Runs a scan over variables of a given simulation. This is intended to generate raw (trackfile) outputs that are
    then processed with the other functions; `fun` will usually take simulation parameters and return a Pandas database
    of a trackfile.

    :param fun: The function to be run
    :param var_axes: Tuple of arrays of values for each variable altered. These should correspond to the inputs of `fun`
    :param filename: .pkl file to save the results to after the scan finishes. Optional. Providing a filename ending in .lzma will apply LZMA compression.
    :param trials: Number of repeated trials to run for each variable combination. Optional.
    :param beep_on_done: Whether to beep when the scan finishes
    :return: List of tuples, each one consisting of the values associated with a result, followed by the result
    """
    to_run = list(itertools.product(*var_axes)) * trials
    results = list()
    for x in tqdm(to_run):
        result = fun(*x)
        results.append(tuple(list(x) + [result]))
    if filename is not None:
        if filename.endswith(".lzma"):
            # Save with pkl + lzma
            with lzma.open(filename, "wb") as file:
                pickle.dump(results, file)
        else:
            # Assume a pickle file
            with open(filename, "wb+") as file:
                pickle.dump(results, file)
    if beep_on_done:
        beep()
    return results


def data_to_map(data):
    """
    Converts the output of `run_scan` into a mapping from values to results

    :param data: Results in the format of results from `run_scan`
    :return: A dict mapping tuples of variable values to lists of result obtained with those values
    """
    result = dict()
    for row in data:
        x = tuple(row[:-1])
        r = row[-1]
        if x not in result:
            result[x] = list()
        result[x].append(r)
    return result


def calc_quantity(fun, data):
    """
    Computes a quantity based on scan data

    :param fun: Function representing the quantity to be calculated. Should take a trackfile dataframe and return the quantity
    :param data: Results in the format of results from `run_scan`
    :return: A dict mapping tuples of variable values to the mean and standard deviation of the quantity
    """
    mapped = data_to_map(data)
    result = dict()
    for k in mapped:
        values = [fun(r) for r in mapped[k]]
        result[k] = np.mean(values), np.std(values)
    return result


def qmap_to_meshgrid(mesh, quantity_map):
    """
    Used for visualizing 2D scans. Converts a quantity mapping to a mesh suitable for passing into `plot_surface` or similar.

    :param mesh: Tuple of meshx and meshy. Result of calling np.meshgrid on the variable ranges
    :param quantity_map: Dict output from calc_quantity
    :return: Mesh of the mean of the quantity.
    """
    meshx, meshy = mesh
    result = np.empty_like(meshx)
    for i in range(len(meshx)):
        for j in range(len(meshx[0])):
            x = meshx[i][j], meshy[i][j]
            result[i][j] = quantity_map[x][0]
    return result


def qmap_to_arrays(array, quantity_map):
    """
    Used for visualizing 1D scans. Converts a quantity mapping into arrays of the mean and standard deviation.
    The output from this function can be passed directly into `errorbar`

    :param array: Variable range array originally used for the scan
    :param quantity_map: Dict output from calc_quantity
    :return: Tuple consisting of the input array, the array of the means, and the array of the stds.
    """
    means, stds = zip(*[quantity_map[(x,)] for x in array])
    means = np.array(list(means))
    stds = np.array(list(stds))
    return array, means, stds


def qmap_to_dataframe(quantity_map, input_names, quantity_name):
    """
    Converts a quantity mapping into a Pandas dataframe, including both the mean and standard deviation

    :param quantity_map: Dict output from calc_quantity
    :param input_names: The names to use for the input values
    :param quantity_name: The name to use for the quantity
    :return: Pandas dataframe with columns corresponding to the input variables and the quantity
    """
    rows = list()
    for k, v in quantity_map.items():
        rows.append(list(k) + list(v))
    return pandas.DataFrame(rows, columns=input_names + [quantity_name, quantity_name + "_std"])


def qmaps_to_dataframe(quantity_maps, input_names, quantity_names):
    """
    Converts and compiles multiple quantity mappings into a Pandas dataframe with the mean and standard deviation of each

    :param quantity_maps: Sequence of dicts output from calc_quantity
    :param input_names: The names to use for the input values
    :param quantity_names: List of the names to use for the quantities
    :return: Pandas dataframe with columns corresponding to the input variables and the quantities given
    """
    rows = list()
    for k in quantity_maps[0]:
        rows.append(list(k) + [q[k][0] for q in quantity_maps] + [q[k][1] for q in quantity_maps])
    return pandas.DataFrame(rows, columns=input_names + quantity_names + [q + "_std" for q in quantity_names])


num_iter = 0


# Pretty minimize function
def minimize(func, start, **kwargs):
    """Wrapper for `scipy.optimize.minimize` that produces an indicator of iteration count and current value"""
    global num_iter
    num_iter = 0

    def callback(intermediate_result):
        global num_iter
        num_iter += 1
        print(f"{num_iter:4} {intermediate_result.fun:.5e}", end="\r")

    print("iter value")
    result = scipy.optimize.minimize(func, start, callback=callback, **kwargs)
    print()
    return result
