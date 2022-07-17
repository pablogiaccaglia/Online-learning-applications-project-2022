import os
import random
from enum import Enum
from random import randint
from time import sleep
from typing import Union

import pandas as pd
import numpy as np
from random import uniform

from entities.LearnableGraph import LearnableGraph
from entities.Graph import Graph
import inspect


def get_probabilities(quantity, padding):
    """it return a random list of probabilities that sum to 1-padding"""

    if int(quantity) <= 0:
        raise Exception("quantity Value Error")
    if float(padding) < 0 or float(padding) >= 1:
        raise Exception("padding Value Error")

    probabilities = []
    random_samples = np.array([0] * quantity)

    for i in range(quantity):
        random_samples[i] = float(randint(1, 100))

    normalizer = (1 - padding) / np.sum(random_samples)

    for i in range(quantity):
        probabilities.append(random_samples[i] * normalizer)

    # add numerical noise to first element to try to ensure sum to 1-padding
    probabilities[0] = probabilities[0] + ((1 - padding) - np.sum(probabilities))

    return probabilities


def random_fully_connected_graph(products = [], padding = 0.1):
    """ Generate a fully connected random graph with the given Products
        the weights will sum to 1-padding """
    return __get_graph_specify_neighbours(products = products,
                                          padding = padding,
                                          num_of_neighbours = len(products) - 1,
                                          weighted = True,
                                          known = True)


def random_fully_connected_unknown_graph(products = []):
    """ Generate a fully connected random unknown weighted graph
        with the given Products"""
    return __get_graph_specify_neighbours(products = products,
                                          num_of_neighbours = len(products) - 1,
                                          padding = None,
                                          weighted = False,
                                          known = False)


def get_ecommerce_graph(products = [], padding = 0.1):
    return __get_graph_specify_neighbours(products = products,
                                          num_of_neighbours = 2,
                                          padding = padding,
                                          weighted = True,
                                          known = True)


def __get_graph_specify_neighbours(products: list,
                                   num_of_neighbours: Union[int, list],
                                   padding = None,
                                   weighted = True, known = True):
    graph = Graph() if known else LearnableGraph()

    if known and not padding:
        padding = [0.1 for _ in range(len(products))]

    if known and isinstance(padding, float):
        padding = [padding for _ in range(len(products))]

    if isinstance(num_of_neighbours, int):
        num_of_neighbours = [num_of_neighbours for _ in range(len(products))]

    for n in num_of_neighbours:
        if n > len(products) - 1:
            raise ValueError("number of neighbours cannot exceed number of products - 1")

    for prod in products:
        graph.add_node(item = prod)

    for i, prod in enumerate(products):

        if known:
            if weighted:
                weights = get_probabilities(num_of_neighbours[i], padding = padding[i])  # to change weights change here
            else:
                weights = [0.0 for _ in range(len(num_of_neighbours))]
        child_nodes = products.copy()
        child_nodes.remove(prod)

        for _ in range(len(products) - 1 - num_of_neighbours[i]):
            child_nodes.pop(random.randrange(len(child_nodes)))

        for k, prod_child in enumerate(child_nodes):
            if known:
                graph.add_edge(prod, prod_child, weights[k])
            else:
                graph.add_edge(prod, prod_child, None)

    return graph


def new_alpha_function(saturation_speed = 1, max_value = 1, activation = 0.1):
    """ When using the alpha functions remember to clip them to 0 """
    return lambda x: (-1 + 2 / (1 + np.exp(- saturation_speed * (x - activation)))) * max_value


def noise_matrix_alpha(max_reduction = 0.1, max_global_influence = 0.1, n_user = 3, n_product = 5):
    """ return a 2D list: one row for user and column for products
        it returns the multiplier for a stochastic reduction on alpha function """
    global_influence = uniform(0.0, max_global_influence)  # set day trend
    # generate a random contraction and add a random addition
    return [
        [1 + uniform(-0.1, 0.1) + random.gauss(0, 0.4 / 3)
         for c in range(n_product)] for r in range(n_user)
    ]


def no_noise_matrix(n_user = 3, n_product = 5):
    return [[1 for c in range(n_product)] for r in range(n_user)]


def table_metadata(n_prod, n_users, avail_budget):
    _col_labels = [str(budget) for budget in avail_budget]

    _row_label_rewards = []
    _row_labels_dp_table = ['0']
    for i in range(1, n_prod + 1):
        for j in range(1, n_users + 1):
            # Cij -> campaign i and user j
            _row_label_rewards.append("C" + str(i) + str(j))
            _row_labels_dp_table.append("+C" + str(i) + str(j))
    return _row_label_rewards, _row_labels_dp_table, _col_labels


def get_prettyprint_array(arr, row_labels = None, col_labels = None):
    return pd.DataFrame(arr, columns = col_labels, index = row_labels)


def clear_output(wait = True, keep_scroll_back = False):
    # Waiting for 1 second to clear the screen
    if wait:
        sleep(0.5)

    if keep_scroll_back:
        """# Clearing the Screen keeping scroll back
        # posix is os name for linux or mac
        if os.name == 'posix':
            os.system('clear')
        # else screen will be cleared for windows
        else:
            os.system('cls')"""

    else:
        os.system('cls' if os.name == 'nt' else "printf '\033c'")


def get_colors(type = 1):
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    CB91_Red = '#fe2c54'
    CB91_Orange = '#fe6d2c'

    colors = [CB91_Blue, CB91_Amber, CB91_Purple, CB91_Green, CB91_Pink, CB91_Violet, CB91_Red, CB91_Orange]

    colors2 = ['#78C850',  # Grass
               '#F08030',  # Fire
               '#6890F0',  # Water
               '#A8B820',  # Bug
               '#A8A878',  # Normal
               '#A040A0',  # Poison
               '#F8D030',  # Electric
               '#E0C068',  # Ground
               '#EE99AC',  # Fairy
               '#C03028',  # Fighting
               '#F85888',  # Psychic
               '#B8A038',  # Rock
               '#705898',  # Ghost
               '#7038F8',  # Dragon
               ]

    return colors.copy() if type == 1 else colors2.copy()


def retrieve_name(var):
    """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def regret_upper_bound_gp_ts(t, arms, sets, input_dimension, K):
    # https://trovo.faculty.polimi.it/01papers/accabi2018when.pdf page 6
    #        C       sqrt(        N   *   log  (C   *        N^3          *       M^3        )  *  (      log   ( N    *  M   )^ (d+1)    )
    if t == 0:
        t = 1
    return K * sets * np.sqrt(
            t * np.log(sets * np.power(t, 3) * np.power(arms, 3)) * np.power(np.log(t * arms),
                                                                             input_dimension + 1))

def regret_upper_bound_gts(t, arms, sets, delta_min, K):

    if t == 0:
        t = 1

    return K * arms * sets * np.log(t) / delta_min


def compute_delta(banditX, banditY, clairvoyantX, clairvoyantY, campaigns):

    min_deltas = [0 for _ in range(campaigns)]

    for c in range(campaigns):

        min_delta = np.inf

        for idx, y in enumerate(banditY[c][0]):

            x = banditX[idx]
            clairvoyantMean = np.interp(x, clairvoyantX, clairvoyantY[c][0])
            delta = np.abs(clairvoyantMean - y)

            if delta < min_delta:
                min_delta = delta

        min_deltas[c] = min_delta

    return sum(min_deltas)


class BanditNames(Enum):
    CusumGPUCB1Learner = 'CUSUM-GP-UCB1'
    CusumGTSLearner = 'CUSUM-GTS'
    GPTS_Learner = 'GP-TS'
    GPUCB1_Learner = 'GP-UCB1'
    GTS_Learner = 'GTS'
    SwGPUCB1_Learner = 'SW-GP-UCB1'
    SwGTSLearner = 'SW-GTS'
