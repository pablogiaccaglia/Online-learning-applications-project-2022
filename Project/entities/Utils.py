import os
import random
from random import randint
from time import sleep
from typing import Union

import pandas as pd
import numpy as np
from random import uniform

from entities.LearnableGraph import LearnableGraph
from entities.Graph import Graph


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


def noise_matrix_alpha(max_reduction = 0.15, max_global_influence = 0.05, n_user = 3, n_product = 5):
    """ return a 2D list: one row for user and column for products
        it returns the multiplier for a stochastic reduction on alpha function """
    global_influence = uniform(0.0, max_global_influence)  # set day trend
    # generate a random contraction and add a random addition
    return [
        [1 - uniform(0, max_reduction) - global_influence + uniform(-0.25, 0.25)
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



def get_colors():
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

    return colors.copy()


