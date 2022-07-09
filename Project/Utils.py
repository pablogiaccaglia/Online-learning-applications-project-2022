import random
from random import randint
from typing import Union
import pandas as pd
from random import uniform
import numpy as np
from Graph import Graph
from random import uniform

from LearnableGraph import LearnableGraph
from Graph import Graph

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
                                   num_of_neighbours,
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


def noise_matrix_alpha(max_reduction=0.15, max_global_influence=0.05, n_user=3, n_product=5):
    """ return a 2D list: one row for user and column for products
        it returns the multiplier for a stochastic reduction on alpha function """
    global_influence = uniform(0.0, max_global_influence)  # set day trend
    # generate a random contraction and add a random addition
    return [
        [1 - uniform(0, max_reduction) - global_influence + uniform(-0.25, 0.25)
         for c in range(n_product)] for r in range(n_user)
    ]


def no_noise_matrix(n_user=3, n_product=5):
    return [[1 for c in range(n_product)] for r in range(n_user)]


def get_prettyprint_array(arr, row_labels = None, col_labels = None):
    return pd.DataFrame(arr, columns = col_labels, index = row_labels)
