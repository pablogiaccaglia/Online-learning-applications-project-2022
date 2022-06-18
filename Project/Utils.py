from random import randint
from typing import Union
import pandas as pd
import numpy as np
from Graph import Graph
from random import uniform


def get_probabilities(quantity, padding=0.5):
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


def random_fully_connected_graph(products=[], padding=0.1):
    """ Generate a fully connected random graph with the given Products
        the weights will sum to 1-padding """
    graph = Graph()

    for prod in products:
        graph.add_node(prod)

    for prod in products:
        weights = get_probabilities(4, padding=padding)  # to change weights change here
        child_products = products.copy()
        child_products.remove(prod)

        for i, prod_child in enumerate(child_products):
            graph.add_edge(prod, prod_child, weights[i])

    return graph


def new_alpha_function(saturation_speed=1, max_value=1, activation=0.1):
    """ When using the alpha functions remember to clip them to 0 """
    return lambda x: (-1 + 2 / (1 + np.exp(- saturation_speed * (x - activation)))) * max_value


def noise_matrix_alpha(max_reduction=0.15, max_global_influence=0.05, n_user=3, n_product=5):
    """ return a 2D list: one row for user and column for products
        it returns the multiplier for a stochastic reduction on alpha function """
    global_influence = uniform(0.0, max_global_influence)  # set day trend
    return [[1 - uniform(0, max_reduction) - global_influence for c in range(n_product)] for r in range(n_user)]


def no_noise_matrix(n_user=3, n_product=5):
    return [[1 for c in range(n_product)] for r in range(n_user)]


def get_prettyprint_array(arr: Union[np.ndarray, list], row_labels: list[str] = None, col_labels: list[str] = None):
    return pd.DataFrame(arr, columns = col_labels, index = row_labels)