import math

import numpy as np
import matplotlib.pyplot as plt

from Graph import Graph
from LearnableGraph import LearnableGraph


class OnlineWeightsLearner:
    # Given the 2 graphs we compute the absolute value of the difference between the probabilities.
    # We return the mean.
    @staticmethod
    def get_total_error(graph1: Graph, graph2: Graph):
        if graph1.n_nodes == graph2.n_nodes:

            adj_matrix_1 = graph1.get_adjacency_matrix()
            adj_matrix_2 = graph2.get_adjacency_matrix()

            error = 0
            total_edges = 0
            for i in range(graph1.n_nodes):
                for j in range(graph2.n_nodes):
                    if not math.isclose(adj_matrix_1[i][j], 0.0):
                        total_edges += 1
                        error += abs(adj_matrix_1[i][j] - adj_matrix_2[i][j])
            return error / total_edges

    @staticmethod
    def __influence_episode(graph: LearnableGraph, seeds, true_graph: Graph):
        binomial_matrix = np.zeros([graph.n_nodes, graph.n_nodes], dtype = np.float16)

        candidates = graph.get_not_dangling_nodes()  # gets the coordinates of nodes having out-degree > 0

        for candidate in candidates:  # each candidate is a node having out degree > 0
            neighbours = graph.get_neighbours(node = candidate)

            for neighbour in neighbours:
                true_probability = true_graph.get_weight(candidate, neighbour, fromId = True)
                reward = np.random.binomial(1, true_probability)
                binomial_matrix[candidate.id - 1][neighbour.id - 1] = reward

        live_edges = binomial_matrix > 0

        activated = []
        new_activated = seeds

        while new_activated:
            activated = new_activated + activated
            new_activated = []
            for active in activated:  # initially just the seed node is activated
                for neighbour in graph.get_neighbours(node = active):
                    if live_edges[active.id - 1][neighbour.id - 1] and not (
                            neighbour in new_activated or neighbour in activated):  # active edges must be considered just once
                        graph.update_beta_values(src = active, dest = neighbour,
                                                 param = 'alpha')  # if the edge activating corresponds to the one with a seed, update beta
                        new_activated.append(neighbour)
                    else:
                        if not (neighbour in new_activated or neighbour in activated):
                            graph.update_beta_values(src = active, dest = neighbour, param = 'beta')

        return len(activated) - len(seeds)

    @staticmethod
    def __monte_carlo_spread(graph: LearnableGraph, seeds, max_repetitions):
        nodes_activation_probabilities = np.zeros(graph.n_nodes, dtype = np.float16)

        for _ in range(max_repetitions):

            # np.random.rand generates a matrix of random (0, 1) numbers!
            # We want a live_edges to work with
            live_edges = graph.get_adjacency_matrix() > np.random.rand(graph.n_nodes,
                                                                       graph.n_nodes)  # generate live-edge graph

            activated = []
            new_activated = seeds
            while new_activated:
                activated = new_activated + activated
                new_activated = []  # keeps track of newly activated nodes, to know when to stop exploration.

                for active in activated:  # initially just the seed is activated

                    for neighbour in graph.get_neighbours(node = active):  # consider the nodes of the active edges

                        # if the edge is active and the dest node has not been already considered, increment counter
                        if live_edges[active.id - 1][neighbour.id - 1] and not (
                                neighbour in new_activated or neighbour in activated):
                            nodes_activation_probabilities[neighbour.id - 1] += 1
                            new_activated.append(neighbour)

        nodes_activation_probabilities = nodes_activation_probabilities / max_repetitions

        return np.mean(nodes_activation_probabilities)  # measures the magnitude of the spread

    @staticmethod
    def __choose_seeds_from_sampling(graph: LearnableGraph, monte_carlo_repetitions):
        candidates = graph.get_not_dangling_nodes()  # gets the coordinates of nodes having out-degree > 0

        # Retrieve for each of them alpha and beta, compute the deviation and update probability
        for candidate in candidates:  # each candidate is a node having out degree > 0
            neighbours = graph.get_neighbours(node = candidate)

            for neighbour in neighbours:
                # update probability given a new Beta sample
                alpha, beta, _ = graph.get_beta_parameters(src = candidate, dest = neighbour)
                sample = np.random.beta(a = alpha, b = beta)
                graph.set_weight(src = candidate, dest = neighbour, weight = sample)

        seeds = OnlineWeightsLearner.__monte_carlo_sampling(graph, monte_carlo_repetitions)

        return seeds

    # The function returns the best possible seed (the one maximizing the influence) given a certain graph

    @staticmethod
    def __monte_carlo_sampling(graph, monte_carlo_repetitions):
        spreads = []
        best_node = None
        nodes = graph.get_all_nodes()
        best_spread = 0

        for node in nodes:  # evaluate the spread of each node as a seed
            spread = OnlineWeightsLearner.__monte_carlo_spread(graph = graph, seeds = [node],
                                                               max_repetitions = monte_carlo_repetitions)
            if spread > best_spread:
                best_spread = spread
                best_node = node

            spreads.append(best_spread)

        return [best_node]

    @staticmethod
    def estimate_weights(true_graph: Graph, simulations, monte_carlo_repetitions):
        # Copy the original graph and convert to a learnable one -> all weights are initially set to 0.5
        graph = LearnableGraph(g = true_graph)

        x_list = []
        x2_list = []
        y_list = []  # ideal error
        y2_list = []

        # total_error = 0.0

        # Main procedure
        for r in range(simulations):
            print("Iteration: " + str(r + 1) + "/" + str(simulations), end = "")
            # epsilon = (1 - r / monte_carlo_repetitions) ** 2
            seeds = OnlineWeightsLearner.__choose_seeds_from_sampling(graph = graph,
                                                                      monte_carlo_repetitions = monte_carlo_repetitions)
            OnlineWeightsLearner.__influence_episode(graph = graph,
                                                     seeds = seeds,
                                                     true_graph = true_graph)

            error = OnlineWeightsLearner.get_total_error(graph, true_graph)
            # total_error += error
            # print(" " + str(total_error))

            x_list.append(r)
            x2_list.append(r)
            # y_list.append(total_error)
            y_list.append(error)
            y2_list.append(0)
            print("", end = "\r")

        print("", end = "")

        plt.plot(x_list, y_list, label = 'Bandit Approximation', color = 'tab:blue', linestyle = '-')
        plt.plot(x2_list, y2_list, label = 'Ideal 0 Value', color = 'tab:orange', linestyle = '--')
        plt.title("Unknown Activation Probabilities - Approximation Error")
        plt.ylabel("Approximation Error")
        plt.xlabel("Time")
        plt.yticks(np.arange(0, max(y_list), 0.1))
        plt.legend()

        plt.show()

        return graph
