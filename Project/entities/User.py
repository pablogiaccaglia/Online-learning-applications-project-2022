import numpy as np

from entities.LearnableGraph import LearnableGraph


class User:
    def __init__(self,
                 id,
                 reservation_prices,
                 lmbda,
                 weighted_graph,
                 alpha_functions,
                 exp_number_purchase,
                 ):
        self.id = id
        self.reservation_prices = reservation_prices
        self.lmbda = lmbda
        self.weighted_graph = weighted_graph  # pass here the full weighted graph associated with the user
        self.alpha_functions = alpha_functions  # list of alpha_function specifing how the user react to a specific campaign
        self.exp_number_purchase = exp_number_purchase
        self.is_estimated_graph = True if isinstance(weighted_graph, LearnableGraph) else False

    def __recursive_visit(self, graph, root_id, node, node_prob, prob_list, purchased_set, secondary_list,
                          expected_profits,
                          user, exp_number_noise, debug,
                          on_estimated_graph = False):
        """ DFS visit of every node: the expected profit array is updated for every successful payment,
            in case of not purchase the navigation is interrupted """

        if node.id not in secondary_list or node in purchased_set:
            # stop in case this is not the second product or user already bought this
            return

        # update probability trace of the path leading to this point (all weights between root and this node & eventual lambda effects )
        prob_list.append(node_prob)

        if on_estimated_graph:
            if node.id == secondary_list[1]:
                prob_list.append(user.lmbda)

        # decide whether to buy
        if user.reservation_prices[node.id - 1] >= node.price:

            prob_to_be_here = np.prod(np.array(prob_list))  # cumulative probability to be in this node of the graph
            expected_profits[root_id] += prob_to_be_here * node.price * user.exp_number_purchase[
                node.id - 1] * exp_number_noise[node.id - 1]  # compute profit of this purchase
            purchased_set.add(node)

            if debug:
                print(
                        f"user {user.id} with rp {user.reservation_prices[node.id - 1]} BOUGHT {user.exp_number_purchase} units of product {node.id} for {node.price}, probability:{prob_to_be_here}")

            for child, weight in graph.get_child_nodes(
                    node):  # for every child of the node try to start a recursive visit
                self.__recursive_visit(
                        graph = graph,
                        root_id = root_id,
                        node = child,
                        node_prob = weight,
                        prob_list = prob_list,
                        purchased_set = purchased_set,
                        secondary_list = node.secondary_list,
                        expected_profits = expected_profits,
                        user = user,
                        exp_number_noise = exp_number_noise,
                        debug = debug
                )

        prob_list.remove(node_prob)

        if on_estimated_graph:
            if node.id == secondary_list[1]:
                prob_list.remove(user.lmbda)

    def set_graph(self, weighted_graph):
        self.weighted_graph = weighted_graph

    def expected_profit(self, exp_number_noise, debug=False):
        nodes = self.weighted_graph.get_all_nodes()
        expected_profits = [0, 0, 0, 0, 0]

        # apply dfs starting from every product page
        for i, node in enumerate(nodes):
            purchased_set = set()  # keep track of the purchased products
            prob_list = [1]  # list with the trace of probabilities collected

            self.__recursive_visit(graph=self.weighted_graph,
                                   root_id=i,
                                   node=node,
                                   node_prob=1,
                                   prob_list=prob_list,
                                   purchased_set=purchased_set,
                                   secondary_list=[node.id, -1],
                                   expected_profits=expected_profits,
                                   user=self,
                                   debug=debug,
                                   exp_number_noise=exp_number_noise,
                                   on_estimated_graph = self.is_estimated_graph
                                   )

            # self._simulation_done = True
            if debug:
                print(f"\t\t$$$ Expected profit of visited products: {expected_profits[node.id - 1]} $$$")
                print(f"\t\t--- End navigation started from {node} ---\n")
                # print(f"\t\t--- Expected profit of visit: {self.value_per_click(node.id)} ---\n")

        return expected_profits

    def change_graph(self, weighted_graph):
        self.weighted_graph = weighted_graph
        self.is_estimated_graph = True if isinstance(weighted_graph, LearnableGraph) else False
