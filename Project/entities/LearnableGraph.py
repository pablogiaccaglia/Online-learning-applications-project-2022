from entities.Graph import Graph
from entities.Product import Product
import numpy as np
from entities.Beta import Beta


class LearnableGraph(Graph):

    def __init__(self, g: 'Graph' = None):
        super().__init__(g)

        self.betas = {}

        if g:
            self.__clear_weights()

            for node in self.graph.keys():
                self.betas[node] = {}
                for neighbor in self.get_neighbours(node):
                    self.betas[node][neighbor] = Beta()

    def add_node(self, item):
        """Add a Product node to the graph """
        super().add_node(item = item)
        self.betas[item] = {}  # dict -> (dest, weight)

    def add_edge(self, src, dest, weight = None):
        """Add an edge to the graph """
        if not isinstance(src, Product):
            raise Exception("source of edge Value Error")
        if not isinstance(dest, Product):
            raise Exception("destination of edge Value Error")

        if not weight:
            w = 0.5  # no knowledge weight
            self.betas[src][dest] = Beta()
        else:
            w = weight
            self.betas[src][dest] = None  # this is not safe, find out better approach

        super().add_edge(src = src, dest = dest, weight = w)

    def get_betas_matrix(self):
        """ get betas matrix representation """
        n_nodes = len(self.node_list)
        betas = np.empty((n_nodes, n_nodes), dtype = object)

        for (src, beta_dict) in self.betas.items():
            row = np.empty(n_nodes, dtype = object)
            for (node, beta) in beta_dict.items():
                row[node.id - 1] = beta

            betas[src.id - 1] = row

        return betas

    def update_beta_values(self, src: Product, dest: Product, param: str) -> None:
        if src not in self.graph:
            raise Exception("source of edge is not in the graph")
        if dest not in self.graph:
            raise Exception("dest of edge is not in the graph")

        if self.betas[src][dest]:
            if param == 'alpha':
                self.betas[src][dest].a += 1
            if param == 'beta':
                self.betas[src][dest].b += 1

            self.betas[src][dest].played += 1

    def get_beta_parameters(self, src: Product, dest: Product):
        """given a source and a destination node, returns the beta parameters
           associated with the corresponding graph's edge"""
        if src not in self.graph:
            raise Exception("source of edge is not in the graph")
        if dest not in self.graph:
            raise Exception("dest of edge is not in the graph")

        if self.betas[src][dest]:
            return self.betas[src][dest].a, self.betas[src][dest].b, self.betas[src][dest].played

    def __clear_weights(self):
        """conversion to known to unknown weights graph"""
        for key in self.graph.keys():
            for i in range(len(self.graph[key])):
                self.graph[key][i] = (self.graph[key][i][0], 0.5)
