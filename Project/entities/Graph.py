from entities.Product import Product
import copy


class Graph:
    """ Implements a graph of instances of Products, it is not necessary to work indexing the
    products but the instantiated objects has to be used"""

    def __init__(self, g: 'Graph' = None):
        if g:
            self.graph = copy.deepcopy(g.graph)
            self.node_list = g.node_list.copy()
            self.n_nodes = len(self.node_list)
            self.ids = copy.deepcopy(g.ids)
        else:
            self.graph = {}
            self.node_list = []
            self.n_nodes = 0
            self.ids = set()

    def add_edge(self, src, dest, weight) -> None:
        """Add an edge to the graph """
        if not isinstance(src, Product):
            raise Exception("source of edge Value Error")
        if not isinstance(dest, Product):
            raise Exception("destination of edge Value Error")
        if src not in self.graph:
            raise Exception("source of edge is not in the graph")
        if dest not in self.graph:
            raise Exception("dest of edge is not in the graph")

        self.graph[src].append((dest, weight))
        self.ids.add(src.id)

    def add_node(self, item) -> None:
        """Add a Product node to the graph """
        if not isinstance(item, Product):
            raise Exception("item is not a product Value Error")
        if item in self.graph:
            print("Item already in the graph")
            return

        self.graph[item] = list()
        self.node_list.append(item)
        self.n_nodes += 1

    def get_child_nodes(self, father_node) -> list:
        """ get all child nodes given a father """
        if not isinstance(father_node, Product):
            raise Exception("node is not a product Value Error")
        if father_node not in self.graph:
            print("No match")
            return []

        return self.graph[father_node]

    def get_all_nodes(self) -> list:
        """ return all nodes composing the graph """
        return [*self.graph].copy()  # this notation make a list of all keys of dict graph

    def printGraph(self) -> None:
        """ print adjacency list representation """
        for (src, node_list) in self.graph.items():
            print(f"--- node: {src} ---")
            for (node, weight) in node_list:
                print(f"\t( {src} )--[w:{weight:.3f}]-—>( {node} )")

    def get_adjacency_matrix(self) -> list:
        """ get adjacency matrix representation """
        matrix = []

        for (src, node_list) in self.graph.items():
            row = [0.0 for _ in range(len(self.graph.items()))]
            for (node, weight) in node_list:
                row[node.id - 1] = weight

            matrix.append(row)

        return matrix

    def get_neighbours(self, node: Product) -> list:
        """ get a list of all the neighbours of a given node """
        if node not in self.graph:
            raise Exception("source node is not in the graph")

        return [pair[0] for pair in self.graph[node]]  # each pair is a (dest, weight) tuple

    def get_not_dangling_nodes(self) -> list:
        """ get a list of all the nodes having out-degree > 0 """
        keysList = list(self.graph.keys())
        return [keysList[i] for i in range(len(keysList)) if len(self.graph[keysList[i]]) > 0]

    # TODO: set_weight and get_weight methods could be improved if a dict of dicts is used instead of a dicts of lists
    #  for self.graph, like self.betas in LearnableGraph. Changes are minimal.

    def set_weight(self, src: Product, dest: Product, weight) -> None:
        """Set edge's weight """
        if src.id not in self.ids:
            raise Exception("source of edge is not in the graph")
        if dest.id not in self.ids:
            raise Exception("dest of edge is not in the graph")

        for i in range(len(self.graph[src])):
            if self.graph[src][i][0] == dest:
                self.graph[src][i] = (self.graph[src][i][0], weight)
                break

    def get_weight(self, src: Product, dest: Product, fromId = False):

        """Set edge's weight """
        if src.id not in self.ids:
            raise Exception("source of edge is not in the graph")
        if dest.id not in self.ids:
            raise Exception("dest of edge is not in the graph")

        if fromId:
            src = [p for p in self.graph.keys() if p.id == src.id][0]
            dest = [p for p in self.graph.keys() if p.id == dest.id][0]

        for i in range(len(self.graph[src])):
            if self.graph[src][i][0] == dest:
                return self.graph[src][i][1]
