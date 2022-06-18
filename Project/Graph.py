from Product import Product


class Graph:
    """ Implements a graph of instances of Products, it is not necessary to work indexing the
    products but the instantiated objects has to be used"""

    def __init__(self):
        self.graph = {}
        self.node_list = []

    def add_edge(self, src, dest, weight):
        """Add an edge to the graph """
        if not isinstance(src, Product):
            raise Exception("source of edge Value Error")
        if not isinstance(dest, Product):
            raise Exception("destination of edge Value Error")

        self.graph[src].append((dest, weight))

    def add_node(self, item):
        """Add a Product node to the graph """
        if not isinstance(item, Product):
            raise Exception("item is not a product Value Error")
        if item in self.graph:
            print("Item already in the graph")
            return

        self.graph[item] = list()
        self.node_list.append(item)

    def get_child_nodes(self, father_node):
        """ get all child nodes given a father """
        if not isinstance(father_node, Product):
            raise Exception("node is not a product Value Error")
        if father_node not in self.graph:
            print("No match")
            return []

        return self.graph[father_node]

    def get_all_nodes(self):
        """ return all nodes composing the graph """
        return [*self.graph]  # this notation make a list of all keys of dict graph

    def printGraph(self):
        """ print adjacency list representation """
        for (src, node_list) in self.graph.items():
            print(f"--- node: {src} ---")
            for (node, weight) in node_list:
                print(f"\t( {src} )--[w:{weight:.3f}]-—>( {node} )")

    def get_adjacency_matrix(self):
        """ print adjacency matrix representation """
        matrix = []

        for (src, node_list) in self.graph.items():
            row = [0.0 for _ in range(len(self.graph.items()))]
            for (node, weight) in node_list:
                row[node.id - 1] = weight

            matrix.append(row)

        return matrix
