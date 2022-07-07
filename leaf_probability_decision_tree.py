import random
import graphviz

import pdb

class LeafProbabilityDecisionTree():

    def __init__(self, variable_list, variable_domain_list,  nutrients, leaf_threshold, leaf_length, random_seed = None):

        random.seed(random_seed)

        self.tree, self.node_number = self.generate_structure(nutrients, leaf_threshold, leaf_length, 0)

    def generate_structure(self, nutrients, leaf_threshold, leaf_length, node_number):

        left_nutrients = random.uniform(0, nutrients)
        right_nutrients = nutrients - left_nutrients

        # print((left_nutrients, right_nutrients))
        if left_nutrients < leaf_threshold:
            left_node = Leaf(node_number + 1, leaf_length)
            used_number = node_number + 1
        else:
            left_node, used_number = self.generate_structure(left_nutrients, leaf_threshold, leaf_length, node_number + 1)

        if right_nutrients < leaf_threshold:
            right_node = Leaf(used_number + 1, leaf_length)
            used_number += 1
        else:
            right_node, used_number = self.generate_structure(right_nutrients, leaf_threshold, leaf_length, used_number + 1)

        return Node(node_number, left_node, right_node, None), used_number

    def print_structure(self):
        structure_string = self.structure_to_string(self.tree)

        print(structure_string)

    def structure_to_string(self, tree):

        left = tree.left_node
        if isinstance(left, Leaf):
            left_string = '[%s]' % left.leaf_number
        else:
            left_string = self.structure_to_string(left)

        right = tree.right_node
        if isinstance(right, Leaf):
            right_string = '[%s]' % right.leaf_number
        else:
            right_string = self.structure_to_string(right)

        return '%s:[%s, %s]' % (tree.node_number, left_string, right_string)


    def plot_structure(self):
        G = graphviz.Digraph(format='png')
        G.attr('node', shape='circle')

        node_count = self.count_nodes(self.tree, 0)
        for i in range(node_count):
            G.node(str(i), str(i))

        edge_list = []
        edge_list = self.extract_edges(self.tree, edge_list)

        for edge in edge_list:
            G.edge(str(edge[0]), str(edge[1]))

        G.render('output')
        G.view()

    def count_nodes(self, tree, count):

        left = tree.left_node

        if isinstance(left, Leaf):
            count += 1
        else:
            count = self.count_nodes(left, count)

        right = tree.right_node
        if isinstance(right, Leaf):
            count += 1
        else:
            count = self.count_nodes(right, count)

        return count + 1

    def extract_edges(self, tree, edge_list):

        left = tree.left_node
        if isinstance(left, Leaf):
            edge_list.append((tree.node_number, left.leaf_number))
        else:
            edge_list = self.extract_edges(left, edge_list)
            edge_list.append((tree.node_number, left.node_number))

        right = tree.right_node
        if isinstance(right, Leaf):
            edge_list.append((tree.node_number, right.leaf_number))
        else:
            edge_list = self.extract_edges(right, edge_list)
            edge_list.append((tree.node_number, right.node_number))

        return edge_list


class Node():

    def __init__(self, node_number, left_node, right_node, inequality_string):

        self.node_number = node_number
        self.inequality_string = inequality_string
        self.left_node = left_node
        self.right_node = right_node

class Leaf():

    def __init__(self, leaf_number, leaf_length):
        self.leaf_number = leaf_number
        self.prob_vector = [0]*leaf_length

if __name__ == '__main__':

    lpdt = LeafProbabilityDecisionTree([], [], 2.0, 1.0, 6, None)
    lpdt.print_structure()

    lpdt.plot_structure()
