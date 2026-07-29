import sys
import numpy as np

sys.path.extend(['../'])
from graph_baseline import tools
'''
num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
'''
num_node = 29
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(0, 1), (0, 5), (0, 12), (0, 13), (1, 9), (1, 12), (1, 16), (2, 19), (2, 24), (2, 25), (3, 4),
                    (3, 11), (3, 15), (3, 18), (4, 8), (4, 14), (4, 15), (5, 6), (5, 12), (5, 13), (6, 7), (6, 12),
                    (6, 13), (6, 26), (7, 8), (7, 14), (7, 15), (7, 26), (8, 14), (8, 15), (9, 10), (9, 16), (9, 17), 
                    (9, 20), (9, 27), (10, 11), (10, 17), (10, 27), (11, 17), (11, 18), (11, 21), (11, 27), (12, 13), 
                    (13, 14), (13, 26), (14, 15), (14, 26), (16, 17), (16, 20), (16, 23), (16, 24), (17, 18), (17, 19),
                    (17, 20), (17, 21), (17, 22), (17, 23), (17, 27), (18, 21), (18, 22), (18, 25), (19, 22), 
                    (19, 23), (19, 24), (19, 25),(20, 23),(21, 22),(22, 25),(23, 24), (28, 26), (28, 27), (28, 10)]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
