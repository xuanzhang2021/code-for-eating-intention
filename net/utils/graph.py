import numpy as np
from face_mesh_connections import FACEMESH_TESSELATION

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose_face',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2

        elif layout == 'openpose_face_25_points':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                             (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                             (12, 13), (13, 14), (14, 8), (13, 14), (14, 8),
                             (0, 17), (1, 17), (9, 17), (11, 17), (1, 18), (11, 18),
                             (17, 18), (1, 16), (7, 16), (11, 15), (12, 15), (16, 15),
                             (18, 15), (19, 15), (12, 15), (16, 15), (16, 18),
                             (16, 21), (16, 22), (16, 23), (24, 21), (24, 22), (24, 23),
                             (19, 20), (12, 20), (7, 20), (8, 20), (14, 20), (7, 19),
                             (16, 19), (21, 22), (22, 23), (1, 21), (2, 21), (3, 21),
                             (5, 23), (6, 23), (7, 23), (3, 24), (4, 24), (5, 24), (12, 19)]
            self.edge = self_link + neighbor_link
            self.center = 16

        elif layout == 'openpose_face_19_points':
            self.num_node = 19
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6),
                             (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1)]
            self.edge = self_link + neighbor_link
            self.center = 10

        elif layout == 'openpose_face_18_points':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 4), (0, 11), (1, 4), (1, 11), (1, 8), (1, 15),  (2, 3), (2, 7), (2, 10),
                             (2, 14), (2, 17), (3, 7), (3, 14), (4, 11), (4, 5), (5, 14),  (6, 7), (6, 11),
                             (6, 13), (7, 14), (8, 9), (8, 16), (8, 1), (8, 11), (8, 15), (9, 10), (9, 12), (9, 13),
                             (10, 17), (10, 16), (10, 14), (11, 12), (11, 8), (12, 15), (12, 13), (12, 5),
                             (12, 9), (13, 14), (13, 9), (13, 6),  (13, 17), (9, 16)]
            self.edge = self_link + neighbor_link
            self.center = 9

            self.edge = self_link + neighbor_link
            self.center = 10
        # idx = np.array([1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54, 57], dtype=np.int64)   # 20个点
        # idx = np.array([1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54], dtype=np.int64)   # 19个点
        # idx = np.array([1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54, 57, 49, 53, 55, 59], dtype=np.int64)   # 24个点(包含嘴巴周围的点)
        # idx = np.array([1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54, 57, 49, 53, 55, 59, 7, 9], dtype=np.int64)   # 26个点(包含嘴巴周围和下巴周围的点)

        elif layout == 'openpose_face_20_points':   # pose 连接所有的点
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6), (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1), (0, 19), (1, 19), (2, 19),
                             (3, 19), (4, 19), (5, 19), (6, 19), (7, 19), (8, 19), (9, 19), (10, 19), (11, 19),
                             (12, 19), (13, 19), (14, 19), (15, 19), (16, 19), (17, 19), (18, 19)]

            self.edge = self_link + neighbor_link
            self.center = 10

        elif layout == 'openpose_face_20_points_with_pose':    # 19个点+一个pose
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6), (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1), (2, 19), (10, 19)]
            # (0, 19), (2, 19),(4, 19), (5, 19), (8, 19), (10, 19), (16, 19), (18, 19) acc=0.9435
            # (2, 19), (10, 19), (17, 19) acc=0.9565 acc=0.9457
            # (10, 19)  acc=0.9348  
            #  (2, 19), (10, 19), (17, 19), (1, 19), (3, 19) acc=0.9348
            # (2, 19), (10, 19) acc=0.9435  acc=0.9630  竖中轴线
            # (2, 19), (17, 19)
            # (1, 19), (3, 19)

            self.edge = self_link + neighbor_link
            self.center = 10

        elif layout == 'openpose_face_20_points_nopose':
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6), (6, 13), (6, 15), (7, 8), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1), (19, 2)]
            # , (19, 16),(19, 17), (19, 18)

            self.edge = self_link + neighbor_link
            self.center = 10

        elif layout == 'openpose_face_21_points_with_pose':    # 20个点+一个pose
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6), (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1), (19, 2), (19, 16),(19, 17), (19, 18), (20, 2), (20, 10)]

            self.edge = self_link + neighbor_link
            self.center = 10
        # idx = np.array([1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54, 57, 49, 53, 55, 59], dtype=np.int64)   # 24个点(包含嘴巴周围的点)
        elif layout == 'openpose_face_25_points_with_pose':    # 24个点+一个pose
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6), (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1), (19, 2), (19, 16),(19, 17), (19, 18), 
                             (20, 16), (20, 9), (20, 17), (20, 23), (21, 17), (21, 11), (21, 18), (21, 22),
                             (22, 18), (22, 19), (22, 2), (23, 16), (23, 19), (23, 2),
                             (24, 2), (24, 10)]

            self.edge = self_link + neighbor_link
            self.center = 10

        elif layout == 'openpose_face_27_points_with_pose':    # 27个点+一个pose
            self.num_node = 27
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6), (6, 13), (6, 15), (7, 8), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1), (19, 2), (19, 16),(19, 17), (19, 18), 
                             (20, 16), (20, 9), (20, 17), (20, 23), (21, 17), (21, 11), (21, 18), (21, 22),
                             (22, 18), (22, 19), (22, 2), (23, 16), (23, 19), (23, 2),
                             (24, 16), (24, 19), (24, 2), (25, 18), (25, 19), (25, 2),
                             (26, 2), (26, 10)]

            self.edge = self_link + neighbor_link
            self.center = 10
        # idx = np.array([1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54, 57, 49, 53, 55, 59, 7, 9, 27, 30], dtype=np.int64)   # 28个点(包含嘴巴周围和下巴周围的点,鼻梁点，增加中轴线)
        elif layout == 'openpose_face_29_points_with_pose':    # 28个点+一个pose  已经运行的非常好，但是graph的连接有点乱
            self.num_node = 29
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6), (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1), (19, 2), (19, 16),(19, 17), (19, 18), 
                             (20, 16), (20, 9), (20, 17), (20, 23), (21, 17), (21, 11), (21, 18), (21, 22),
                             (22, 18), (22, 19), (22, 2), (23, 16), (23, 19), (23, 2),
                             (24, 16), (24, 19), (24, 2), (25, 18), (25, 19), (25, 2),
                             (26, 27), (26, 12), (26, 15), (26, 10), (27, 10), (27, 9), (27, 11),   # 有的结果是基于(26，25)
                             (28, 26), (28, 27), (28, 10) ]  # 最后一个点代表pose

            self.edge = self_link + neighbor_link
            self.center = 27  # 原来脸部30的点，鼻尖
        
        elif layout == 'openpose_face_29_points_with_pose_k4':    # 28个点+一个pose  使用K-nearest neighborhood  K=4
            self.num_node = 29
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 5), (0, 12), (0, 13), (1, 9), (1, 12), (1, 16), (2, 19), (2, 24), (2, 25), (3, 4),
                                (3, 11), (3, 15), (3, 18), (4, 8), (4, 14), (4, 15), (5, 6), (5, 12), (5, 13), (6, 7), (6, 12),
                                (6, 13), (6, 26), (7, 8), (7, 14), (7, 15), (7, 26), (8, 14), (8, 15), (9, 10), (9, 16), (9, 17), 
                                (9, 20), (9, 27), (10, 11), (10, 17), (10, 27), (11, 17), (11, 18), (11, 21), (11, 27), (12, 13), 
                                (13, 14), (13, 26), (14, 15), (14, 26), (16, 17), (16, 20), (16, 23), (16, 24), (17, 18), (17, 19),
                                (17, 20), (17, 21), (17, 22), (17, 23), (17, 27), (18, 21), (18, 22), (18, 25), (19, 22), 
                                (19, 23), (19, 24), (19, 25),(20, 23),(21, 22),(22, 25),(23, 24), (28, 26), (28, 27), (28, 10) ]  # 最后一个点代表pose
            # 删掉（2, 22) (19, 21), 
            self.edge = self_link + neighbor_link
            self.center = 27  # 原来脸部30的点，鼻尖

        elif layout == 'openpose_face_28_points_no_pose':    # 28个点+一个pose
            self.num_node = 28
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6), (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1), (19, 2), (19, 16),(19, 17), (19, 18), 
                             (20, 16), (20, 9), (20, 17), (20, 23), (21, 17), (21, 11), (21, 18), (21, 22),
                             (22, 18), (22, 19), (22, 2), (23, 16), (23, 19), (23, 2),
                             (24, 16), (24, 19), (24, 2), (25, 18), (25, 19), (25, 2),
                             (26, 27), (26, 12), (26, 15), (26, 10), (27, 10), (27, 9), (27, 11),   # 有的结果是基于(26，25)
                            ]  

            self.edge = self_link + neighbor_link
            self.center = 27  # 原来脸部30的点，鼻尖

        elif layout == 'openpose_face_29_points_with_pose_k4_improve':    # 28个点+一个pose  使用K-nearest neighborhood  K=4
            self.num_node = 29
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 5), (0, 12), (1, 9), (1, 12), (1, 16), (2, 19), (2, 24), (2, 25), (3, 4),
                                (3, 11), (3, 15), (3, 18), (4, 8), (4, 15), (5, 6), (5, 12), (5, 13), (6, 7), (6, 12),
                                (6, 13), (6, 26), (7, 8), (7, 14), (7, 15), (7, 26), (8, 14), (8, 15), (9, 10), (9, 16), (9, 17), 
                                (9, 20), (9, 27), (10, 11), (10, 17), (10, 27), (11, 17), (11, 18), (11, 21), (11, 27), (12, 13), 
                                (13, 26), (14, 15), (14, 26),  (16, 20), (16, 23), (16, 24), (17, 19),
                                (17, 20), (17, 21), (17, 22), (17, 23), (18, 21), (18, 22), (18, 25), (19, 22), 
                                (19, 23), (19, 24), (19, 25),(20, 23),(21, 22),(22, 25),(23, 24), 
                                (1, 24), (3, 25), (26, 27), (9, 12), (11, 15), (28, 26), (28, 27), (28, 10) ]  # 最后一个点代表pose
            self.edge = self_link + neighbor_link
            self.center = 27  # 原来脸部30的点，鼻尖

        elif layout == 'openpose_face_33_points_with_pose':    # 28个点+一个pose
            self.num_node = 33
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6), (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1), (19, 2), (19, 16),(19, 17), (19, 18), 
                             (20, 16), (20, 9), (20, 17), (20, 23), (21, 17), (21, 11), (21, 18), (21, 22),
                             (22, 18), (22, 19), (22, 2), (23, 16), (23, 19), (23, 2),
                             (24, 16), (24, 19), (24, 2), (25, 18), (25, 19), (25, 2),
                             (26, 27), (26, 12), (26, 15), (26, 10), (27, 10), (27, 9), (27, 11), 
                             (28, 20), (28, 17), (28, 31), (29, 17), (29, 21), (29, 30),
                             (30, 22), (30, 19), (31, 19), (31, 23),
                             (32, 26), (32, 27), (32, 10)]  # 最后一个点代表pose

            self.edge = self_link + neighbor_link
            self.center = 27  # 原来脸部30的点，鼻尖



        elif layout == 'openpose_face_69_points_with_pose':   # 没有连接confidence score
            self.num_node = 69
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
                             (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
                             # 左眉毛
                             (17, 18), (18, 19), (19, 20), (20, 21),
                             # 右眉毛
                             (22, 23), (23, 24), (24, 25), (25, 26),
                             # 鼻梁
                             (27, 28), (28, 29), (29, 30),
                             # 鼻翼
                             (31, 32), (32, 33), (33, 34), (34, 35),
                             # 左眼（闭合环）
                             (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
                             # 右眼（闭合环）
                             (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
                             # 外嘴唇（闭合环）
                             (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
                             (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
                             # 内嘴唇（闭合环）
                             (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60),
                             (68, 8), (68, 33) ]  # pose点连接鼻梁和下巴
            self.edge = self_link + neighbor_link
            self.center = 30  # 鼻尖

        elif layout == 'openpose_RTMPose_20_points':   # 没有连接confidence score
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # 躯干与肩的连接
                (0, 5), (0, 6),
                # 左上肢骨架
                (5, 7), (7, 9),  # 肩-肘-腕
                (9, 95), (9, 99), (9, 103), (9, 107), (9, 111),  # 腕-五指尖
                # 右上肢骨架
                (6, 8), (8, 10),  # 肩-肘-腕
                (10, 116), (10, 120), (10, 124), (10, 128), (10, 132),  # 腕-五指尖
                # 个可选：左右肩直接连（肩宽线）
                (5, 6) ]
            self.edge = self_link + neighbor_link
            self.center = 1  # 鼻尖

        elif layout == 'openpose_facemesh_478_points':   # 没有连接confidence score
            self.num_node = 478
            self_link = [(i, i) for i in range(self.num_node)]
            triangles = [
                tuple(int(v.split("/")[0]) - 1 for v in line.split()[1:])
                for line in open("canonical_face_model.obj")
                if line.startswith("f ")
            ]
            print(f"Triangles: {len(triangles)}")
            # === 生成无向邻边 ===
            edge_set = set()
            for (i1, i2, i3) in triangles:
                edge_set.update([
                    tuple(sorted((i1, i2))),
                    tuple(sorted((i2, i3))),
                    tuple(sorted((i3, i1)))
                ])

            neighbor_link = list(edge_set)
            # neighbor_link = list(FACEMESH_TESSELATION)   
            # 佳实践：双向连接（以防 Tessellation 是单向）
            neighbor_link = list(set(neighbor_link + [(b, a) for (a, b) in neighbor_link]))
            self.edge = self_link + neighbor_link
            self.center = 4  # 鼻尖


        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))   # 初始化adjacency 矩阵，为0
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':    # 所有连接都在同一个图中
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':  # 不同hop的连接放在不同的图中
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':    # 根据节点与中心节点的距离，分为三类连接
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):   # 对有向图邻接矩阵的“列归一化  A_norm  = A * D^(-1)
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):    # 无向图邻接矩阵的对称归一化 A_norm  = D^(-1/2) * A * D^(-1/2)
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
