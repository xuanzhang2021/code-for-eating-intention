import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

# 定义节点数量
num_nodes = 19

# 定义自连接边（可以在绘图时忽略，因为自环在可视化中不明显）
self_link = [(i, i) for i in range(num_nodes)]

# 定义邻接边
neighbor_link = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6),
    (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
    (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
    (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
    (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
    (3, 11), (9, 16), (5, 1), (12, 1)
]

# 创建图对象
G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(neighbor_link)

# 设置节点标签（可选）
labels = {i: str(i) for i in range(num_nodes)}

# 使用 spring_layout 布局算法自动计算节点位置
pos = nx.spring_layout(G, k=0.5, iterations=50)

# 绘制图形
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edgelist=neighbor_link, width=2, alpha=0.6, edge_color='gray')
nx.draw_networkx_labels(G, pos, labels, font_size=12)

# 显示图形
plt.title("Graph of 'openpose_face_19_points'", fontsize=16)
plt.axis('off')
plt.show()

data = np.load('data_preprocessing/BIWI_test/point19_BIWI_test_data_20241002.npy')  # 数据形状：(4999, 2, 1, 19, 1)
sample = data[0]  # 形状：(2, 1, 19, 1)
sample = np.squeeze(sample)  # 形状：(2, 19)
x_coords = sample[0]  # 形状：(19,)
y_coords = -sample[1]  # 形状：(19,)
pos = {i: (x_coords[i], y_coords[i]) for i in range(19)}

# 6. 绘制图形
# 定义邻接边（根据您的邻接边定义）
neighbor_link = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6),
    (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
    (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
    (13, 14), (14, 15), (15, 3), (16, 1), (16, 2), (16, 13), (17, 9),
    (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
    (3, 11), (9, 16), (5, 1), (12, 1)
]

# 创建图对象
G = nx.Graph()
G.add_nodes_from(range(19))
G.add_edges_from(neighbor_link)

# 设置节点标签（可选）
labels = {i: str(i) for i in range(19)}

# 绘制图形
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edgelist=neighbor_link, width=2, alpha=0.6, edge_color='gray')
nx.draw_networkx_labels(G, pos, labels, font_size=12)

# 显示图形
plt.title("Graph of 'openpose_face_19_points' with Landmark Positions", fontsize=16)
plt.axis('off')
plt.show()


# 将 pos 中的坐标提取为 numpy 数组，形状：(19, 2)
positions = np.array([pos[i] for i in range(19)])  # 形状：(19, 2)

# 为每个点找到其在欧氏空间中最近的 5 个邻居
# 计算所有点之间的距离矩阵，形状：(19, 19)
dist_matrix = distance.cdist(positions, positions, 'euclidean')

neighbor_links = []

for i in range(19):
    # 获取除自身外的其他点的索引
    indices = np.array([j for j in range(19) if j != i])

    # 获取对应的距离
    distances = dist_matrix[i, indices]

    # 根据距离排序，得到最近的 5 个邻居
    nearest_indices = indices[np.argsort(distances)][:5]

    # 添加邻接边（无向图，所以添加一次即可）
    for j in nearest_indices:
        if (i, j) not in neighbor_links and (j, i) not in neighbor_links:
            neighbor_links.append((i, j))

# 创建图并绘制
G = nx.Graph()
G.add_nodes_from(range(19))
G.add_edges_from(neighbor_links)

# 绘制图形
plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edgelist=neighbor_links, width=2, alpha=0.6, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

plt.title("第一个数据样本的 5-近邻图", fontsize=16)
plt.axis('off')
plt.show()


