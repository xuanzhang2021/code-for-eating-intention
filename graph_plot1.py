import numpy as np

path = "/media/zhangxuan/KINGSTON/paper2/1450308271_2d_sparse_smooth_lmk_pose.npz"
data = np.load(path, allow_pickle=True)

print("keys:", data.files)
for k in data.files:
    arr = data[k]
    print(f"\n{k}:")
    print("type:", type(arr))
    print("shape:", getattr(arr, "shape", None))
    print("dtype:", getattr(arr, "dtype", None))
    print("value preview:", arr if np.isscalar(arr) else arr[:5])


import matplotlib.pyplot as plt

idx = np.array(
    [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45,
     48, 51, 54, 57, 49, 53, 55, 59, 7, 9, 27, 30],
    dtype=np.int64
)

########## 绘制28个点，下
neighbor_link_28 = [(0, 1), (0, 5), (0, 12), (1, 9), (1, 12), (1, 16), (2, 19), (2, 24), (2, 25), (3, 4),
                                (3, 11), (3, 15), (3, 18), (4, 8), (4, 15), (5, 6), (5, 12), (5, 13), (6, 7), (6, 12),
                                (6, 13), (6, 26), (7, 8), (7, 14), (7, 15), (7, 26), (8, 14), (8, 15), (9, 10), (9, 16), (9, 17), 
                                (9, 20), (9, 27), (10, 11), (10, 17), (10, 27), (11, 17), (11, 18), (11, 21), (11, 27), (12, 13), 
                                (13, 26), (14, 15), (14, 26),  (16, 20), (16, 23), (16, 24), (17, 19),
                                (17, 20), (17, 21), (17, 22), (17, 23), (18, 21), (18, 22), (18, 25), (19, 22), 
                                (19, 23), (19, 24), (19, 25),(20, 23),(21, 22),(22, 25),(23, 24), 
                                (1, 24), (3, 25), (26, 27), (9, 12), (11, 15)]  # , (28, 26), (28, 27), (28, 10) 
data = np.load("/media/zhangxuan/KINGSTON/paper2/1450308271_2d_sparse_smooth_lmk_pose.npz", allow_pickle=True)
landmarks = data["landmarks"]

t = 0
frame = landmarks[t]
x68 = frame[0]
y68 = frame[1]

x = x68[idx]
y = y68[idx]

plt.figure(figsize=(10, 10))

# 原始68点
plt.scatter(x68, y68, c='lightgray', s=15, label='68 landmarks')
for k in range(68):
    plt.text(x68[k], y68[k], str(k), fontsize=7, color='gray')

# graph边
for i, j in neighbor_link_28:
    plt.plot([x[i], x[j]], [y[i], y[j]], 'b-', linewidth=1.5)

# 选中的28点
plt.scatter(x, y, c='yellow', s=120, edgecolors='black', linewidths=2, label='selected 28 points', zorder=4)

for k in range(28):
    plt.text(x[k], y[k], str(k), fontsize=10, color='blue', weight='bold')

plt.gca().invert_yaxis()
plt.axis('equal')
plt.legend()
plt.title(f"Frame {t}: 68 landmarks + 28-point graph")
plt.show()
##############绘制28个点，上


##############绘制19个点，下
idx = np.array(
    [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54],
    dtype=np.int64
)

neighbor_link_19 = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (12, 5), (12, 7), (5, 6),
                             (6, 13), (6, 15), (7, 8), (7, 12), (7, 14), (8, 15), (8, 3), (8, 4),
                             (9, 10), (10, 11), (10, 13), (10, 14), (12, 13), (12, 9), (11, 15),
                             (13, 14), (14, 15), (15, 3), (16 ,1), (16, 2), (16, 13), (17, 9),
                             (17, 10), (17, 11), (18, 2), (18, 3), (18, 14), (18, 11), (1, 9),
                             (3, 11), (9, 16), (5, 1), (12, 1)]

data = np.load("/media/zhangxuan/KINGSTON/paper2/1450308271_2d_sparse_smooth_lmk_pose.npz", allow_pickle=True)
landmarks = data["landmarks"]

t = 0
frame = landmarks[t]
x68 = frame[0]
y68 = frame[1]

x = x68[idx]
y = y68[idx]

plt.figure(figsize=(10, 10))

# 原始68点
plt.scatter(x68, y68, c='lightgray', s=15, label='68 landmarks')
for k in range(68):
    plt.text(x68[k], y68[k], str(k), fontsize=7, color='gray')

# graph边
for i, j in neighbor_link_19:
    plt.plot([x[i], x[j]], [y[i], y[j]], 'b-', linewidth=1.5)

# 选中的28点
plt.scatter(x, y, c='yellow', s=120, edgecolors='black', linewidths=2, label='selected 19 points', zorder=4)

for k in range(19):
    plt.text(x[k], y[k], str(k), fontsize=10, color='blue', weight='bold')

plt.gca().invert_yaxis()
plt.axis('equal')
plt.legend()
plt.title(f"Frame {t}: 68 landmarks + 28-point graph")
plt.show()

##############绘制19个点，上


def knn_graph_from_points(points, k=3):
    N = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)

    edges = set()
    for i in range(N):
        nn_idx = np.argsort(dist[i])[:k]
        for j in nn_idx:
            edges.add(tuple(sorted((i, j))))
    return sorted(list(edges))


data = np.load("/media/zhangxuan/KINGSTON/paper2/1450308271_2d_sparse_smooth_lmk_pose.npz", allow_pickle=True)
landmarks = data["landmarks"]

idx = np.array(
    [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45,
     48, 51, 54, 57, 49, 53, 55, 59, 7, 9, 27, 30],
    dtype=np.int64
)

t = 0
frame = landmarks[t]
x68 = frame[0]
y68 = frame[1]

x = x68[idx]
y = y68[idx]
points = np.stack([x, y], axis=1)

k = 3
edges = knn_graph_from_points(points, k=k)

print("neighbor_link_knn =", edges)

plt.figure(figsize=(8, 8))

# 画kNN边
for i, j in edges:
    plt.plot([x[i], x[j]], [y[i], y[j]], 'b-', linewidth=1.5)

# 画点
plt.scatter(x, y, c='red', s=80, edgecolors='black', zorder=3)

# 标注点编号
for i in range(len(x)):
    plt.text(
        x[i], y[i], str(i),
        fontsize=10, color='darkgreen', weight='bold',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.8),
        zorder=4
    )

plt.gca().invert_yaxis()
plt.axis('equal')
plt.title(f"28-point kNN graph (k={k})")
plt.show()



from sklearn.neighbors import NearestNeighbors
def load_landmarks_txt(txt_path):
    """
    读取 3x68 或 68x3 的 landmark txt
    返回:
        pts3d: (68, 3)
    """
    data = np.loadtxt(txt_path)

    # 你的数据看起来是 3x68
    if data.shape == (3, 68):
        pts3d = data.T   # -> (68, 3)
    elif data.shape == (68, 3):
        pts3d = data
    else:
        raise ValueError(f"Unexpected shape: {data.shape}, expected (3,68) or (68,3)")

    return pts3d


def knn_graph_from_points_sklearn(points, k=5):
    """
    points: (N, 2)
    返回无向边列表
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nbrs.fit(points)

    distances, indices = nbrs.kneighbors(points)

    edges = set()
    for i in range(points.shape[0]):
        # indices[i][0] 通常是自己
        for j in indices[i][1:k + 1]:
            edges.add(tuple(sorted((i, j))))

    return sorted(list(edges))


def draw_graph(points, edges, labels=None, title='KNN Graph'):
    """
    points: (N, 2)
    edges: [(i,j), ...]
    """
    plt.figure(figsize=(8, 8))

    # 画边
    for i, j in edges:
        x1, y1 = points[i]
        x2, y2 = points[j]
        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1)

    # 画点
    plt.scatter(points[:, 0], points[:, 1], c='red', s=40)

    # 标注编号
    if labels is None:
        labels = list(range(len(points)))

    for i, (x, y) in enumerate(points):
        plt.text(x + 1, y + 1, str(labels[i]), fontsize=9, color='green')

    plt.title(title)
    plt.gca().invert_yaxis()   # 人脸图像坐标通常 y 向下，反转更像原图
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    txt_path = '/media/zhangxuan/KINGSTON/paper2/test_2d_sparse.txt'

    # 你给的 landmark 索引
    '''
    idx = np.array(
        [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45, 48, 51, 54],
        dtype=np.int64
    )
    '''

    idx = np.array(
        [1, 3, 8, 13, 15, 17, 21, 22, 26, 31, 33, 35, 36, 39, 42, 45,
        48, 51, 54, 57, 49, 53, 55, 59, 7, 9, 27, 30],
        dtype=np.int64
    )

    # 1. 读取 68 个 3D 点
    pts3d = load_landmarks_txt(txt_path)   # (68, 3)

    # 2. 取指定点
    selected_pts3d = pts3d[idx]            # (19, 3)

    # 3. 只取 2D
    selected_pts2d = selected_pts3d[:, :2] # (19, 2)

    # 4. 建 KNN 图
    edges = knn_graph_from_points_sklearn(selected_pts2d, k=4)

    print("Selected points shape:", selected_pts2d.shape)
    print("Edges:")
    for e in edges:
        print(e)

    # 5. 绘图
    draw_graph(
        selected_pts2d,
        edges,
        labels=idx,   # 显示原始 landmark 编号
        title='KNN Graph on Selected 68 Facial Landmarks'
    )





