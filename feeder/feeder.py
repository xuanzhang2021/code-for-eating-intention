import numpy as np
import torch
from . import tools
import time


def compute_frame_center_and_scale(
    landmarks_frame,
    center_mode='mean',
    scale_mode='rms',
    eps=1e-6
):
    """
    对单帧 landmarks 计算几何中心与尺度

    参数:
        landmarks_frame: shape (C, V_lm, M)
        center_mode: 'mean'
        scale_mode: 'rms' or 'max'
    返回:
        center: shape (C, 1, M)
        scale:  shape (1, 1, M)
    """
    # center: (C, 1, M)
    if center_mode == 'mean':
        center = landmarks_frame.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"暂不支持的 center_mode: {center_mode}")

    # diff: (C, V_lm, M)
    diff = landmarks_frame - center

    # 只用前两个通道 (x, y) 计算尺度；如果你是 3D 也可改成前3维
    if diff.shape[0] < 2:
        raise ValueError(f"坐标通道数 C={diff.shape[0]} 太小，至少需要2维坐标")

    dist = np.sqrt(np.sum(diff[:2] ** 2, axis=0, keepdims=True))  # (1, V_lm, M)

    if scale_mode == 'rms':
        scale = np.sqrt(np.mean(dist ** 2, axis=1, keepdims=True)) + eps  # (1,1,M)
    elif scale_mode == 'max':
        scale = np.max(dist, axis=1, keepdims=True) + eps  # (1,1,M)
    else:
        raise ValueError(f"暂不支持的 scale_mode: {scale_mode}")

    return center.astype(np.float32), scale.astype(np.float32)


def geometric_normalize_sample(
    data_numpy,
    landmark_indices,
    pose_index=None,
    transform_pose=False,
    center_mode='mean',
    scale_mode='rms',
    eps=1e-6
):
    """
    对单个样本做几何归一化
    data_numpy shape: (C, T, V, M)

    返回:
        normalized_data: (C, T, V, M)
    """
    data_numpy = data_numpy.astype(np.float32)
    normalized_data = data_numpy.copy()

    C, T, V, M = normalized_data.shape

    if landmark_indices is None or len(landmark_indices) == 0:
        raise ValueError("landmark_indices 不能为空")

    if min(landmark_indices) < 0 or max(landmark_indices) >= V:
        raise ValueError(f"landmark_indices 超出范围, V={V}")

    if pose_index is not None:
        if pose_index < 0 or pose_index >= V:
            raise ValueError(f"pose_index={pose_index} 超出范围, V={V}")
        if pose_index in landmark_indices:
            raise ValueError(f"pose_index={pose_index} 与 landmark_indices 重叠")

    for t in range(T):
        # landmarks_frame: (C, V_lm, M)
        landmarks_frame = normalized_data[:, t, landmark_indices, :]

        center, scale = compute_frame_center_and_scale(
            landmarks_frame,
            center_mode=center_mode,
            scale_mode=scale_mode,
            eps=eps
        )

        # 对 landmarks 做几何归一化
        normalized_data[:, t, landmark_indices, :] = (
            landmarks_frame - center
        ) / scale

        # pose 是否跟着一起变换
        if transform_pose and pose_index is not None:
            pose_frame = normalized_data[:, t, pose_index:pose_index+1, :]  # (C,1,M)
            normalized_data[:, t, pose_index:pose_index+1, :] = (
                pose_frame - center
            ) / scale

    return normalized_data


class Feeder(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True,
                 normalization=True,
                 landmark_indices=None,
                 pose_index=28,
                 transform_pose=False,
                 center_mode='mean',
                 scale_mode='rms',
                 eps=1e-6):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization

        # 默认前 28 个是 face landmarks
        self.landmark_indices = list(landmark_indices) if landmark_indices is not None else list(range(28))

        # 第 28 个点是 pose
        self.pose_index = pose_index
        self.transform_pose = transform_pose

        self.center_mode = center_mode
        self.scale_mode = scale_mode
        self.eps = eps

        # 几何归一化也会重写数据，所以不能 mmap
        if self.normalization:
            mmap = False

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        self.label = np.load(self.label_path)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[:100]
            self.data = self.data[:100]

        # 兼容 (N, T, V, C) -> (N, C, T, V, 1)
        if self.data.ndim == 4 and self.data.shape[-1] in (2, 3, 4):
            self.data = self.data.transpose(0, 3, 1, 2)
            self.data = self.data[..., np.newaxis]

        if self.data.ndim != 5:
            raise ValueError(f"期望 data shape 为 (N,C,T,V,M)，但实际得到 {self.data.shape}")

        print("=" * 60)
        print("Feeder 数据加载信息:")
        print(f"  data shape: {self.data.shape}")
        print(f"  label shape: {self.label.shape}")
        print(f"  normalization: {self.normalization}")
        print(f"  landmark_indices: {self.landmark_indices}")
        print(f"  pose_index: {self.pose_index}")
        print(f"  transform_pose: {self.transform_pose}")
        print(f"  center_mode: {self.center_mode}")
        print(f"  scale_mode: {self.scale_mode}")

        _, _, _, V, _ = self.data.shape

        # 合法性检查
        if len(self.landmark_indices) == 0:
            raise ValueError("landmark_indices 不能为空")

        if min(self.landmark_indices) < 0 or max(self.landmark_indices) >= V:
            raise ValueError(f"landmark_indices 超出范围, V={V}")

        if self.pose_index is not None:
            if self.pose_index < 0 or self.pose_index >= V:
                raise ValueError(f"pose_index={self.pose_index} 超出范围, V={V}")

            if self.pose_index in self.landmark_indices:
                raise ValueError(
                    f"pose_index={self.pose_index} 与 landmark_indices 重叠，请检查配置"
                )

        if self.normalization:
            self._preprocess_geometric_normalization()

        print("=" * 60)

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def _preprocess_geometric_normalization(self):
        """一次性对整个数据集做几何归一化"""
        print("开始几何归一化...")
        start_time = time.time()

        N = len(self.data)
        normalized_data = np.zeros_like(self.data, dtype=np.float32)

        for i in range(N):
            normalized_data[i] = geometric_normalize_sample(
                self.data[i],
                landmark_indices=self.landmark_indices,
                pose_index=self.pose_index,
                transform_pose=self.transform_pose,
                center_mode=self.center_mode,
                scale_mode=self.scale_mode,
                eps=self.eps
            )

            if (i + 1) % 1000 == 0 or (i + 1) == N:
                print(f"  进度: {i + 1}/{N}")

        self.data = normalized_data
        elapsed = time.time() - start_time
        print(f"几何归一化完成，耗时: {elapsed:.2f} 秒")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index], dtype=np.float32)
        label = self.label[index]

        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)

        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index