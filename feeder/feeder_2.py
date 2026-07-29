# sys
import os
import sys
import numpy as np
import random
import pickle
import time

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# operation
from . import tools

# 20260718  之前的sample level 一直运行的OK

def normalize_sample(data_numpy, landmark_indices=range(29), pose_index=28, eps=1e-8):  # 这块的19是默认参数，实际参数是通过feeder传递
    """对单个样本进行归一化处理"""
    # print('data_numpy.shape', data_numpy.shape)   # (C, T, V, M) 逐个归一化数据
    # print('landmark_indices', landmark_indices)
    normalized_data = data_numpy.copy().astype(np.float32)
    V = normalized_data.shape[2]
    # print('V', V)
    
    # landmarks z-score
    landmarks_data = normalized_data[:, :, landmark_indices, :]
    mean = landmarks_data.mean()
    std = landmarks_data.std() + eps
    normalized_data[:, :, landmark_indices, :] = (landmarks_data - mean) / std
    
    '''
    if V > pose_index:
        # pose: 角度转弧度 + z-score
        # print('归一化姿态角度数据...')
        pose_data = normalized_data[:, :, pose_index:pose_index+1, :]
        pose_rad = pose_data * (np.pi / 180.0)
        mean = pose_rad.mean()
        std = pose_rad.std() + eps
        normalized_data[:, :, pose_index, :] = ((pose_rad - mean) / std).squeeze(2)
    '''
    
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
                 pose_index=28):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.landmark_indices = list(landmark_indices) if landmark_indices is not None else list(range(19))
        self.pose_index = pose_index
        
        # 如果要归一化，不能用 mmap（因为要修改数据）
        if self.normalization:
            mmap = False
            
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.label = np.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            
        # 兼容 (N,T,V,C) -> (N,C,T,V,1)
        if self.data.ndim == 4 and self.data.shape[-1] in (2, 3):
            self.data = self.data.transpose(0, 3, 1, 2)
            self.data = self.data[..., np.newaxis]
        
        print("=" * 50)
        print("Feeder 数据加载信息:")
        print(f"  data shape: {self.data.shape}")
        print(f"  label shape: {self.label.shape}")
        print(f"  normalization: {self.normalization}")
        
        # ===== 一次性预处理归一化 =====
        if self.normalization:
            self._preprocess_normalization()
            
        print("=" * 50)

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def _preprocess_normalization(self):
        """一次性对整个数据集进行归一化"""
        print("开始预处理归一化...")
        start_time = time.time()
        
        N = len(self.data)
        normalized_data = np.zeros_like(self.data, dtype=np.float32)
        
        for i in range(N):
            normalized_data[i] = normalize_sample(
                self.data[i],
                landmark_indices=self.landmark_indices,
                pose_index=self.pose_index
            )
            # 打印进度
            if (i + 1) % 10000 == 0:
                print(f"  进度: {i + 1}/{N}")
        
        self.data = normalized_data
        elapsed = time.time() - start_time
        print(f"归一化完成，耗时: {elapsed:.2f} 秒")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # 直接取数据，已经归一化过了
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # 其他处理（数据增强等）
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index