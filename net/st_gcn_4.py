import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph, normalize_undigraph
from net.utils.st_gat import AttnHeadV10, AttnHeadV11
import random
import numpy as np
import sys
from scipy.signal import savgol_filter

# 跑通了基础的attention机制，基于相似度的边权重机制，在上述基础上，基于时空关系的注意力机制

def adj_to_bias_v3(adj, nhood=1):  # adj shape: (m, v, v)
    m, v, last_dim = adj.shape  # 图的数量、节点数
    assert v == last_dim, "The adjacency matrix must be square (v x v)."

    mt = torch.empty_like(adj)  # 初始化输出张量，形状与 adj 相同

    for graph in range(m):  # 遍历每个图
        # 初始化为单位矩阵
        mt[graph] = torch.eye(v, device=adj.device)

        # 逐层扩展邻居范围
        for _ in range(nhood):
            mt[graph] = torch.matmul(
                mt[graph],
                adj[graph] + torch.eye(v, device=adj.device)
            )

        # 将邻接矩阵二值化
        for i in range(v):
            for j in range(v):
                if mt[graph, i, j] > 0.0:
                    mt[graph, i, j] = 1.0
    # print('mt.shape', mt.shape)
    # 返回偏置矩阵
    return -1e9 * (1.0 - mt)


# 这段代码要改，目前是把第20个节点的特征替换成了注意力机制计算得到的特征，要改为3维转换为3维
def cross_attribute_transform(X, v_alpha=19, v_beta=1, d_k=8, d_u=1):
    """
    Compute the output of the multi-head attention mechanism based on Query, Key, and Value inputs.
    Then:
    1. Expand the second-to-last dimension of Y_alpha to 3 dimensions.
    2. Restore Y_alpha to the original shape (N, T, C, V, M).
    3. Replace the 20th node of the input X with Y_alpha, while keeping the first 19 nodes of X unchanged.

    Parameters:
    - X: Input tensor with shape (N, T, C, V, M)
    - V_alpha: Input feature dimension for Query (V_alpha)
    - V_beta: Input feature dimension for Key/Value (V_beta)
    - d_k: Feature dimension of Query/Key in attention
    - d_u: Output feature dimension of Value

    Returns:
    - X_modified: Modified tensor with the 20th node replaced by Y_alpha
    """

    # Extract Query input (C: 2, V: V_alpha)
    x_alpha = X[:, :2, :, :v_alpha, :]  # (N, T, 2, 19, M)
    x_alpha = x_alpha.permute(0, 2, 4, 1, 3)  # Rearrange dimensions: (N, T, M, C_alpha, V_alpha)

    # Extract Key and Value input (C: 3, V: V_beta)
    x_beta = X[:, :, :, -1:, :]  # (N, T, 3, 1, M)
    x_beta = x_beta.permute(0, 2, 4, 1, 3)  # Rearrange dimensions: (N, T, M, C_beta, V_beta)

    # Define linear transformation matrices
    device = x_alpha.device
    w_q_alpha = torch.nn.Parameter(torch.rand(v_alpha, d_k, device=device))  # (V_alpha, d_k)
    w_k_beta = torch.nn.Parameter(torch.rand(v_beta, d_k, device=device))  # (V_beta, d_k)
    w_v_beta = torch.nn.Parameter(torch.rand(v_beta, d_u, device=device))  # (V_beta, d_u)

    # Compute Q, K, V
    q_alpha = torch.einsum('ntmcv,vd->ntmcd', x_alpha, w_q_alpha)  # (N, T, M, C, d_k)
    k_beta = torch.einsum('ntmcv,vd->ntmcd', x_beta, w_k_beta)  # (N, T, M, C, d_k)
    value_beta = torch.einsum('ntmcv,vd->ntmcd', x_beta, w_v_beta)  # (N, T, M, C, d_u)

    # Compute attention scores
    d_k_sqrt = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_scores = torch.einsum('ntmcd,ntmed->ntmce', q_alpha, k_beta) / d_k_sqrt  # (N, T, M, C_alpha, C_beta)

    # Normalize attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)  # (N, T, M, C_alpha, C_beta)

    # Weighted sum of V_beta
    y_alpha = torch.einsum('ntmce,ntmed->ntmcd', attention_weights, value_beta)  # (N, T, M, 2, d_u)
    x_combined = torch.cat([x_alpha, y_alpha], dim=-1)  # Shape: [10, 30, 1, 2, 20]
    x_combined = x_combined.permute(0, 3, 1, 4, 2)  # n, c, t, v, m = [10, 2, 30, 20, 1]

    return x_combined


def savitzky_golay_filter_to_channels(data, window_length=5, polyorder=2):
    """
    对数据的c通道的x, y, z应用Savitzky-Golay滤波器。

    参数:
    data (numpy.ndarray): 具有形状(n, c, t, v, m)的数据。
    window_length (int): 滤波窗口长度，必须是正奇数。
    polyorder (int): 多项式阶数，必须小于window_length。

    返回:
    numpy.ndarray: 滤波后的数据。
    """
    # 检查输入参数
    if window_length % 2 == 0:
        raise ValueError("window_length必须是奇数")
    if polyorder >= window_length:
        raise ValueError("polyorder必须小于window_length")

    # 创建一个与输入数据形状相同的数组来存储滤波后的结果
    filtered_data = data.clone()

    # 遍历n和c
    for i in range(data.shape[0]):  # 遍历n
        for j in range(data.shape[1]):  # 遍历c
            # 对v维度的前三个元素(x, y, z)应用滤波
            for k in range(3):  # 假设x, y, z对应于v的前三个元素
                # 提取时间序列
                # time_series = data[i, j, :, k, 0]  # m维度为1，所以索引为0
                time_series = data[i, j, :, k, 0].cpu().numpy()  # 转换为numpy数组
                # 应用Savitzky-Golay滤波器
                filtered_time_series = savgol_filter(time_series, window_length, polyorder)
                # 存储滤波后的结果
                filtered_data[i, j, :, k, 0] = torch.tensor(filtered_time_series, device=data.device)

    return filtered_data


def gaussian_similarity(input_tensor, sigma=1.0):
    """
    使用高斯核计算节点间的相似度。

    参数:
        input_tensor: torch.Tensor, shape为(n, c, t, v, m)
        sigma: float, 高斯核带宽参数

    返回:
        similarity_matrix: torch.Tensor, shape为(n, v, v)，表示每个batch内节点间的相似度矩阵
    """

    # 第一步：对 c, t, m 维度聚合，得到节点特征 (这里取平均)
    node_features = input_tensor.mean(dim=[1, 2, 4])  # (n, v)

    # 如果希望保留特征维度而非聚合到标量，可以使用:
    # node_features = input_tensor.mean(dim=[2,4]).permute(0,2,1)  # (n, v, c)

    # 这里我们使用保留特征维度的方式，以便更精确的相似度计算
    node_features = input_tensor.mean(dim=[2, 4]).permute(0, 2, 1)  # (n, v, c)

    n, v, c = node_features.shape

    # 扩展维度计算两两节点的差异
    features_i = node_features.unsqueeze(2)  # (n, v, 1, c)
    features_j = node_features.unsqueeze(1)  # (n, 1, v, c)

    # 计算欧氏距离平方
    diff = features_i - features_j  # (n, v, v, c)
    dist_squared = (diff ** 2).sum(dim=-1)  # (n, v, v)

    # 计算高斯核相似度
    similarity_matrix = torch.exp(-dist_squared / (2 * sigma ** 2))  # (n, v, v)

    return similarity_matrix

def normalize_undigraph_torch(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}
    A: (..., V, V) torch tensor
    """
    deg = A.sum(dim=-1)                              # (..., V)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)        # (..., V)
    return deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def save_output_to_file(self, filename):
        # 打开或创建一个txt文件用于写入
        with open(filename, 'w') as f:
            # 将标准输出重定向到文件
            original_stdout = sys.stdout  # 保存原始标准输出流
            sys.stdout = f  # 重定向到文件
            try:
                print('A', self.graph.A.shape, self.graph.A)  # 打印内容会写入文件
            finally:
                sys.stdout = original_stdout  # 恢复原始标准输出流

    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()
        random.seed(1)
        # load graph
        # print(in_channels)
        self.sim_detach = True  # 是否阻断相似度矩阵的梯度传播，避免训练不稳定
        self.in_channels = in_channels
        self.graph = Graph(**graph_args)
        # print('A', self.graph.A.shape, self.graph.A)
        # self.save_output_to_file('output.txt')

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        # build networks
        spatial_kernel_size = A.size(0)
        # temporal_kernel_size = 9
        temporal_kernel_size = 1
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),
        ))
        self.attn_after_layer = 1  # 在第几层之后插入注意力（0-indexed）
        attn_channels = in_channels  
        self.attn = AttnHeadV11(
            in_features=attn_channels,
            out_features=attn_channels,   # 建议先保持通道不变，方便接你后面的 reshape + st_gcn
            edge_dim=3,
            time_hidden=64,
            activation=F.elu,
            in_drop=0.0,
            coef_drop=0.0,
            residual=True,)

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        
        # fcn for prediction
        self.fcn = nn.Conv2d(128, 256, kernel_size=1)
       
        # linear layer  这块的out_features 决定该模型的数据输出维度，为什么是1，不是2？
        self.linear = nn.Linear(256, num_class)
        # self.preprocessed_data = self.compute_derivatives(input_data)

    def forward(self, x):
        # data normalization
        # print('x.shape_before_before:', x.shape)
        # x: (n, c, t, v, m) at this point
        # x = cross_attribute_transform(x)
        # x = savitzky_golay_filter_to_channels(x)
        # print('x:', x.shape)
        ######################计算 相似度矩阵 S_kvv ############################################## 方式一
        S_nvv = gaussian_similarity(x, sigma=1.0)  # (n, v, v)
        # 可选：阻断梯度，避免相似度分支让训练不稳定
        if self.sim_detach:
            S_nvv = S_nvv.detach()
        # 聚合成全局相似度 (v,v)，以兼容 st_gcn 的 (K,v,v)
        S_nvv = S_nvv.mean(dim=0)  # (v, v)
        # 可选：归一化/稳定（推荐至少做一个）
        # 让每行和为1（注意：softmax会把对角/非对角都映射到(0,1)）
        # S_vv = torch.softmax(S_vv, dim=-1)
        # 扩展到 (K,v,v)
        S_kvv = S_nvv.unsqueeze(0)  # (1,v,v) -> broadcast to (K,v,v) in multiplication

        # 乘性融合（保留骨架先验，同时用相似度调制）
        A_eff = (self.A) * (1.0 + S_kvv)
        A_eff = normalize_undigraph_torch(A_eff)

        ######################计算 相似度矩阵 S_kvv ############################################ 方式一
        

        # 用 A 构造 mask/bias（不存在的边给一个很小的值）
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # n, m, v, c, t
        x = x.view(n * m, v * c, t)
        # x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        
        # 或者加性融合（更像“多图融合”）
        # A_eff = self.sim_alpha * (self.A * importance) + (1.0 - self.sim_alpha) * (S_kvv)

        A0 = (self.A.sum(dim=0) > 0)  # [V,V] bool
        bias_vv = torch.where(A0, torch.zeros((v, v), device=x.device), torch.full((v, v), -1e9, device=x.device),)
        bias = bias_vv.view(1, 1, v, v)  # [1,1,V,V] -> broadcast到 [N,T,V,V]
        # A_eff: [3,V,V] -> [V,V,3] -> [1,1,V,V,3]  (edge_dim=3)
        g = self.A.to(x.device).float().permute(1, 2, 0).contiguous().view(1, 1, v, v, 3)
        # g = A_eff.to(x.device).float().permute(1, 2, 0).contiguous().view(1, 1, v, v, 3)
        x, alpha = self.attn(x, bias=bias, g=g)  
        # print('alpha:', alpha.shape)
        # print('x.shape_after_attn:', x.shape)
        # print('x.shape:', x.shape)

        # forwad
        for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):   # 这块的 for 循环指的是多层 ST-GCN block
            # print('A', self.A.shape)
            # print('x', x.shape)

            # x, _ = gcn(x, A_eff * importance)
            x, _ = gcn(x, self.A * importance)
            # 在第 attn_after_layer 层之后应用注意力
            # if i == self.attn_after_layer:
                # x, alpha = self.attn(x, bias=bias, g=g)

            self.searched_graph_tmp = []

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(n, m, -1, 1, 1).mean(dim=1)
       
        # prediction
        x = self.fcn(x)  
        # add
        # x = self.sigmoid(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)
        return x

    def extract_feature(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        # print('x_1', x.shape)
        # print('A_1', A.shape)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        #x
        return self.relu(x), A
