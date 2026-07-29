import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph, normalize_undigraph
from net.utils.st_gat import AttnHeadV10, AttnHeadV11, CrossAttributeTransformSimple, CrossAttributeTransform1
import random
import numpy as np
import sys
from scipy.signal import savgol_filter

# 跑通了基础的attention机制，基于相似度的边权重机制，在上述基础上，基于时空关系的注意力机制  目前是case论文提交的代码---20260312


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
        self.sim_detach = True  # 是否阻断相似度矩阵的梯度传播，避免训练不稳定
        self.in_channels = in_channels
        self.graph = Graph(**graph_args)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        # ===== 新增：可学习的相似度融合参数 =====
        num_nodes = A.size(1)  # V
        
        # 可学习的融合权重，初始化为负值使 sigmoid 后接近 0
        # 这样初始时 A_eff ≈ A，相似度几乎不起作用
        # self.sim_beta = nn.Parameter(torch.tensor(-3.0))
        self.sim_beta = nn.Parameter(torch.tensor(0.5))  # 初始时 sigmoid ≈ 0.5
        
        # 可学习的 sigma 参数
        self.sim_sigma = nn.Parameter(torch.tensor(1.0))
        
        # 可学习的节点嵌入（用于计算相似度，替代原始坐标）
        # self.sim_embed = nn.Parameter(torch.randn(num_nodes, 32) * 0.01)
        
        # 是否使用可学习嵌入（True）还是原始坐标（False）
        self.use_learnable_embed = True
        # ===== 新增结束 =====

        # 额外的相似度嵌入（可学习）
        self.sim_embed = nn.Parameter(torch.randn(num_nodes, 32) * 0.01)
        self.sim_weight = nn.Parameter(torch.tensor(0.0))  # 控制相似度的影响程度

        # self.cross_attr_transform = CrossAttributeTransformSimple(v_alpha=19)
        self.cross_attr_transform = CrossAttributeTransform1(v_lmk=19, v_pose=1, d_k=16)
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 1
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))
        self.attn_after_layer = 1
        attn_channels = in_channels  
        self.attn = AttnHeadV11(
            in_features=attn_channels,
            out_features=attn_channels,
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
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        embed_dim = 32
        num_node = num_nodes # 根据你的数据集调整

        self.sim_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(num_node, embed_dim))  # 每层一个独立的嵌入
            for _ in self.st_gcn_networks])
        # 每层一个可学习的融合系数 beta
        self.layer_betas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))  # 初始化为0，sigmoid后约0.5
            for _ in range(len(self.st_gcn_networks))])


    # ===== 新增：计算可学习相似度的方法 =====
    def compute_learnable_similarity(self):
        """
        基于可学习节点嵌入计算相似度矩阵
        Returns:
            S: (V, V) 相似度矩阵，值在 [0, 1] 之间
        """
        # L2 归一化
        embed_norm = F.normalize(self.sim_embed, dim=-1)  # (V, 16)
        # 余弦相似度 -> [-1, 1]
        S = torch.matmul(embed_norm, embed_norm.transpose(0, 1))  # (V, V)
        # 映射到 [0, 1]
        S = (S + 1.0) / 2.0
        return S
    
    def compute_coordinate_similarity(self, x):
        """
        基于坐标计算高斯相似度（原始方法的改进版）
        Args:
            x: (N, C, T, V) 输入特征
        Returns:
            S: (V, V) 相似度矩阵
        """
        # 使用可学习的 sigma
        sigma = self.sim_sigma.abs().clamp(min=0.1)
        S_nvv = gaussian_similarity(x, sigma=sigma)  # (N, V, V)
        
        if self.sim_detach:
            S_nvv = S_nvv.detach()
        # 聚合成全局相似度
        S = S_nvv.mean(dim=0)  # (V, V)
        return S
    # ===== 新增结束 =====

    def forward(self, x):
        n, c, t, v, m = x.size()

        x = self.cross_attr_transform(x)
        
        # ===== 修改：数据预处理移到前面 =====
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # n, m, v, c, t
        x = x.view(n * m, v * c, t)
        # x = self.data_bn(x)  # 如果需要可以取消注释
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)
        # ===== 修改结束 =====
        '''
        # ===== 修改：计算相似度矩阵（方案4 残差式融合）=====
        if self.use_learnable_embed:
            # 使用可学习嵌入计算相似度
            S_vv = self.compute_learnable_similarity()  # (V, V)
        else:
            # 使用坐标计算相似度（需要原始 x，这里用处理后的 x 近似）
            S_vv = self.compute_coordinate_similarity(x)  # (V, V)
        
        # 残差式融合的核心：beta 控制相似度的影响程度
        # beta 初始化为接近 0，所以初始时 A_eff ≈ A
        # 训练过程中如果相似度有用，beta 会变大
        beta = torch.sigmoid(self.sim_beta)  # [0, 1]
        
        # 扩展相似度到 (K, V, V)
        S_kvv = S_vv.unsqueeze(0)  # (1, V, V) -> broadcast to (K, V, V)
        
        # 残差式乘性融合
        # 当 beta ≈ 0 时：A_eff ≈ A（相似度不起作用）
        # 当 beta ≈ 1 时：A_eff = A * (1 + S)（相似度完全起作用）
        A_eff = self.A * (1.0 + beta * S_kvv)
        # # 加性融合（相似度作为额外的边）
        A_eff = (1.0 - beta) * self.A + beta * S_kvv
        
        # 归一化
        A_eff = normalize_undigraph_torch(A_eff)
        # ===== 修改结束 =====
        '''
    
        alpha = torch.sigmoid(self.sim_weight)

        # ===== 修改：Attention 使用原始 A（避免与相似度重复调制）=====
        A0 = (self.A.sum(dim=0) > 0)  # [V, V] bool
        bias_vv = torch.where(
            A0, 
            torch.zeros((v, v), device=x.device), 
            torch.full((v, v), -1e9, device=x.device),
        )
        bias = bias_vv.view(1, 1, v, v)
        
        # Attention 用原始 A，不用 A_eff
        g = self.A.to(x.device).float().permute(1, 2, 0).contiguous().view(1, 1, v, v, 1)   # spitial的时候(1, 1, v, v, 3)，uniform (1, 1, v, v, 1)
        # x, alpha_x = self.attn(x, bias=bias, g=g)
        # ===== 修改结束 =====

        # ===== 修改：GCN 层使用 A_eff =====
        # for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
        for gcn, sim_embed, beta_param in zip(self.st_gcn_networks, self.sim_embeds, self.layer_betas):
            # 使用相似度增强的邻接矩阵
            # x, _ = gcn(x, A_eff * importance)
            # x, _ = gcn(x, self.A * importance)
            # print(f"Layer {i} edge_importance - mean: {importance.mean():.4f}, std: {importance.std():.4f}, min: {importance.min():.4f}, max: {importance.max():.4f}")
            # eff_importance = importance * (1 + alpha * S)
            # x, _ = gcn(x, self.A * eff_importance)
            # '''
            # 计算相似度，consin 相似度 这个相似度结果92%  case 论文中使用的是这个相似度，测试了高斯核相似度，效果不如余弦相似度
            S = F.cosine_similarity(sim_embed.unsqueeze(1), sim_embed.unsqueeze(0), dim=2)
            S = (S + 1) / 2
            S = S.unsqueeze(0)  # (1, V, V) -> 广播到 (K, V, V)
            beta = torch.sigmoid(beta_param)  # [0, 1]
            S = self.A * (1.0 + beta * S)
            x, _ = gcn(x, self.A * S)   # 这块S乘2次，精度稳稳的在95%以上
            
        # ===== 修改结束 =====

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(n, m, -1, 1, 1).mean(dim=1)
       
        # prediction
        x = self.fcn(x)  
        x = x.view(x.size(0), -1)
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
        return self.relu(x), A
