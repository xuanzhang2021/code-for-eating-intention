import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph, normalize_undigraph
from net.utils.st_gat import AttnHeadGATConcat, AttnHeadGAT, AttnHeadV10, AttnHeadV11, CrossAttributeTransformSimple, CrossAttributeTransform1, CrossAttributeTransform2
import random
import numpy as np
import sys
from scipy.signal import savgol_filter

# 跑通了基础的attention机制，基于相似度的边权重机制，在上述基础上，基于时空关系的注意力机制  目前是case论文提交的代码---20260312


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
        # self.cross_attr_transform = CrossAttributeTransform1(v_lmk=19, v_pose=1, d_k=16)
        # 实验结果证明使用mask比偏置效果更好
        self.cross_attr_transform = CrossAttributeTransform2(v_lmk=19, v_pose=1, d_k=16, full_adj_matrix=A, use_mask=True, use_value_proj=False)
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
        '''
        self.attn = AttnHeadV10(    # ATT11测试结果90.1%   # ATT10测试结果 89.3
            in_features=attn_channels,
            out_features=attn_channels,
            edge_dim=1,
            time_hidden=64,
            activation=F.elu,
            in_drop=0.0,
            coef_drop=0.0,
            residual=True,)
        
        '''
        self.attn = AttnHeadGAT(            # 测试结果94.1%
            in_features=attn_channels,
            out_features=attn_channels,
            activation=F.elu,
            in_drop=0.0,
            coef_drop=0.0,
            residual=True,
            negative_slope= 0.2,)
        
        '''
        self.attn = AttnHeadGATConcat(    # concat attention 效果不好，acc= 94.8%
            in_features=attn_channels,
            out_features=attn_channels,
            activation=F.elu,
            in_drop=0.0,
            coef_drop=0.0,
            residual=True,
            negative_slope=0.2,)
        '''
        
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
        
        # 每层一个可学习 sigma 参数
        self.layer_sigmas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))   # softplus(0)≈0.693
            for _ in range(len(self.st_gcn_networks))])
        
        self.layer_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))   # sigmoid后约0.5
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
        # print('self.A', self.A.shape, self.A)
        x = self.cross_attr_transform(x)
        
        # ===== 修改：数据预处理移到前面 =====
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # n, m, v, c, t
        x = x.view(n * m, v * c, t)
        # x = self.data_bn(x)  # 如果需要可以取消注释
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)
        # ===== 修改结束 =====
    
        # alpha = torch.sigmoid(self.sim_weight)

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
        x, alpha_x = self.attn(x, bias=bias, g=g)  # atten 11
        # x, alpha_x = self.attn(x, bias=bias)   # 这行注意力机制也可以放在for循环中
        # ===== 修改结束 =====

        # ===== 修改：GCN 层使用 A_eff =====
        # for i, (gcn, importance) in enumerate(zip(self.st_gcn_networks, self.edge_importance)):
            # x, _ = gcn(x, self.A * importance)
        '''  这块是case论文中使用的consin相似度的结果
        ################################################################# cosin相似度
        for gcn, sim_embed, beta_param in zip(self.st_gcn_networks, self.sim_embeds, self.layer_betas):
            
            # 计算相似度，consin 相似度 这个相似度结果92%  case 论文中使用的是这个相似度，测试了高斯核相似度，效果不如余弦相似度
            S = F.cosine_similarity(sim_embed.unsqueeze(1), sim_embed.unsqueeze(0), dim=2)
            S = (S + 1) / 2
            S = S.unsqueeze(0)  # (1, V, V) -> 广播到 (K, V, V)
            beta = torch.sigmoid(beta_param)  # [0, 1]
            S = self.A * (1.0 + beta * S)
            x, _ = gcn(x, self.A * S)   # 这块S乘2次，精度稳稳的在95%以上
        ################################################################## cosin相似度
        '''
        '''
        ################################################################# gaussian相似度
        for gcn, sim_embed, beta_param, sigma_param in zip(self.st_gcn_networks, self.sim_embeds, self.layer_betas, self.layer_sigmas):

            # 可选：先归一化 embedding，让高斯核更稳定
            sim_embed = F.normalize(sim_embed, p=2, dim=-1)   # (V, D)

            # 高斯核相似度
            diff = sim_embed.unsqueeze(1) - sim_embed.unsqueeze(0)   # (V, V, D)
            dist2 = (diff ** 2).sum(dim=2)                           # (V, V)
            # sigma = F.softplus(sigma_param) + 1e-6                   # 保证 sigma > 0
            sigma = 2.0
            S = torch.exp(-dist2 / (2 * sigma ** 2))                 # (V, V), in (0, 1]
            S = S.unsqueeze(0)                                       # (1, V, V)
            beta = torch.sigmoid(beta_param)                         # [0, 1]

            # 保持和原 cosine 版本一致的融合逻辑
            S = self.A * (1.0 + beta * S)
            x, _ = gcn(x, self.A * S)
        ################################################################# gaussian相似度
            '''
        for gcn, sim_embed, beta_param, alpha_param in zip(self.st_gcn_networks, self.sim_embeds, self.layer_betas, self.layer_alphas):
            
            # x, alpha_x = self.attn(x, bias=bias)
            # 先归一化，便于 cosine 和 gaussian 共用
            sim_embed = F.normalize(sim_embed, p=2, dim=-1)   # (V, D)

            # ===== 1) cosine similarity =====
            S_cos = F.cosine_similarity(sim_embed.unsqueeze(1), sim_embed.unsqueeze(0), dim=2)   # (V, V)
            S_cos = (S_cos + 1.0) / 2.0                        # 映射到 [0, 1]

            # ===== 2) gaussian similarity =====
            diff = sim_embed.unsqueeze(1) - sim_embed.unsqueeze(0)   # (V, V, D)
            dist2 = (diff ** 2).sum(dim=2)                           # (V, V)
            sigma = 2.0
            S_gauss = torch.exp(-dist2 / (2 * sigma ** 2))           # (V, V)   

            # ===== 3) fuse cosine and gaussian =====
            alpha = torch.sigmoid(alpha_param)                       # [0, 1]
            S = alpha * S_cos + (1.0 - alpha) * S_gauss             # (V, V)

            # ===== 4) keep original fusion logic =====
            S = S.unsqueeze(0)                                       # (1, V, V)
            beta = torch.sigmoid(beta_param)                         # [0, 1]
            S = self.A * (1.0 + beta * S) # 去掉这一行 91.7%
            # x, _ = gcn(x, self.A * S)
            
            x, _ = gcn(x, S)   # 直接用 S 作为邻接矩阵，效果更好 95.65%
            

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
