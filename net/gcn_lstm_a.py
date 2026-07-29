import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.graph import Graph
import random
import numpy as np
import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# gcn_lstm with bayesian adaptive adjacent metrix


class Model(nn.Module):
    r"""Graph Convolutional Network with LSTM for spatiotemporal data.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        lstm_hidden_size (int): Hidden size of the LSTM layer
        lstm_num_layers (int): Number of layers in the LSTM
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
    """

    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting, **kwargs):
        lstm_hidden_size = kwargs.get('lstm_hidden_size', 256)  # Default value if not provided
        lstm_num_layers = kwargs.get('lstm_num_layers', 2)  # Default value if not provided
        super().__init__()
        random.seed(1)

        # Load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # print('A', A)
        self.register_buffer('A', A)

        # Build GCN layers
        self.gcn_layers = nn.ModuleList((
            GCN(nfeat=in_channels, nclass=64, dropout=0.5),
            GCN(nfeat=64, nclass=128, dropout=0.5),
        ))
        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.gcn_layers
            ])
        else:
            self.edge_importance = [1] * len(self.gcn_layers)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128*19,  # Output channels of GCN
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        # 输入形状: (N, C, T_in, V, M)
        n, c, t, v, m = x.size()
        # print('x_6', x.shape)
        x = x.view(n * m, c, t, v)
        # print('x_6', x.shape)
        x = x.permute(0, 2, 3, 1).contiguous()  # 将特征维度 64 移到最后，形状变为 [512, 30, 19, 2]

        # print('edge_importance', self.edge_importance)
        # Step 2: 通过 GCN 层
        for gcn, importance in zip(self.gcn_layers, self.edge_importance):
            # print('A', self.A.shape)
            # print('x', x.shape)
            x = gcn(x, self.A * importance)  # (N * M, V, out_features)
        n, t, v, c = x.size()
        x = x.reshape(n, t, v * c)  # 或者 x.view(512, 30, -1)
        # print('x_6', x.shape)    # [512, 30, 19, 128]
        # Step 6: 通过 LSTM 层
        lstm_out, _ = self.lstm(x)  # (N, T_in, lstm_hidden_size)

        # Step 7: 时间维度 T_in 的全局平均池化
        x = lstm_out.mean(dim=1)  # (N, lstm_hidden_size)

        # Step 8: 全连接层，输出 (N, 1)
        out = self.fc(x)  # (N, 1)

        return out

    def extract_feature(self, x):
        """Extract features for further analysis."""
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(n * m, v, c * t)  # Combine time and channel dimensions into one

        # Forward GCN
        for gcn, importance in zip(self.gcn_layers, self.edge_importance):
            x = gcn(x, self.A * importance)  # (N * M, V, out_features)

        # Extract features
        x = x.mean(dim=1)  # Global average pooling over vertices (V) -> (N * M, out_features)
        x = x.view(n, m, -1).mean(dim=1)  # Average over instances (M) -> (N, out_features)

        return x


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        # 定义了图卷积层的输入特征数
        self.in_features = in_features
        # 输出特征数
        self.out_features = out_features
        # 权重 in_fratures = 1433 out需要设置
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # print('weight', self.weight.shape)
        # 是否使用偏差
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

# 初始化权重和偏置的数值，采用了较小的随机数，并考虑了输入特征数
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

# 定义了图卷积层的前向传播，输入input是节点的特征表示，A是图的邻接矩阵，
    def forward(self, input, A):
        # print('weight1', self.weight.shape)
        # print('input1', input.shape)
        kernel_size=A.size(0)
        # print('kernel_size', kernel_size)
        batch_size, time_steps, num_nodes, num_features = input.shape  # [512, 30, 19, 2]
        input_reshaped = input.reshape(-1, num_features)  # [512 * 30 * 19, 2]
        # torch.mm()：这个函数用于执行两个张量之间的矩阵乘法
        support = torch.mm(input_reshaped, self.weight)
        support = support.view(batch_size, time_steps, num_nodes, -1)  # [512, 30, 19, 64]
        # print('support', support.shape)
        # print('A', A.shape)
        # 通过邻接矩阵进行稀疏矩阵乘法
        # output = torch.spmm(A, support)
        n, t, v, c = support.size()
        support = support.view(n, t, kernel_size, v,  c // kernel_size)
        output = torch.einsum('kvw,ntkvc->ntkwc', (A, support))
        n, t, k, w, c = output.size()
        output = output.view(n, t, w, -1)
        # print('output', output.shape)
        # 如果定义了偏置，则将其加到输出上
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# 使用图卷积网络进行节点分类
class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN, self).__init__()
        # 两层卷积，nfeat为输入特征数，nhid为隐藏层特征数，nclass为输出类别数，dropout是丢弃率
        self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc2 = GraphConvolution(nclass, nclass)
        self.dropout = dropout

    def forward(self, x, A):
        # 前向传播模型。输入特征进入第一层卷积，然后通过rule激活函数,工程上也可以在relu之前进行batchnormalsize
        x = F.relu(self.gc1(x, A))
        # 防止过拟合
        x = F.dropout(x, self.dropout, training=self.training)
        return x
        # 第二层卷积输出
        # x = self.gc2(x, A)
        # 通过log_softmax函数进行分类
        # return F.log_softmax(x, dim=1)
