import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph, normalize_undigraph
from net.utils.st_gat import (AttnHeadGATConcat, AttnHeadGAT, AttnHeadV10, AttnHeadV11,
                              CrossAttributeTransformSimple, CrossAttributeTransform1,
                              CrossAttributeTransform2, AttnHeadGATMotionEdge)


# =========================================================
#  再STGCN7的基础上 解耦 k 跳邻接 + 对称归一化
# =========================================================
def k_adjacency(A, k, with_self=True, self_factor=1):
    """恰好 k 跳的解耦邻接矩阵 (V, V),numpy 输入输出。"""
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = (np.minimum(np.linalg.matrix_power(A + I, k), 1)
          - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1))
    if with_self:
        Ak = Ak + self_factor * I
    return Ak


def normalize_adjacency_np(A):
    """D^{-1/2} A D^{-1/2},numpy。"""
    deg = A.sum(-1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D = np.diag(deg_inv_sqrt)
    return (D @ A @ D).astype(np.float32)


def normalize_undigraph_torch(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    deg = A.sum(dim=-1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    return deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)


# =========================================================
#  1) 多尺度解耦空间图卷积
# =========================================================
class MultiScaleGraphConv(nn.Module):
    """
    A_binary: (V, V) 二值物理邻接(不含自环)
    num_scales: 尺度数 K(每个尺度捕获恰好 k 跳,k=0..K-1)
    """
    def __init__(self, in_channels, out_channels, A_binary,
                 num_scales=4, dropout=0.0):
        super().__init__()
        self.num_scales = num_scales

        A_powers = [normalize_adjacency_np(k_adjacency(A_binary, k, with_self=True))
                    for k in range(num_scales)]
        A_powers = np.stack(A_powers, axis=0)                 # (K, V, V)
        self.register_buffer('A_powers', torch.tensor(A_powers, dtype=torch.float32))

        # 可学习残差邻接(类似 edge_importance,但作用在每个尺度上)
        self.A_res = nn.Parameter(torch.zeros_like(self.A_powers))

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.drop = nn.Dropout(dropout, inplace=False)

    def forward(self, x, S=None):
        """
        x: (N, C, T, V)
        S: 可选相似度调制 (V, V) 或 (1, V, V),来自你原有机制,只调制连接强度
        """
        N, C, T, V = x.size()
        A = self.A_powers.to(x.dtype) + self.A_res            # (K, V, V)

        if S is not None:
            if S.dim() == 3:
                S = S.squeeze(0)                              # (V, V)
            A = A * (1.0 + S.unsqueeze(0))                    # 广播到 (K, V, V)

        # 多尺度聚合并拼接: (K,V,W) x (N,C,T,W) -> (N,K,C,T,V)
        out = torch.einsum('kvw,nctw->nkctv', A, x)
        out = out.reshape(N, self.num_scales * C, T, V)
        out = self.mlp(out)
        return self.drop(out), A


# =========================================================
#  3) 多尺度时间卷积 (MS-TCN)
# =========================================================
class MultiScaleTCN(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 2, 3, 4),
                 stride=1, dropout=0.0):
        super().__init__()
        num_branches = len(dilations) + 1          # +1 是 maxpool 分支
        branch_ch = channels // num_branches
        assert branch_ch > 0, "channels 太小,减少 dilations 数量"

        self.branches = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size - 1) // 2 * d
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, branch_ch, 1),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_ch, branch_ch, (kernel_size, 1),
                          stride=(stride, 1), padding=(pad, 0), dilation=(d, 1)),
                nn.BatchNorm2d(branch_ch),
            ))
        # maxpool 分支
        self.branches.append(nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((kernel_size, 1), stride=(stride, 1),
                        padding=((kernel_size - 1) // 2, 0)),
            nn.BatchNorm2d(branch_ch),
        ))

        self.out = nn.Sequential(
            nn.Conv2d(branch_ch * num_branches, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout, inplace=False),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        out = torch.cat(feats, dim=1)
        return self.out(out)


# =========================================================
#  改写后的 st_gcn:多尺度空间聚合 + 多尺度时间卷积
# =========================================================
class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A_binary, num_scales=4,
                 stride=1, temporal_kernel=3, dropout=0, residual=True):
        super().__init__()

        self.gcn = MultiScaleGraphConv(in_channels, out_channels,
                                       A_binary, num_scales=num_scales)
        self.tcn = MultiScaleTCN(out_channels, kernel_size=temporal_kernel,
                                 dilations=(1, 2, 3, 4), stride=stride,
                                 dropout=dropout)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, S=None):
        res = self.residual(x)
        x, A = self.gcn(x, S)
        x = self.relu(x)          # gcn 后激活
        x = self.tcn(x) + res
        return self.relu(x), A


# =========================================================
#  Model:保留你所有的相似度 / 注意力机制
# =========================================================
class Model(nn.Module):
    def save_output_to_file(self, filename):
        with open(filename, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            try:
                print('A', self.graph.A.shape, self.graph.A)
            finally:
                sys.stdout = original_stdout

    def save_adjacency_data(self, save_dir, tag=""):
        os.makedirs(save_dir, exist_ok=True)
        A_physical_path = os.path.join(save_dir, 'A_physical.npy')
        if not os.path.exists(A_physical_path):
            A_np = self.A.detach().cpu().numpy()
            np.save(A_physical_path, A_np)
            print(f"✅ Saved: {A_physical_path}  shape={A_np.shape}")
        if hasattr(self, '_cached_S') and self._cached_S is not None:
            S_np = self._cached_S.detach().cpu().numpy()
            filepath = os.path.join(save_dir, f'S_{tag}.npy')
            np.save(filepath, S_np)
            print(f"✅ Saved: {filepath}  shape={S_np.shape}")
        else:
            print("[WARN] No cached S, run a forward pass first.")

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, num_scales=4, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.graph = Graph(**graph_args)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)                          # (K, V, V) 保留给注意力用

        num_nodes = A.size(1)                                 # V

        # ===== 构造二值物理邻接(去自环),供多尺度图卷积使用 =====
        A_binary = (self.graph.A.sum(0) > 0).astype(np.float32)   # (V, V)
        np.fill_diagonal(A_binary, 0.0)

        # 提示:V 较小时高尺度可能全连满,可打印检查后调小 num_scales
        self.num_scales = num_scales

        # 跨属性变换(保持你原来的配置)
        self.cross_attr_transform = CrossAttributeTransform2(
            v_lmk=28, v_pose=1, d_k=16,
            full_adj_matrix=A, use_mask=True, use_value_proj=False)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # ===== 多尺度 st_gcn 主干(结构与你原来层数一致)=====
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64,  A_binary, num_scales, stride=1, residual=False),
            st_gcn(64,          64,  A_binary, num_scales, stride=1),
            st_gcn(64,          128, A_binary, num_scales, stride=2),
            st_gcn(128,         128, A_binary, num_scales, stride=1),
            st_gcn(128,         256, A_binary, num_scales, stride=2),
            st_gcn(256,         256, A_binary, num_scales, stride=1),
        ))

        # ===== 注意力(保持你的最终版本)=====
        attn_channels = in_channels
        self.attn = AttnHeadGATMotionEdge(
            in_features=attn_channels,
            out_features=attn_channels,
            edge_dim=1,
            activation=F.elu,
            in_drop=0.0,
            coef_drop=0.0,
            residual=True,
            negative_slope=0.2,
        )

        # ===== 每层独立的相似度嵌入 / 融合系数 =====
        embed_dim = 32
        self.sim_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(num_nodes, embed_dim))
            for _ in self.st_gcn_networks])
        self.layer_betas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in self.st_gcn_networks])
        self.layer_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in self.st_gcn_networks])

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def _build_similarity(self, sim_embed, alpha_param):
        """cosine + gaussian 融合,返回 (V, V) 的相似度矩阵。"""
        sim_embed = F.normalize(sim_embed, p=2, dim=-1)

        S_cos = F.cosine_similarity(sim_embed.unsqueeze(1),
                                    sim_embed.unsqueeze(0), dim=2)
        S_cos = (S_cos + 1.0) / 2.0

        diff = sim_embed.unsqueeze(1) - sim_embed.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=2)
        sigma = 2.0
        S_gauss = torch.exp(-dist2 / (2 * sigma ** 2))

        alpha = torch.sigmoid(alpha_param)
        return alpha * S_cos + (1.0 - alpha) * S_gauss        # (V, V)

    # def forward(self, x):
    def forward(self, x, return_attn=False):
        n, c, t, v, m = x.size()
        x = self.cross_attr_transform(x)

        # 数据标准化
        x = x.permute(0, 4, 3, 1, 2).contiguous()             # n, m, v, c, t
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        # ===== 进主干前的全局时空注意力(保留)=====
        A0 = (self.A.sum(dim=0) > 0)                          # (V, V) bool
        bias_vv = torch.where(
            A0,
            torch.zeros((v, v), device=x.device),
            torch.full((v, v), -1e9, device=x.device))
        bias = bias_vv.view(1, 1, v, v)

        S_attn = self._build_similarity(self.sim_embeds[0], self.layer_alphas[0])
        S_attn = self.A * S_attn                              # 沿用你原来的调制
        x, alpha_x = self.attn(x, S_attn, bias=bias)

        # ===== 多尺度 GCN 主干:每层用相似度 S 调制连接强度 =====
        S_last = None
        for gcn, sim_embed, beta_param, alpha_param in zip(
                self.st_gcn_networks, self.sim_embeds,
                self.layer_betas, self.layer_alphas):

            S = self._build_similarity(sim_embed, alpha_param)   # (V, V)
            beta = torch.sigmoid(beta_param)
            S = beta * S                                         # 只提供“增量”调制
            x, _ = gcn(x, S)                                     # 传入多尺度图卷积
            S_last = S

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(n, m, -1, 1, 1).mean(dim=1)

        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        self._cached_S = S_last
        if return_attn:
            return x, {
                'alpha_x': alpha_x.detach(),
                'S_attn': S_attn.detach(),
                'A_eff': S_last.detach()
            }
        return x