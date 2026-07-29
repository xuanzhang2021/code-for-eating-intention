import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttnHeadV8(nn.Module):   # 所有的特征拼接进 logits 输入（最直观）
    """
    GAT attention head (你的张量排布: seq_ini: [n, c, t, v, m])

    按你前面定义的逻辑，把 “接收端 i 的速度/加速度 (Δt)” 拼进 e_{t,ij}：

      z_{t,ij} = [ h_{t,i} || h_{t,j} || g_{ij} || v_{t,i} || a_{t,i} ]
      e_{t,ij} = a(z_{t,ij})
      α_{t,ij} = softmax_j(e_{t,ij} + bias_{ij})
      y_{t,i} = Σ_j α_{t,ij} * value_{t,j}

    其中 v/a 仅取接收端 i，因此对同一个 i、t，所有 j 共享同一份 v_{t,i}, a_{t,i}。
    """

    def __init__(
        self,
        in_features: int,          # c (输入通道，比如 3)
        out_features: int,         # F
        edge_dim: int = 1,         # 这里把 A_ij 当成标量 -> 1 维再映射
        activation=F.elu,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
        residual: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.negative_slope = negative_slope

        # h_{t,i} = W_x x_{t,i}
        self.W_x = nn.Linear(in_features, out_features, bias=False)

        # value: u_{t,j} = W_v x_{t,j}  (也可以用 h 直接做 value，但这里更清晰)
        self.W_v = nn.Linear(in_features, out_features, bias=False)

        # g_{ij} = W_A A_{ij}，A_{ij} 为标量时 edge_dim=1
        self.W_A = nn.Linear(edge_dim, out_features, bias=False)

        # a(·): 把拼接后的向量 -> 标量 logit
        # z 维度: 2F + F(edge_embed) + 3(v) + 3(a) = 3F + 6
        self.attn_fc1 = nn.Linear(3 * out_features + 6, out_features)
        self.attn_fc2 = nn.Linear(out_features, 1)

        # residual projection (如果 c != F)
        if residual and in_features != out_features:
            # seq_ini 是 [n,c,t,v,m]，Conv3d 期望 [n,c,D,H,W] -> 把 (t,v,m) 当作 (D,H,W)
            self.res_conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=False)
        else:
            self.res_conv = None

    @staticmethod
    def _delta_t(x_tm: torch.Tensor) -> torch.Tensor:
        """
        x_tm: [n,m,t,v,c]
        return: Δx: [n,m,t,v,c] with t=0 padded zeros
        """
        n, m, t, v, c = x_tm.shape
        dx = torch.zeros_like(x_tm)
        if t >= 2:
            dx[:, :, 1:, :, :] = x_tm[:, :, 1:, :, :] - x_tm[:, :, :-1, :, :]
        return dx

    def forward(self, seq_ini: torch.Tensor, bias_mat: torch.Tensor) -> torch.Tensor:
        """
        seq_ini: [n, c, t, v, m]
        bias_mat: 期望是 [v, v] 或 [n, v, v] 或 [1, v, v]，用于 mask/bias
        return: [n, out_features, t, v, m]
        """
        if self.in_drop > 0.0:
            seq_ini = F.dropout(seq_ini, p=self.in_drop, training=self.training)

        # -> [n,m,t,v,c]
        seq = seq_ini.permute(0, 4, 2, 3, 1).contiguous()
        n, m, t, v, c = seq.shape
        assert c == self.in_features, f"in_features mismatch: got {c}, expected {self.in_features}"

        # ------- 构造接收端 i 的 v_{t,i}, a_{t,i} (Δt) -------
        vel = self._delta_t(seq)          # [n,m,t,v,c]  (若 c=3 就是 3D 速度)
        acc = self._delta_t(vel)          # [n,m,t,v,c]

        # ------- 计算 h 与 value -------
        # h: [n,m,t,v,F]
        h = self.W_x(seq)
        # value: [n,m,t,v,F]
        val = self.W_v(seq)

        # ------- 处理 bias_mat 到 [n,m,t,v,v] -------
        if isinstance(bias_mat, np.ndarray):
            bias_mat = torch.tensor(bias_mat, dtype=seq.dtype, device=seq.device)

        if bias_mat.dim() == 2:
            # [v,v] -> [n,m,t,v,v]
            bias = bias_mat[None, None, None, :, :].expand(n, m, t, v, v)
        elif bias_mat.dim() == 3:
            # [n,v,v] -> [n,m,t,v,v]
            bias = bias_mat[:, None, None, :, :].expand(n, m, t, v, v)
        elif bias_mat.dim() == 5:
            # already [n,m,t,v,v]
            bias = bias_mat
        else:
            raise ValueError(f"Unsupported bias_mat shape: {tuple(bias_mat.shape)}")

        # ------- 图结构项 g_{ij} -------
        # 这里假设 A_ij 就是 bias_mat 的“是否连边”信息不合适，因为 bias 常含 -inf。
        # 更常见：你另有邻接 A (0/1)。若你只有 bias_mat，这里给一个简化：用 A_ij = 1(bias > -1e8)
        A = (bias > -1e8).to(seq.dtype)  # [n,m,t,v,v]  (0/1)
        g = self.W_A(A.unsqueeze(-1))    # [n,m,t,v,v,F]

        # ------- 拼接构造 z_{t,ij}，其中 v/a 取接收端 i -------
        # h_i: [n,m,t,v,1,F] -> expand 到 j 维
        h_i = h.unsqueeze(4).expand(n, m, t, v, v, self.out_features)
        # h_j: [n,m,t,1,v,F] -> expand 到 i 维
        h_j = h.unsqueeze(3).expand(n, m, t, v, v, self.out_features)

        # vel_i / acc_i: [n,m,t,v,c] -> [n,m,t,v,1,c] -> expand 到 j 维
        vel_i = vel.unsqueeze(4).expand(n, m, t, v, v, c)
        acc_i = acc.unsqueeze(4).expand(n, m, t, v, v, c)

        # z: [n,m,t,v,v, 3F+2c]；当 c=3 -> 3F+6
        z = torch.cat([h_i, h_j, g, vel_i, acc_i], dim=-1)

        # ------- e_{t,ij} = a(z_{t,ij}) -------
        e = self.attn_fc2(F.leaky_relu(self.attn_fc1(z), negative_slope=self.negative_slope)).squeeze(-1)
        # e: [n,m,t,v,v]

        # 加 bias/mask 后 softmax（沿 j 维）
        e = e + bias
        alpha = F.softmax(e, dim=-1)  # [n,m,t,v,v]

        if self.coef_drop > 0.0:
            alpha = F.dropout(alpha, p=self.coef_drop, training=self.training)

        if self.in_drop > 0.0:
            val = F.dropout(val, p=self.in_drop, training=self.training)

        # ------- 聚合 y_{t,i} = Σ_j α_{ij} * value_j -------
        # alpha: [n,m,t,i,j], val: [n,m,t,j,F] -> out: [n,m,t,i,F]
        out = torch.einsum("bmti j, bmtj f -> bmtif", alpha, val)

        # ------- residual -------
        if self.residual:
            if self.res_conv is not None:
                res = self.res_conv(seq_ini).permute(0, 4, 2, 3, 1).contiguous()  # [n,m,t,v,F]
                out = out + res
            else:
                # in_features == out_features 时，直接加回 seq(作为特征)不太对，因为 seq 的 c==F 才行
                # 这里用 val 的输入投影后的形态更统一：out += val
                out = out + val

        # -> [n, F, t, v, m]
        out = out.permute(0, 4, 2, 3, 1).contiguous()

        return self.activation(out)
    
class AttnHeadV9(nn.Module):
    def __init__(
        self,
        in_features: int,          # c
        out_features: int,         # F
        edge_dim: int = 1,         # A_ij 的维度（标量则=1）
        time_hidden: int = 64,
        activation=F.elu,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
        residual: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.negative_slope = negative_slope

        self.W_x = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)

        # edge embed: g_ij = W_A A_ij  -> F
        self.W_A = nn.Linear(edge_dim, out_features, bias=False)

        # 空间logit MLP: [h_i(F), h_j(F), g_ij(F)] => 3F -> F -> 1
        self.attn_fc1 = nn.Linear(3 * out_features, out_features)
        self.attn_fc2 = nn.Linear(out_features, 1)

        # 时间偏置 MLP: [vel_i(c), acc_i(c)] => 2c -> hidden -> 1
        self.time_fc1 = nn.Linear(2 * in_features, time_hidden)
        self.time_fc2 = nn.Linear(time_hidden, 1)

        # residual projection
        if residual and in_features != out_features:
            self.res_conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=False)
        else:
            self.res_conv = None

    @staticmethod
    def _delta_t(x_tm: torch.Tensor) -> torch.Tensor:
        dx = torch.zeros_like(x_tm)
        if x_tm.shape[2] >= 2:
            dx[:, :, 1:, :, :] = x_tm[:, :, 1:, :, :] - x_tm[:, :, :-1, :, :]
        return dx

    def forward(self, x, bias, g):
        """
        x:    [n, c, t, v, m]
        bias: [n, m, t, v, v] (或可广播)
        g:    A_ij 或边特征: [n, m, t, v, v, edge_dim]
              （如果你只有 [v,v] 的邻接，可在外部扩到这个形状）
        """
        n, c, t, v, m = x.shape

        # [n,m,t,v,c]
        x_nm = x.permute(0, 4, 2, 3, 1).contiguous()

        if self.in_drop > 0:
            x_nm = F.dropout(x_nm, p=self.in_drop, training=self.training)

        # h/value: [n,m,t,v,F]
        h = self.W_x(x_nm)
        val = self.W_v(x_nm)

        # vel/acc: [n,m,t,v,c]
        vel = self._delta_t(x_nm)
        acc = self._delta_t(vel)

        # edge embed: g_emb: [n,m,t,v,v,F]
        g_emb = self.W_A(g)

        Fout = h.shape[-1]
        h_i = h.unsqueeze(4).expand(n, m, t, v, v, Fout)
        h_j = h.unsqueeze(3).expand(n, m, t, v, v, Fout)

        # 空间logits e_s: [n,m,t,v,v]
        z_s = torch.cat([h_i, h_j, g_emb], dim=-1)  # 3F
        e_s = self.attn_fc2(F.leaky_relu(self.attn_fc1(z_s), negative_slope=self.negative_slope)).squeeze(-1)

        # 时间偏置 beta_i: [n,m,t,v]
        s_i = torch.cat([vel, acc], dim=-1)  # 2c
        beta = self.time_fc2(F.leaky_relu(self.time_fc1(s_i), negative_slope=self.negative_slope)).squeeze(-1)

        # 合成 logits: [n,m,t,v,v]
        e = e_s + beta.unsqueeze(-1) + bias

        # attention
        alpha = torch.softmax(e, dim=-1)
        if self.coef_drop > 0:
            alpha = F.dropout(alpha, p=self.coef_drop, training=self.training)

        # 聚合: val_j
        val_e = val.unsqueeze(3).expand(n, m, t, v, v, Fout)  # [n,m,t,i,j,F]
        y = (alpha.unsqueeze(-1) * val_e).sum(dim=4)          # [n,m,t,v,F]

        # residual
        if self.residual:
            if self.res_conv is not None:
                res = self.res_conv(x).permute(0, 4, 2, 3, 1).contiguous()  # -> [n,m,t,v,F]
            else:
                res = x_nm  # [n,m,t,v,c]==F
            y = y + res

        # activation
        if self.activation is not None:
            y = self.activation(y)

        # back to [n,F,t,v,m]
        y = y.permute(0, 4, 2, 3, 1).contiguous()
        return y, alpha


class AttnHeadV10(nn.Module):
    def __init__(
        self,
        in_features: int,          # C
        out_features: int,         # F
        edge_dim: int = 1,         # 边特征最后一维
        time_hidden: int = 64,
        activation=F.elu,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
        residual: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.negative_slope = negative_slope

        # h = W_x x, value = W_v x
        self.W_x = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        # edge embed: g_ij -> F
        self.W_A = nn.Linear(edge_dim, out_features, bias=False)

        self.attn_i = nn.Linear(out_features, 1, bias=False)
        self.attn_j = nn.Linear(out_features, 1, bias=False)
        self.attn_g = nn.Linear(out_features, 1, bias=False)
        self.attn_out = nn.Linear(out_features, 1, bias=True)

        # 时间偏置 MLP: [vel_i, acc_i] => 2C -> hidden -> 1
        self.time_fc1 = nn.Linear(in_features, 1)
        # self.time_fc2 = nn.Linear(time_hidden, 1)

        # residual projection: [N,C,T,V] 用 Conv2d
        if residual and in_features != out_features:
            self.res_conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=False)
        else:
            self.res_conv = None

    @staticmethod
    def _delta_t_4d(x_ntvc: torch.Tensor) -> torch.Tensor:
        """
        x_ntvc: [N, T, V, C]
        return: [N, T, V, C]，时间差分（t维）
        """
        dx = torch.zeros_like(x_ntvc)
        if x_ntvc.shape[1] >= 2:
            dx[:, 1:, :, :] = x_ntvc[:, 1:, :, :] - x_ntvc[:, :-1, :, :]
        return dx

    def forward(self, x, bias, g):
        """
        x:    [N, C, T, V]
        bias: 可广播到 [N, T, V, V]（例如 [1,1,V,V] 或 [1,T,V,V]）
        g:    边特征，可广播到 [N, T, V, V, edge_dim]
              如果你只有 [V,V]，请扩成 [1,1,V,V,1]
        """
        assert x.dim() == 4, f"x must be 4D [N,C,T,V], got {x.shape}"
        N, C, T, V = x.shape

        # [N, T, V, C]
        x_ntvc = x.permute(0, 2, 3, 1).contiguous()

        if self.in_drop > 0:
            x_ntvc = F.dropout(x_ntvc, p=self.in_drop, training=self.training)

        # h/value: [N, T, V, F]
        h = self.W_x(x_ntvc)
        val = self.W_v(x_ntvc)

        # vel/acc: [N, T, V, C]
        vel = self._delta_t_4d(x_ntvc)
        # acc = self._delta_t_4d(vel)

        # g_emb: [N, T, V, V, F]
        # 要求 g 最后一维是 edge_dim
        # print('g.shape:', g.shape)  [1, 1, 68, 68, 3]
        g_emb = self.W_A(g)
        # 显式广播到 [N, T, V, V, F]，以便后续 torch.cat
        if g_emb.shape[0] == 1 and N != 1:
            g_emb = g_emb.expand(N, -1, -1, -1, -1)
        if g_emb.shape[1] == 1 and T != 1:
            g_emb = g_emb.expand(-1, T, -1, -1, -1)

        Fout = h.shape[-1]

        # h: [N,T,V,F], g_emb: [1,1,V,V,F] or [N,T,V,V,F]
        h_i = self.attn_i(h)                       # [N,T,V,1]
        h_j = self.attn_j(h).transpose(2, 3)       # [N,T,1,V]
        A_ij = self.attn_g(g_emb).squeeze(-1)       # [*,*,V,V]

        e_s = h_i + h_j + A_ij                      # [N,T,V,V]
        # e_s = h_i + h_j
        e_s = F.leaky_relu(e_s, negative_slope=self.negative_slope)

        # ---------- 时间偏置 beta: [N,T,V] ----------
        # s_i = torch.cat([vel, acc], dim=-1)    # [N,T,V,2C]
        # s_i = vel                          # [N,T,V,C]
        # beta = self.time_fc1(s_i).squeeze(-1)                          # [N,T,V]
        # e_s = e_s + beta.unsqueeze(-1)
        e = F.leaky_relu(e_s, negative_slope=self.negative_slope) + bias

        # print('s_i.shape:', s_i.shape)   s_i.shape: torch.Size([256, 30, 68, 6])
        # beta = self.time_fc2(F.leaky_relu(self.time_fc1(s_i), negative_slope=self.negative_slope)).squeeze(-1)                          # [N,T,V]
        # beta = F.leaky_relu(self.time_fc1(s_i), negative_slope=self.negative_slope).squeeze(-1)                          # [N,T,V]
        # 合成 logits: [N, T, V, V]
        # e = e_s + beta.unsqueeze(-1) + bias

        # attention: 对 j 维 softmax（最后一维）
        alpha = torch.softmax(e, dim=-1)
        if self.coef_drop > 0:
            alpha = F.dropout(alpha, p=self.coef_drop, training=self.training)

        # ---------- 聚合（用 matmul，避免 val_e=[N,T,V,V,F]） ----------
        # alpha: [N,T,V,V], val: [N,T,V,F]
        # 需要 val_j: [N,T,V,F]，对 j 做加权和 -> [N,T,V,F]
        y = torch.matmul(alpha, val)           # [N,T,V,F]

        # residual
        if self.residual:
            if self.res_conv is not None:
                res = self.res_conv(x)         # [N,F,T,V]
                res = res.permute(0, 2, 3, 1).contiguous()  # [N,T,V,F]
            else:
                res = x_ntvc                   # 这里要求 C == Fout
            y = y + res

        if self.activation is not None:
            y = self.activation(y)

        # back to [N,F,T,V]
        y = y.permute(0, 3, 1, 2).contiguous()
        return y, alpha

# 把运动特征直接融入到权重里面
class AttnHeadV11(nn.Module):
    def __init__(
        self,
        in_features: int,          # C
        out_features: int,         # F
        edge_dim: int = 1,         # 边特征最后一维
        time_hidden: int = 64,     # 运动门控 MLP 隐藏层维度
        activation=F.elu,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
        residual: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.negative_slope = negative_slope

        # h = W_x x, value = W_v x
        self.W_x = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        
        # edge embed: g_ij -> F
        self.W_A = nn.Linear(edge_dim, out_features, bias=False)

        # 注意力参数
        self.attn_i = nn.Linear(out_features, 1, bias=False)
        self.attn_j = nn.Linear(out_features, 1, bias=False)
        self.attn_g = nn.Linear(out_features, 1, bias=False)

        # ========== 方案A: 运动门控（作用于 Value） ==========
        # 输入: [vel, acc] 拼接 -> 2C
        # 输出: 门控信号 -> out_features
        self.motion_gate = nn.Sequential(
            nn.Linear(in_features * 2, time_hidden),
            nn.LeakyReLU(negative_slope),
            nn.Linear(time_hidden, out_features),
            nn.Sigmoid()  # 输出 (0, 1)
        )

        # ========== 方案C: 运动注意力偏置（目标节点偏置） ==========
        # 输入: [vel, acc] -> 2C
        # 输出: 标量偏置 -> 1
        self.motion_bias = nn.Sequential(
            nn.Linear(in_features * 2, time_hidden),
            nn.LeakyReLU(negative_slope),
            nn.Linear(time_hidden, 1)
        )

        # residual projection
        if residual and in_features != out_features:
            self.res_conv = nn.Conv2d(
                in_channels=in_features, 
                out_channels=out_features, 
                kernel_size=1, 
                bias=False
            )
        else:
            self.res_conv = None

    @staticmethod
    def _delta_t_4d(x_ntvc: torch.Tensor) -> torch.Tensor:
        """计算时间维度的一阶差分（速度/加速度）"""
        dx = torch.zeros_like(x_ntvc)
        if x_ntvc.shape[1] >= 2:
            dx[:, 1:, :, :] = x_ntvc[:, 1:, :, :] - x_ntvc[:, :-1, :, :]
        return dx

    def forward(self, x, bias, g):
        """
        参数:
            x:    [N, C, T, V] 节点特征
            bias: 可广播到 [N, T, V, V] 的邻接偏置
            g:    边特征，可广播到 [N, T, V, V, edge_dim]
        
        返回:
            y:     [N, F, T, V] 输出特征
            alpha: [N, T, V, V] 注意力权重
        """
        assert x.dim() == 4, f"x must be 4D [N,C,T,V], got {x.shape}"
        N, C, T, V = x.shape

        # ========== 维度转换 ==========
        # [N, C, T, V] -> [N, T, V, C]
        x_ntvc = x.permute(0, 2, 3, 1).contiguous()

        if self.in_drop > 0:
            x_ntvc = F.dropout(x_ntvc, p=self.in_drop, training=self.training)

        # ========== 计算 h 和 value ==========
        h = self.W_x(x_ntvc)      # [N, T, V, F]
        val = self.W_v(x_ntvc)    # [N, T, V, F]

        # ========== 计算运动特征 ==========
        vel = self._delta_t_4d(x_ntvc)           # [N, T, V, C] 速度
        acc = self._delta_t_4d(vel)              # [N, T, V, C] 加速度
        motion = torch.cat([vel, acc], dim=-1)   # [N, T, V, 2C]

        # ========== 方案A: 运动门控（作用于 Value） ==========
        # gate: [N, T, V, F]，每个节点每个时刻的门控信号
        gate = self.motion_gate(motion)          # [N, T, V, F]
        # val = val * gate                        # 逐元素乘法

        # ========== 方案C: 运动注意力偏置（目标节点偏置） ==========
        # beta_j: [N, T, V]，运动剧烈的节点 j 获得更高偏置
        beta = self.motion_bias(motion).squeeze(-1)  # [N, T, V]

        # ========== 边特征处理 ==========
        g_emb = self.W_A(g)  # [..., V, V, F]
        
        # 显式广播到 [N, T, V, V, F]
        if g_emb.shape[0] == 1 and N != 1:
            g_emb = g_emb.expand(N, -1, -1, -1, -1)
        if g_emb.shape[1] == 1 and T != 1:
            g_emb = g_emb.expand(-1, T, -1, -1, -1)

        # ========== 计算注意力分数 ==========
        h_i = self.attn_i(h)                     # [N, T, V, 1]
        h_j = self.attn_j(h).transpose(2, 3)    # [N, T, 1, V]
        A_ij = self.attn_g(g_emb).squeeze(-1)   # [N, T, V, V]

        # 空间注意力 + 目标节点运动偏置
        # beta.unsqueeze(-2): [N, T, 1, V]，广播到 [N, T, V, V]
        # 效果: 运动剧烈的节点 j 会被所有节点 i 更多地关注
        e_s = h_i + h_j + A_ij + beta.unsqueeze(-2)  # [N, T, V, V]
        # e_s = h_i + h_j + A_ij   # 只有空间关系作为注意力分数
        
        e = F.leaky_relu(e_s, negative_slope=self.negative_slope) + bias

        # ========== Softmax 归一化 ==========
        alpha = torch.softmax(e, dim=-1)         # [N, T, V, V]
        
        if self.coef_drop > 0:
            alpha = F.dropout(alpha, p=self.coef_drop, training=self.training)

        # ========== 聚合（使用门控后的 value） ==========
        # alpha: [N, T, V, V], val: [N, T, V, F]
        # y_i = sum_j(alpha_ij * val_j)
        y = torch.matmul(alpha, val)             # [N, T, V, F]

        # ========== 残差连接 ==========
        if self.residual:
            if self.res_conv is not None:
                res = self.res_conv(x)           # [N, F, T, V]
                res = res.permute(0, 2, 3, 1).contiguous()  # [N, T, V, F]
            else:
                res = x_ntvc                     # 要求 C == F
            y = y + res

        # ========== 激活函数 ==========
        if self.activation is not None:
            y = self.activation(y)

        # ========== 维度转换回原格式 ==========
        # [N, T, V, F] -> [N, F, T, V]
        y = y.permute(0, 3, 1, 2).contiguous()
        
        return y, alpha
    

# 下面一个注意力机制是几个特征加载一起
class AttnHeadGAT(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=F.elu,
        in_drop: float = 0.0,
        coef_drop: float = 0.0,
        residual: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.negative_slope = negative_slope

        # 标准 GAT: 一个线性变换
        self.W = nn.Linear(in_features, out_features, bias=False)

        # 注意力参数
        self.attn_i = nn.Linear(out_features, 1, bias=False)
        self.attn_j = nn.Linear(out_features, 1, bias=False)

        # residual projection
        if residual and in_features != out_features:
            self.res_conv = nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=1,
                bias=False
            )
        else:
            self.res_conv = None

    def forward(self, x, bias):
        """
        参数:
            x:    [N, C, T, V] 节点特征
            bias: 可广播到 [N, T, V, V] 的邻接偏置

        返回:
            y:     [N, F, T, V]
            alpha: [N, T, V, V]
        """
        assert x.dim() == 4, f"x must be 4D [N,C,T,V], got {x.shape}"
        N, C, T, V = x.shape

        # [N, C, T, V] -> [N, T, V, C]
        x_ntvc = x.permute(0, 2, 3, 1).contiguous()

        if self.in_drop > 0:
            x_ntvc = F.dropout(x_ntvc, p=self.in_drop, training=self.training)

        # 节点特征映射
        h = self.W(x_ntvc)   # [N, T, V, F]

        # 注意力分数
        h_i = self.attn_i(h)                  # [N, T, V, 1]
        h_j = self.attn_j(h).transpose(2, 3)  # [N, T, 1, V]

        e = F.leaky_relu(h_i + h_j, negative_slope=self.negative_slope) + bias
        alpha = torch.softmax(e, dim=-1)      # [N, T, V, V]

        if self.coef_drop > 0:
            alpha = F.dropout(alpha, p=self.coef_drop, training=self.training)

        # 聚合
        y = torch.matmul(alpha, h)            # [N, T, V, F]

        # residual
        if self.residual:
            if self.res_conv is not None:
                res = self.res_conv(x)        # [N, F, T, V]
                res = res.permute(0, 2, 3, 1).contiguous()
            else:
                res = x_ntvc                  # 只有 C == F 时才合法
            y = y + res

        if self.activation is not None:
            y = self.activation(y)

        # [N, T, V, F] -> [N, F, T, V]
        y = y.permute(0, 3, 1, 2).contiguous()

        return y, alpha
    
# 下面一个注意力机制是几个特征拼接在一起
class AttnHeadGATConcat(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation=F.elu,
        in_drop=0.0,
        coef_drop=0.0,
        residual=True,
        negative_slope=0.2,
    ):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Linear(2 * out_features, 1, bias=False)
        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.negative_slope = negative_slope

        if residual and in_features != out_features:
            self.res_conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        else:
            self.res_conv = None

    def forward(self, x, bias):
        N, C, T, V = x.shape
        x_ntvc = x.permute(0, 2, 3, 1).contiguous()   # [N,T,V,C]

        if self.in_drop > 0:
            x_ntvc = F.dropout(x_ntvc, p=self.in_drop, training=self.training)

        h = self.W(x_ntvc)  # [N,T,V,F]

        h_i = h.unsqueeze(3).expand(-1, -1, -1, V, -1)   # [N,T,V,V,F]
        h_j = h.unsqueeze(2).expand(-1, -1, V, -1, -1)   # [N,T,V,V,F]
        h_cat = torch.cat([h_i, h_j], dim=-1)            # [N,T,V,V,2F]

        e = self.attn(h_cat).squeeze(-1)                 # [N,T,V,V]
        # e = F.leaky_relu(e, negative_slope=self.negative_slope) + bias
        e = F.leaky_relu(e, negative_slope=self.negative_slope)


        alpha = torch.softmax(e, dim=-1)

        if self.coef_drop > 0:
            alpha = F.dropout(alpha, p=self.coef_drop, training=self.training)

        y = torch.matmul(alpha, h)                       # [N,T,V,F]

        if self.residual:
            if self.res_conv is not None:
                res = self.res_conv(x).permute(0, 2, 3, 1).contiguous()
            else:
                res = x_ntvc
            y = y + res

        if self.activation is not None:
            y = self.activation(y)

        y = y.permute(0, 3, 1, 2).contiguous()
        return y, alpha
    
'''  下面这个函数是已经调试通的，并且运行良好的，截至202605016
class AttnHeadGATMotionEdge(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        edge_dim,
        activation=F.elu,
        in_drop=0.0,
        coef_drop=0.0,
        residual=True,
        negative_slope=0.2,
    ):
        super().__init__()

        # 节点投影
        self.W_x = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)

        # 边特征映射
        self.W_g = nn.Linear(edge_dim, out_features, bias=False)

        # 注意力向量
        self.u_s = nn.Parameter(torch.empty(out_features))
        self.u_t = nn.Parameter(torch.empty(out_features))
        self.u_g = nn.Parameter(torch.empty(out_features))

        # 运动偏置 beta_j(t) = MLP(m_j(t))
        self.motion_mlp = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, 1)
        )

        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.negative_slope = negative_slope

        if residual and in_features != out_features:
            self.res_conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        else:
            self.res_conv = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_g.weight)

        nn.init.xavier_uniform_(self.u_s.unsqueeze(0))
        nn.init.xavier_uniform_(self.u_t.unsqueeze(0))
        nn.init.xavier_uniform_(self.u_g.unsqueeze(0))

        for m in self.motion_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.res_conv is not None:
            nn.init.xavier_uniform_(self.res_conv.weight)

    def forward(self, x, edge_feat, bias=None):
        """
        x:         [N, C, T, V]
        edge_feat: [N, T, V, V, G]
        bias:      [N, T, V, V] or None
        """
        N, C, T, V = x.shape
        x_ntvc = x.permute(0, 2, 3, 1).contiguous()   # [N,T,V,C]

        if self.in_drop > 0:
            x_ntvc = F.dropout(x_ntvc, p=self.in_drop, training=self.training)

        # 1) 节点投影
        h = self.W_x(x_ntvc)       # [N,T,V,F]
        value = self.W_v(x_ntvc)   # [N,T,V,F]

        # 2) 运动特征
        v = torch.zeros_like(x_ntvc)   # [N,T,V,C]
        v[:, 1:] = x_ntvc[:, 1:] - x_ntvc[:, :-1]

        a = torch.zeros_like(x_ntvc)   # [N,T,V,C]
        a[:, 1:] = v[:, 1:] - v[:, :-1]

        m = torch.cat([v, a], dim=-1)  # [N,T,V,2C]

        # 3) 运动偏置 beta_j(t)
        beta = self.motion_mlp(m).squeeze(-1)   # [N,T,V]
        beta_j = beta.unsqueeze(2)              # [N,T,1,V]
        
        # 这段if代码是临时加的
        if edge_feat.dim() == 3 and edge_feat.size(0) == 1:
            edge_feat = edge_feat.squeeze(0)   # [20,20]
            edge_feat = edge_feat.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,20,20,1]
            edge_feat = edge_feat.expand(N, T, V, V, 1)

        # 4) 边特征映射
        g_hat = self.W_g(edge_feat)   # [N,T,V,V,F]

        # 5) 分项注意力打分
        score_s = torch.einsum('ntif,f->nti', h, self.u_s).unsqueeze(-1)   # [N,T,V,1]
        score_t = torch.einsum('ntjf,f->ntj', h, self.u_t).unsqueeze(-2)   # [N,T,1,V]
        score_g = torch.einsum('ntijf,f->ntij', g_hat, self.u_g)           # [N,T,V,V]

        # score = score_s + score_t + score_g + beta_j                       # [N,T,V,V]
        # score = score_s + score_t + beta_j    # 92.3
        # score = score_s   # 93.26  94.7
        score = score_s + score_t   #  94.7，95.8, 94.0
        # score = score_s + score_t + score_g   # 参数是1：92.3，93.9，0.2： 0.1：93.6，

        e = F.leaky_relu(score, negative_slope=self.negative_slope)

        if bias is not None:
            e = e + bias

        # 6) 注意力权重
        alpha = torch.softmax(e, dim=-1)   # [N,T,V,V]

        if self.coef_drop > 0:
            alpha = F.dropout(alpha, p=self.coef_drop, training=self.training)

        # 7) 聚合输出
        y = torch.matmul(alpha, value)     # [N,T,V,F]

        # residual
        if self.residual:    # true的时候94.7，false的时候 88.6
            if self.res_conv is not None:
                res = self.res_conv(x).permute(0, 2, 3, 1).contiguous()   # [N,T,V,F]
            else:
                res = x_ntvc
            y = y + res

        if self.activation is not None:
            y = self.activation(y)

        y = y.permute(0, 3, 1, 2).contiguous()   # [N,F,T,V]
        return y, alpha
'''
class AttnHeadGATMotionEdge(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        edge_dim,
        activation=F.elu,
        in_drop=0.0,
        coef_drop=0.0,
        residual=True,
        negative_slope=0.2,
    ):
        super().__init__()

        # 先验强度控制因子 λ（可学习）
        self.lambda_prior = nn.Parameter(torch.tensor(1.0))
        self.epsilon = 1e-6

        # 节点投影
        self.W_x = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)

        # 边特征映射
        self.W_g = nn.Linear(edge_dim, out_features, bias=False)

        # 注意力向量
        self.u_s = nn.Parameter(torch.empty(out_features))
        self.u_t = nn.Parameter(torch.empty(out_features))
        self.u_g = nn.Parameter(torch.empty(out_features))

        # 运动偏置 beta_j(t) = MLP(m_j(t))
        self.motion_mlp = nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, 1)
        )

        self.activation = activation
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        self.negative_slope = negative_slope

        if residual and in_features != out_features:
            self.res_conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        else:
            self.res_conv = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_g.weight)

        nn.init.xavier_uniform_(self.u_s.unsqueeze(0))
        nn.init.xavier_uniform_(self.u_t.unsqueeze(0))
        nn.init.xavier_uniform_(self.u_g.unsqueeze(0))

        for m in self.motion_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.res_conv is not None:
            nn.init.xavier_uniform_(self.res_conv.weight)

    def forward(self, x, edge_feat, bias=None):
        """
        x:         [N, C, T, V]
        edge_feat: [N, T, V, V, G]
        bias:      [N, T, V, V] or None
        """
        N, C, T, V = x.shape
        x_ntvc = x.permute(0, 2, 3, 1).contiguous()   # [N,T,V,C]

        if self.in_drop > 0:
            x_ntvc = F.dropout(x_ntvc, p=self.in_drop, training=self.training)

        # 1) 节点投影
        h = self.W_x(x_ntvc)       # [N,T,V,F]
        value = self.W_v(x_ntvc)   # [N,T,V,F]

        # 2) 运动特征
        v = torch.zeros_like(x_ntvc)   # [N,T,V,C]
        v[:, 1:] = x_ntvc[:, 1:] - x_ntvc[:, :-1]

        a = torch.zeros_like(x_ntvc)   # [N,T,V,C]
        a[:, 1:] = v[:, 1:] - v[:, :-1]

        m = torch.cat([v, a], dim=-1)  # [N,T,V,2C]

        # 3) 运动偏置 beta_j(t)
        beta = self.motion_mlp(m).squeeze(-1)   # [N,T,V]
        beta_j = beta.unsqueeze(2)              # [N,T,1,V]
        
        # 这段if代码是临时加的
        if edge_feat.dim() == 3 and edge_feat.size(0) == 1:
            edge_feat = edge_feat.squeeze(0)   # [20,20]
            edge_feat = edge_feat.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1,1,20,20,1]
            edge_feat = edge_feat.expand(N, T, V, V, 1)

        # 4) 边特征映射
        g_hat = self.W_g(edge_feat)   # [N,T,V,V,F]

        # 5) 分项注意力打分
        score_s = torch.einsum('ntif,f->nti', h, self.u_s).unsqueeze(-1)   # [N,T,V,1]
        score_t = torch.einsum('ntjf,f->ntj', h, self.u_t).unsqueeze(-2)   # [N,T,1,V]
        score_g = torch.einsum('ntijf,f->ntij', g_hat, self.u_g)           # [N,T,V,V]

        # score = score_s + score_t + score_g + beta_j                       # [N,T,V,V]
        # score = score_s + score_t + beta_j    # 92.3
        # score = score_s   # 93.26  94.7
        score = score_s + score_t   #  94.7，95.8, 94.0
        # score = score_s + score_t + score_g   # 参数是1：92.3，93.9，0.2： 0.1：93.6，

        e = F.leaky_relu(score, negative_slope=self.negative_slope)

        if bias is not None:
            e = e + bias

        # 6) 注意力权重
        # alpha = torch.softmax(e, dim=-1)   # [N,T,V,V]   这个是标准的softmanx公式

        # 6) 注意力权重 — 融合相似度先验 A^s
        # 将 edge_feat 降到 [N, T, V, V] 作为相似度先验
        if edge_feat.dim() == 5:
            A_s = edge_feat.mean(dim=-1)      # [N,T,V,V,G] -> [N,T,V,V]
        elif edge_feat.dim() == 4:
            A_s = edge_feat                    # [N,T,V,V]
        elif edge_feat.dim() == 2:
            A_s = edge_feat.unsqueeze(0).unsqueeze(0).expand(N, T, V, V)
        else:
            A_s = edge_feat

        # α = softmax( z + λ·log(A^s + ε) )
        log_prior = self.lambda_prior * torch.log(A_s + self.epsilon)   # [N,T,V,V]
        alpha = torch.softmax(e + log_prior, dim=-1)                    # [N,T,V,V]

        if self.coef_drop > 0:
            alpha = F.dropout(alpha, p=self.coef_drop, training=self.training)

        # 7) 聚合输出
        y = torch.matmul(alpha, value)     # [N,T,V,F]

        # residual
        if self.residual:    # true的时候94.7，false的时候 88.6
            if self.res_conv is not None:
                res = self.res_conv(x).permute(0, 2, 3, 1).contiguous()   # [N,T,V,F]
            else:
                res = x_ntvc
            y = y + res

        if self.activation is not None:
            y = self.activation(y)

        y = y.permute(0, 3, 1, 2).contiguous()   # [N,F,T,V]
        return y, alpha
    

class CrossAttributeTransform(nn.Module):
    """
    将pose的欧拉角(yaw, pitch, roll)转换为与landmarks一致的3D坐标
    使用注意力机制，让pose点关注所有landmark点来生成其3D表示
    """
    
    def __init__(self, v_lmk=19, v_pose=1, d_k=16, method='attention'):
        """
        Args:
            v_lmk: landmark节点数量 (19)
            v_beta: pose节点数量 (1)
            d_k: 注意力机制的key/query维度
            method: 转换方法 ['attention', 'mlp', 'geometric']
        """
        super().__init__()
        self.v_lmk = v_lmk
        self.v_pose = v_pose
        self.d_k = d_k
        self.method = method
        
        c_in = 3  # 输入通道数 (x, y, z) 或 (yaw, pitch, roll)
        c_out = 3  # 输出通道数 (x, y, z)
        
        if method == 'attention':
            # 注意力机制的参数
            self.W_q = nn.Linear(c_in * v_lmk, d_k, bias=False)  # Query from landmarks
            self.W_k = nn.Linear(c_in * v_pose, d_k, bias=False)   # Key from pose
            self.W_v = nn.Linear(c_in * v_pose, c_out, bias=False)  # Value from pose
            
            # 额外的pose编码器
            self.pose_encoder = nn.Sequential(
                nn.Linear(c_in, 32),
                nn.ReLU(),
                nn.Linear(32, c_out)
            )
            
            # 融合权重
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
            
        elif method == 'mlp':
            # MLP方法：直接用MLP转换pose
            self.pose_mlp = nn.Sequential(
                nn.Linear(c_in, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, c_out)
            )
            
        elif method == 'geometric':
            # 几何方法：欧拉角转3D方向向量
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.zeros(3))
            
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (N, C, T, V, M) where C=3, V=20 (19 landmarks + 1 pose)
        Returns:
            x_out: (N, C, T, V, M) where the 20th node is converted to 3D coordinates
        """
        N, C, T, V, M = x.size()
        
        # 分离landmarks和pose
        landmarks = x[:, :, :, :self.v_lmk, :]  # (N, 3, T, 19, M)
        pose = x[:, :, :, self.v_lmk:, :]       # (N, 3, T, 1, M)
        
        if self.method == 'attention':
            pose_3d = self._attention_transform(landmarks, pose)
        elif self.method == 'mlp':
            pose_3d = self._mlp_transform(pose)
        elif self.method == 'geometric':
            pose_3d = self._geometric_transform(pose, landmarks)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # 拼接回去
        x_out = torch.cat([landmarks, pose_3d], dim=3)  # (N, 3, T, 20, M)
        
        return x_out
    
    def _attention_transform(self, landmarks, pose):
        """
        使用注意力机制：pose作为query，landmarks作为key/value
        生成与landmarks空间一致的3D坐标
        """
        N, C, T, V_alpha, M = landmarks.size()
        
        # 重排维度便于处理: (N, C, T, V, M) -> (N*T*M, V, C) -> (N*T*M, V*C)
        landmarks_flat = landmarks.permute(0, 2, 4, 3, 1).contiguous()  # (N, T, M, V, C)
        landmarks_flat = landmarks_flat.view(N * T * M, V_alpha * C)    # (N*T*M, V*C)
        
        pose_flat = pose.permute(0, 2, 4, 3, 1).contiguous()  # (N, T, M, 1, C)
        pose_flat = pose_flat.view(N * T * M, C)               # (N*T*M, C)
        
        # 计算注意力
        Q = self.W_q(landmarks_flat)  # (N*T*M, d_k)
        K = self.W_k(pose_flat)       # (N*T*M, d_k)
        
        # 注意力分数 (这里是单个pose对所有landmarks的attention)
        attn_scores = torch.sum(Q * K.unsqueeze(1).expand(-1, V_alpha, -1), dim=-1)  # (N*T*M, V_alpha)
        attn_scores = attn_scores / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (N*T*M, V_alpha)
        
        # 加权landmarks得到pose的3D表示
        landmarks_for_attn = landmarks.permute(0, 2, 4, 3, 1).contiguous()  # (N, T, M, V, C)
        landmarks_for_attn = landmarks_for_attn.view(N * T * M, V_alpha, C)  # (N*T*M, V, C)
        
        pose_from_landmarks = torch.einsum('nv,nvc->nc', attn_weights, landmarks_for_attn)  # (N*T*M, C)
        
        # 同时用MLP编码原始pose
        pose_encoded = self.pose_encoder(pose_flat)  # (N*T*M, C)
        
        # 融合两种表示
        alpha = torch.sigmoid(self.fusion_weight)
        pose_3d = alpha * pose_from_landmarks + (1 - alpha) * pose_encoded  # (N*T*M, C)
        
        # 还原维度: (N*T*M, C) -> (N, T, M, 1, C) -> (N, C, T, 1, M)
        pose_3d = pose_3d.view(N, T, M, 1, C)
        pose_3d = pose_3d.permute(0, 4, 1, 3, 2).contiguous()  # (N, C, T, 1, M)
        
        return pose_3d
    
    def _mlp_transform(self, pose):
        """
        简单MLP转换：直接将欧拉角转换为3D坐标
        """
        N, C, T, V_beta, M = pose.size()
        
        # (N, C, T, 1, M) -> (N*T*M, C)
        pose_flat = pose.permute(0, 2, 4, 3, 1).contiguous()
        pose_flat = pose_flat.view(N * T * M, C)
        
        # MLP转换
        pose_3d = self.pose_mlp(pose_flat)  # (N*T*M, C)
        
        # 还原维度
        pose_3d = pose_3d.view(N, T, M, 1, C)
        pose_3d = pose_3d.permute(0, 4, 1, 3, 2).contiguous()
        
        return pose_3d
    
    def _geometric_transform(self, pose, landmarks):
        """
        几何转换：将欧拉角转换为方向向量，然后加到landmarks质心上
        """
        N, C, T, V_beta, M = pose.size()
        
        # 提取欧拉角: (N, 3, T, 1, M)
        yaw = pose[:, 0, :, 0, :]    # (N, T, M)
        pitch = pose[:, 1, :, 0, :]  # (N, T, M)
        roll = pose[:, 2, :, 0, :]   # (N, T, M)
        
        # 计算landmarks质心
        centroid = landmarks.mean(dim=3, keepdim=True)  # (N, 3, T, 1, M)
        
        # 欧拉角转方向向量
        # 这里假设yaw, pitch, roll已经是弧度制
        # 如果是角度制，需要先转换
        x = torch.cos(pitch) * torch.sin(yaw)
        y = -torch.sin(pitch)
        z = torch.cos(pitch) * torch.cos(yaw)
        
        # roll影响（绕前方向旋转）
        x = x + 0.1 * torch.sin(roll)
        y = y + 0.1 * (1 - torch.cos(roll))
        
        direction = torch.stack([x, y, z], dim=1)  # (N, 3, T, M)
        direction = direction.unsqueeze(3)  # (N, 3, T, 1, M)
        
        # 缩放和偏移
        scale = self.scale.abs()
        pose_3d = centroid + scale * direction + self.bias.view(1, 3, 1, 1, 1)
        
        return pose_3d


class CrossAttributeTransformSimple(nn.Module):
    """
    简化版：直接用可学习的线性变换将pose(yaw,pitch,roll)转换为3D坐标(x,y,z)
    """
    
    def __init__(self, v_alpha=19):
        super().__init__()
        self.v_alpha = v_alpha
        
        # 简单的线性变换 + 非线性激活
        self.transform = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.transform[0].weight)
            if self.transform[0].bias is not None:
                nn.init.zeros_(self.transform[0].bias)
            nn.init.xavier_uniform_(self.transform[3].weight)
            if self.transform[3].bias is not None:
                nn.init.zeros_(self.transform[3].bias)
    
    def forward(self, x):
        """
        Args:
            x: (N, C, T, V, M) where C=3, V=20 (19 landmarks + 1 pose)
        Returns:
            x_out: (N, C, T, V, M) with pose converted to 3D coordinates
        """
        N, C, T, V, M = x.size()
        
        # 分离landmarks和pose
        landmarks = x[:, :, :, :self.v_alpha, :]  # (N, 3, T, 19, M)
        pose = x[:, :, :, self.v_alpha:, :]       # (N, 3, T, 1, M)
        
        # 方法：不用squeeze，直接permute后reshape
        # (N, 3, T, 1, M) -> (N, T, 1, M, 3) -> (N*T*M, 3)
        pose_permuted = pose.permute(0, 2, 3, 4, 1).contiguous()  # (N, T, 1, M, 3)
        pose_flat = pose_permuted.view(N * T * M, C)  # (N*T*M, 3)
        
        # 应用变换
        pose_3d = self.transform(pose_flat)  # (N*T*M, 3)
        
        # 还原维度: (N*T*M, 3) -> (N, T, 1, M, 3) -> (N, 3, T, 1, M)
        pose_3d = pose_3d.view(N, T, 1, M, C)  # (N, T, 1, M, 3)
        pose_3d = pose_3d.permute(0, 4, 1, 2, 3).contiguous()  # (N, 3, T, 1, M)
        
        # 拼接
        x_out = torch.cat([landmarks, pose_3d], dim=3)  # (N, 3, T, 20, M)
        
        return x_out
    
# 标准的transform公式
class CrossAttributeTransform1(nn.Module):
    """
    将 pose 的欧拉角 (yaw, pitch, roll) 转换为与 landmarks 一致的 3D 坐标
    输入数据格式:
        x: (N, C, T, V, M)
    其中:
        C = 3
        V = 20 = 19 landmarks + 1 pose
    输出:
        x_out: (N, C, T, V, M)
    其中第 20 个节点被替换为转换后的 3D pose 表示
    """

    def __init__(self, v_lmk=19, v_pose=1, d_k=16):
        super().__init__()
        self.v_lmk = v_lmk
        self.v_pose = v_pose
        self.d_k = d_k

        c_in = 3   # landmark xyz 或 pose(yaw,pitch,roll)
        c_out = 3  # 输出转换后的 xyz

        # pose -> query
        self.W_q = nn.Linear(c_in, d_k, bias=False)

        # landmark -> key
        self.W_k = nn.Linear(c_in, d_k, bias=False)
        self.W_v = nn.Linear(c_in, c_out, bias=False)

        # 原始 pose 编码器
        self.pose_encoder = nn.Sequential(
            nn.Linear(c_in, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, c_out)
        )

        # 融合系数
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V, M), V = 20
               前19个节点是 landmarks，最后1个节点是 pose
        Returns:
            x_out: (N, C, T, V, M)
        """
        N, C, T, V, M = x.size()

        # 这两行是在做输入合法性检查
        assert C == 3, f"Expected C=3, but got C={C}"
        assert V == self.v_lmk + self.v_pose, \
            f"Expected V={self.v_lmk + self.v_pose}, but got V={V}"

        # 前19个 landmarks, 后1个 pose
        landmarks = x[:, :, :, :self.v_lmk, :]   # (N, 3, T, 19, M)
        pose = x[:, :, :, self.v_lmk:, :]        # (N, 3, T, 1, M)

        pose_3d = self._attention_transform(landmarks, pose)

        # 拼回原图
        x_out = torch.cat([landmarks, pose_3d], dim=3)  # (N, 3, T, 20, M)
        return x_out

    def _attention_transform(self, landmarks, pose):
        """
        pose 作为 Query, landmarks 作为 Key/Value
        输出转换后的 pose_3d
        Args:
            landmarks: (N, 3, T, 19, M)
            pose:      (N, 3, T, 1, M)
        Returns:
            pose_3d:   (N, 3, T, 1, M)
        """
        N, C, T, V_lmk, M = landmarks.size()

        # landmarks: (N, 3, T, 19, M) -> (N, T, M, 19, 3) -> (N*T*M, 19, 3)
        landmarks_tok = landmarks.permute(0, 2, 4, 3, 1).contiguous()
        landmarks_tok = landmarks_tok.view(N * T * M, V_lmk, C)

        # pose: (N, 3, T, 1, M) -> (N, T, M, 1, 3) -> (N*T*M, 1, 3)
        pose_tok = pose.permute(0, 2, 4, 3, 1).contiguous()
        pose_tok = pose_tok.view(N * T * M, 1, C)

        # Q from pose: (NTM, 1, d_k)
        Q = self.W_q(pose_tok)

        # K from landmarks: (NTM, 19, d_k)
        K = self.W_k(landmarks_tok)

        # attention scores: (NTM, 1, 19)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 对原始 landmarks xyz 加权求和，得到几何空间中的 pose 表示
        # landmarks_tok: (NTM, 19, 3)
        # attn_weights:  (NTM, 1, 19)
        # pose_from_landmarks: (NTM, 1, 3) -> (NTM, 3)
        # V = self.W_v(landmarks_tok)   # (NTM, 19, 3)
        V = landmarks_tok   # 第二种实现方式：直接使用attn_weights对原始landmarks加权求和，不经过线性变换
        pose_from_landmarks = torch.matmul(attn_weights, V).squeeze(1)

        # 原始 pose 编码分支
        pose_flat = pose_tok.squeeze(1)   # (NTM, 3)
        pose_encoded = self.pose_encoder(pose_flat)  # (NTM, 3)

        # 融合
        alpha = torch.sigmoid(self.fusion_weight)
        pose_3d = alpha * pose_from_landmarks + (1.0 - alpha) * pose_encoded  # (NTM, 3)

        # 还原回 (N, 3, T, 1, M)
        pose_3d = pose_3d.view(N, T, M, 1, C)
        pose_3d = pose_3d.permute(0, 4, 1, 3, 2).contiguous()

        return pose_3d

# 加了偏置和mask
class CrossAttributeTransform2(nn.Module):
    """
    将 pose 的欧拉角 (yaw, pitch, roll) 转换为与 landmarks 一致的 3D 坐标
    输入:  x: (N, C, T, V, M)  其中:
            C = 3
            V = 20 = 19 landmarks + 1 pose
    输出:
        x_out: (N, C, T, V, M)
        其中第 20 个节点被替换为转换后的 3D pose 表示
    """

    def __init__(
        self,
        v_lmk=19,
        v_pose=1,
        d_k=16,
        full_adj_matrix=None,   # 原始邻接矩阵, shape 可为 (1,20,20) 或 (20,20)
        use_mask=False,         # True: 硬掩码, False: 软偏置
        use_value_proj=False    # 是否对 V 使用线性映射
    ):
        super().__init__()
        self.v_lmk = v_lmk
        self.v_pose = v_pose
        self.d_k = d_k
        self.use_mask = use_mask
        self.use_value_proj = use_value_proj

        c_in = 3
        c_out = 3

        # pose -> query
        self.W_q = nn.Linear(c_in, d_k, bias=False)

        # landmark -> key/value
        self.W_k = nn.Linear(c_in, d_k, bias=False)
        self.W_v = nn.Linear(c_in, c_out, bias=False)

        # 原始 pose 编码器
        self.pose_encoder = nn.Sequential(
            nn.Linear(c_in, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, c_out)
        )

        # 融合系数 alpha = sigmoid(fusion_weight)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # 图偏置强度 beta
        self.beta = nn.Parameter(torch.tensor(1.0))

        # 从 full_adj_matrix 中提取 pose->landmarks 的关系矩阵 G
        G = self._build_pose_to_landmark_bias(full_adj_matrix)
        self.register_buffer("adj_bias", G)   # shape: (1, 19)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_pose_to_landmark_bias(self, full_adj_matrix):
        """
        从原始 full_adj_matrix 提取 pose -> landmarks 的关系矩阵
        支持输入:
            (1, 20, 20)
            (20, 20)
        输出:  G: (1, 19)
        """
        if full_adj_matrix is None:
            # 默认无偏置
            return torch.zeros(self.v_pose, self.v_lmk, dtype=torch.float32)

        A = torch.as_tensor(full_adj_matrix, dtype=torch.float32)

        if A.dim() == 3:
            # (1, 20, 20)
            assert A.size(0) == 1, f"Expected first dim = 1, but got {A.size(0)}"
            A = A[0]
        elif A.dim() == 2:
            # (20, 20)
            pass
        else:
            raise ValueError(f"full_adj_matrix must have shape (1,20,20) or (20,20), but got {A.shape}")

        expected_V = self.v_lmk + self.v_pose
        assert A.shape == (expected_V, expected_V), \
            f"Expected full_adj_matrix shape = {(expected_V, expected_V)}, but got {A.shape}"

        # 提取 pose -> landmarks
        # 前19个是 landmarks, 最后1个是 pose
        G = A[self.v_lmk:self.v_lmk + self.v_pose, :self.v_lmk]  # (1, 19)

        return G

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (N, C, T, V, M), V = 20
            return_attention: 是否返回注意力相关中间结果

        Returns:
            x_out: (N, C, T, V, M)
            若 return_attention=True, 额外返回一个 dict
        """
        N, C, T, V, M = x.size()

        assert C == 3, f"Expected C=3, but got C={C}"
        assert V == self.v_lmk + self.v_pose, \
            f"Expected V={self.v_lmk + self.v_pose}, but got V={V}"

        landmarks = x[:, :, :, :self.v_lmk, :]   # (N, 3, T, 19, M)
        pose = x[:, :, :, self.v_lmk:, :]        # (N, 3, T, 1, M)

        pose_3d, attn_dict = self._attention_transform(
            landmarks, pose, return_attention=return_attention
        )

        x_out = torch.cat([landmarks, pose_3d], dim=3)  # (N, 3, T, 20, M)

        if return_attention:
            return x_out, attn_dict
        return x_out

    def _attention_transform(self, landmarks, pose, return_attention=False):
        """
        pose 作为 Query, landmarks 作为 Key/Value
        输出转换后的 pose_3d
        Args:
            landmarks: (N, 3, T, 19, M)
            pose:      (N, 3, T, 1, M)
        Returns:
            pose_3d:   (N, 3, T, 1, M)
        """
        N, C, T, V_lmk, M = landmarks.size()

        # landmarks: (N, 3, T, 19, M) -> (N, T, M, 19, 3) -> (NTM, 19, 3)
        landmarks_tok = landmarks.permute(0, 2, 4, 3, 1).contiguous()
        landmarks_tok = landmarks_tok.view(N * T * M, V_lmk, C)

        # pose: (N, 3, T, 1, M) -> (N, T, M, 1, 3) -> (NTM, 1, 3)
        pose_tok = pose.permute(0, 2, 4, 3, 1).contiguous()
        pose_tok = pose_tok.view(N * T * M, self.v_pose, C)

        # Q, K
        Q = self.W_q(pose_tok)          # (NTM, 1, d_k)
        K = self.W_k(landmarks_tok)     # (NTM, 19, d_k)

        # 内容相似度项
        content_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)  # (NTM, 1, 19)

        # 图偏置项
        # adj_bias: (1, 19) -> (1, 1, 19)
        G = self.adj_bias.unsqueeze(0)  # (1, 1, 19)

        if self.use_mask:
            # 硬掩码方式
            # G 非0的位置保留，为0的位置屏蔽
            attn_scores = content_scores.masked_fill(G == 0, float('-inf'))
            relation_bias = None
        else:
            # 软偏置方式
            relation_bias = self.beta * G                    # (1, 1, 19)
            attn_scores = content_scores + relation_bias     # broadcast -> (NTM, 1, 19)

        attn_weights = F.softmax(attn_scores, dim=-1)        # (NTM, 1, 19)
 
        if self.use_value_proj:
            V = self.W_v(landmarks_tok)   # (NTM, 19, 3)
        else:
            V = landmarks_tok             # (NTM, 19, 3)

        # 加权求和
        pose_from_landmarks = torch.matmul(attn_weights, V).squeeze(1)  # (NTM, 3)

        # pose 自身编码分支
        pose_flat = pose_tok.squeeze(1)               # (NTM, 3)
        pose_encoded = self.pose_encoder(pose_flat)   # (NTM, 3)

        # 融合
        alpha = torch.sigmoid(self.fusion_weight)
        pose_3d = alpha * pose_from_landmarks + (1.0 - alpha) * pose_encoded  # (NTM, 3)

        # 恢复形状: (NTM, 3) -> (N, 3, T, 1, M)
        pose_3d = pose_3d.view(N, T, M, self.v_pose, C)
        pose_3d = pose_3d.permute(0, 4, 1, 3, 2).contiguous()

        if return_attention:
            attn_dict = {
                "content_scores": content_scores,   # (NTM, 1, 19)
                "attn_scores": attn_scores,         # (NTM, 1, 19)
                "attn_weights": attn_weights,       # (NTM, 1, 19)
                "adj_bias": G,                      # (1, 1, 19)
                "beta": self.beta.detach(),
                "alpha": alpha.detach()
            }
            if relation_bias is not None:
                attn_dict["relation_bias"] = relation_bias
        else:
            attn_dict = None

        return pose_3d, attn_dict