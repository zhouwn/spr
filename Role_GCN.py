#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
#from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from torch_sparse import matmul, SparseTensor, spspmm, coalesce
from torch_geometric.utils import degree, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data
import math
from torch.nn.modules.module import Module
from dgl.nn.pytorch.softmax import edge_softmax
import dgl
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn.conv import hypergraph_conv

from incidence_matrix import get_simplex_idx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HiGCN_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, bias=True, **kwargs):
        super(HiGCN_prop, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        self.alpha = alpha
        self.Init = Init

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha]=1.0
        elif self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3/(self.K+1))
            torch.nn.init.uniform_(self.temp,-bound,bound)
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))

    def forward(self, x, HL):

        HL = HL.to(device)
        HL = HL.float()
        x = x.float()

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = matmul(HL, x)
            # x = torch.matmul(HL, x)
            gamm = self.temp[k + 1]
            hidden = hidden + gamm * x

        return hidden

    # 自定义打印结构
    def __repr__(self):
        return '{}(K={}, filterWeights={})'.format(self.__class__.__name__, self.K, self.temp)


class Role_GCN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Role_GCN, self).__init__()
        node_feature_shape = kwargs['node_feature_shape']
        role_feature_shape = kwargs['role_feature_shape']
        self.num_hid = kwargs['num_hid']
        self.K = kwargs['K']
        self.alpha = kwargs['alpha']
        self.dprate = kwargs['dprate']
        self.Init = kwargs['Init']


        self.lin_in = Linear(node_feature_shape, self.num_hid)
        self.hgc = HiGCN_prop(self.K, self.alpha, self.Init)
        self.fc1 = nn.Linear(self.num_hid, node_feature_shape)

        self.role_weight = nn.Sequential(nn.Linear(role_feature_shape, node_feature_shape//2), nn.ReLU(),
                                nn.Linear(node_feature_shape//2, 1))  # 权重的卷积维度


        # self.Hyper_lin1 = Linear(8, 32, bias=True)
        # self.HyperConv1 = HGNN_demo(in_ch=32, n_class=1, n_hid=64)


        # self.Init = args.Init

        # self.a = fuse

    # def reset_parameters(self):
    #     self.hgc.reset_parameters()

    def get_HL_fast(self, role_weights, node_role_matrix):
        """
        使用高效的稀疏矩阵乘法来动态计算 HL 矩阵。

        参数:
            role_weights: 形状为 [S_role, 1] 的角色权重张量。
            node_role_matrix_sparse: 形状为 [N, S_role] 的 torch.sparse_coo_tensor。
        """
        device = role_weights.device

        if not isinstance(node_role_matrix, torch.Tensor):
            node_role_matrix = torch.tensor(node_role_matrix, dtype=torch.float32)

            # 2. 将张量移动到正确的设备
        node_role_matrix = node_role_matrix.to(device)

        # 3. 关键修正：确保张量是稀疏格式
        # 如果它还不是一个稀疏张量，就地将其转换。
        if not node_role_matrix.is_sparse:
            node_role_matrix_sparse = node_role_matrix.to_sparse()
        else:
            node_role_matrix_sparse = node_role_matrix

        N, S_role = node_role_matrix_sparse.shape

        # 1. 创建 H_w = H * W
        # H 是 (N, S_role), W 是 (S_role, S_role) 的对角阵
        # 这等价于按列缩放 H 的值
        H = node_role_matrix_sparse.coalesce()
        H_indices = H.indices()  # shape: (2, num_non_zero)
        H_values = H.values()  # shape: (num_non_zero,)

        # H_indices[1] 包含每个非零元素对应的“角色”索引
        # 我们可以用它来从 role_weights 中提取权重
        col_indices = H_indices[1]
        weights_for_values = role_weights.flatten()[col_indices]

        # 缩放 H 的非零值
        H_w_values = H_values * weights_for_values

        # H_w 是一个形状为 (N, S_role) 的新稀疏矩阵
        H_w_indices = H_indices

        # 2. 计算 M = H_w * H^T
        # 这是一个稀疏-稀疏矩阵乘法 (SpSpMM)
        H_t_indices = torch.stack([H_indices[1], H_indices[0]], dim=0)
        H_t_values = H_values

        # 使用 torch_sparse 库中的 spspmm，它非常高效且支持反向传播
        M_indices, M_values = spspmm(H_w_indices, H_w_values, H_t_indices, H_t_values, N, S_role, N)

        # 3. 对 M 进行对称归一化 -> HL = D^(-1/2) * M * D^(-1/2)
        row, col = M_indices
        # 计算 D^(-1/2)
        deg = degree(col, N, dtype=M_values.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        # 应用归一化
        norm_values = deg_inv_sqrt[row] * M_values * deg_inv_sqrt[col]

        # 4. 创建最终的稀疏张量 HL
        # 使用 coalesce 来合并重复的边
        HL_indices, HL_values = coalesce(M_indices, norm_values, N, N)

        # 使用 PyTorch Geometric 的 SparseTensor 格式，因为它对 GNN 操作有优化


        # 为了与你现有的 HiGCN_prop 兼容，我们可能需要返回 torch_sparse.SparseTensor
        # 如果你的 HiGCN_prop 使用 PyG 的 MessagePassing, SparseTensor 是最佳选择

        HL = SparseTensor(row=HL_indices[0], col=HL_indices[1], value=HL_values, sparse_sizes=(N, N))

        return HL

    def forward(self, G_loop, node_feature, role_feature, node_role_matrix):
        node_feature = node_feature.to(device).float()
        role_feature = role_feature.to(device).float()

        role_weights1 = self.role_weight(role_feature)
        role_weights = F.softmax(((role_weights1 - role_weights1.mean()) / role_weights1.std()), dim=0)

        role_HL = self.get_HL_fast(role_weights, node_role_matrix)

        xx = F.dropout(node_feature, p=self.dprate, training=self.training)
        xx = self.lin_in(xx)
        xx = F.dropout(xx, p=self.dprate, training=self.training)
        xx = self.hgc(xx, role_HL)


        scores2 = F.dropout(xx, p=self.dprate, training=self.training)
        scores3 = self.fc1(scores2)

        ###x_concat = F.dropout(x_concat, p=0.5, training=self.training)

        #Hyper_lin = self.Hyper_lin1(node_feature)
        #hyper_score = self.HyperConv1(feature=Hyper_lin, G=KMeans_adj_norm)

        #DismanScore = torch.sigmoid(torch.add((1-self.a) * hyper_score, self.a * x_concat3))


        return scores3



class HGNN_demo(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN_demo, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv_demo(in_ch, n_hid)
        self.hgc2 = HGNN_conv_demo(n_hid, n_class)

    def forward(self, feature, G):
        feature = feature.float()
        G = G.float()

        feature = F.relu(self.hgc1(feature, G))
        feature = F.dropout(feature, self.dropout)
        feature = self.hgc2(feature, G)
        return feature


class HGNN_conv_demo(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_demo, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        self.weight = self.weight.float()
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        device = x.device
        G_tensor = torch.tensor(G, device=device)  # 将图结构信息转换为 PyTorch 张量并移动到正确的设备上

        # 检查数据类型并进行必要的转换
        if G_tensor.dtype != x.dtype:
            G_tensor = G_tensor.to(x.dtype)
        x = G_tensor.matmul(x)
        return x



class VertexConv(nn.Module):
    """
    A Vertex Convolution layer
    Transform (N, k, d) feature to (N, d) feature by transform matrix and 1-D convolution
    将输入 region_feats 中的多个区域（邻域）的特征合并成一个维度为 d 的输出特征。这种操作通常用于将多个邻域的信息整合成一个更简洁的表示，
    这一部分是论文中的节点卷积部分，即将超图中的节点聚合到超边上
    (R,n,d) -> (R,d)
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension 输入特征的维度
        :param k: k neighbors 考虑的邻居数量，也就是超边中的节点数量
        将输入的特征从 (N, k, d) 形状转换为 (N, d) 形状
        """
        super().__init__()

        self.trans = Transform(dim_in, k)              # (N, k, d) -> (N, k, d)
        self.convK1 = nn.Conv1d(k, 1, 1)              # (N, k, d) -> (N, 1, d) 最后一个1代表卷积核的大小，用于在邻域特征之间进行合并或变换

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, d)
        """
        transformed_feats = self.trans(region_feats)
        pooled_feats = self.convK1(transformed_feats)    # (N, 1, d)


        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats

class Transform(nn.Module):
    """
    A Vertex Transformation module
    Permutation invariant transformation: (N, k, d) -> (N, k, d)
    通过对邻域内的特征进行卷积和Softmax操作来学习每个邻域内特征的排列不变性，将输入特征重新排列为 (N, k, d) 形状
    计算超图中每个节点的权重，随后进行加权求和
    """
    def __init__(self, dim_in, k):
        """
        :param dim_in: input feature dimension，节点特征维度
        :param k: k neighbors，超边中节点数量
        """
        super().__init__()

        self.convKK = nn.Conv1d(k, k * k, dim_in, groups=k)  # dim_in输入特征的维度
        self.activation = nn.Softmax(dim=-1)
        self.dp = nn.Dropout()

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        region_feats = region_feats.float()
        N, k, _ = region_feats.size()  # (N, k, d)
        conved = self.convKK(region_feats)  # (N, k*k, 1) 每个节点的特征与其邻居节点的特征进行组合
        multiplier = conved.view(N, k, k)  # (N, k, k) 接下来需要对邻居节点之间的相似性进行操作
        multiplier = self.activation(multiplier)  # softmax along last dimension 生成相似性矩阵
        transformed_feats = torch.matmul(multiplier, region_feats)  # (N, k, d)
        return transformed_feats
