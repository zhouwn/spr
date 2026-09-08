import math
import copy
import torch
import time
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from utils import cos_dis
from graphrole import RecursiveFeatureExtractor, RoleExtractor
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics.pairwise import cosine_similarity
from generate_role import extra_role, extra_indic, generate_role_node_list, extra_feature


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RoleLayer(torch.nn.Module):
    """
    A Dynamic Hypergraph Convolution Layer
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.n_role = kwargs['N_Role']  # 角色图中的数量
        self.feature_dim = kwargs['dim_in']  # 节点特征维度
        self.role_node_num = kwargs['role_node_num']  # 超图中的角色数量
        self.activation = kwargs['activation']
        self.dropout_rate = kwargs['dropout_rate']
        self.k_dim = kwargs['k_dim']  # knn计算时的值
        self.vc_s = VertexConv(self.feature_dim, self.role_node_num)  # 节点卷积，生成角色特征
        self.vc_n = VertexConv(self.feature_dim, self.k_dim)

        self.Hyper_R = True
        self.Hyper_N = False

        self.lin1 = nn.Linear(self.feature_dim, self.feature_dim//2, bias=True)  # 经过GCN/HGNN后节点维度为16，现在需要把他们转成8

    def _vertex_conv(self, func, x):
        return func(x)

    def _edge_conv(self, x):
        return self.ec(x)

    def hyper_Role(self, role_node_idx, node_feature):

        role_node_idx = pad_sequence([torch.tensor(sublist) for sublist in role_node_idx],
                                     batch_first=True, padding_value=-1).to(device)

        role_node_feature = torch.zeros(role_node_idx.size(0), role_node_idx.size(1), node_feature.size(1),
                                        device=device, dtype=torch.float64).to(device)

        # 将节点特征赋予给每个角色对应的节点，并处理编号为-1的节点特征表示为0
        for i in range(role_node_idx.size(0)):
            for j in range(role_node_idx.size(1)):
                node_index = role_node_idx[i][j].item()
                if node_index != -1:
                    role_node_feature[i][j] = node_feature[int(node_index)]
        role_node_idx = role_node_idx.to(torch.int64)

        return role_node_feature, role_node_idx

    def hyper_Role_N(self, G):
        feature_extractor = RecursiveFeatureExtractor(G)
        features = feature_extractor.extract_features()

        # n_roles = 10
        role_extractor = RoleExtractor(n_roles=self.n_role)
        role_extractor.extract_role_factors(features)

        node_roles_ft = role_extractor.role_percentage
        return node_roles_ft


    def _nearest_select(self, G, node_roles_ft, node_feature):
        """
        :param ids: indices selected during train/valid/test, torch.LongTensor
        """
        node_list = list(G.nodes())
        node_roles_ft = torch.from_numpy(node_roles_ft.to_numpy())
        similarity_matrix = torch.from_numpy(cosine_similarity(node_roles_ft))

        nearest_neighbors = []  # 初始化一个空的列表来存储最近邻居的索引
        for i, node in enumerate(node_list):
            similarities = similarity_matrix[i]  # 获取当前节点的相似度向量
            max_sim_indices = torch.where(similarities == 1)[0].tolist()  # 找出相似度为1的所有节点的索引
            np.random.seed(i)
            if len(max_sim_indices) > self.k_dim:
                selected_indices = np.random.choice(max_sim_indices, self.k_dim, replace=False)  # 如果相似度为1的节点数多于k，则从中随机选择k个
            else:
                _, selected_indices = torch.topk(similarities, self.k_dim, dim=0)  # 否则，直接选择相似度最高的k个节点

            nearest_neighbors.append(selected_indices.tolist())
        nearest_neighbors = torch.tensor(nearest_neighbors)  # 转换为张量

        N = len(nearest_neighbors)
        d = node_feature.shape[1]  # 这里特征是用rolx生成的特征，如需修改，换成 node_feature
        # idx.view(-1)的 shape 为 N*kn
        nearest_feature = node_feature[nearest_neighbors.view(-1)].view(N, self.k_dim, d)   # (N, kn, d) 节点N的kn个邻居节点的特征
        nearest_feature = (nearest_feature).to(device)
        return nearest_feature

    def forward(self, G, role_node_idx, node_feature, edge_list, node_role_norm_matrix):

        if self.Hyper_R == True:
            role_node_feature, role_node_index = self.hyper_Role(role_node_idx, node_feature)  # 这一步是生成角色-节点-特征 (R, N, d)

            rs = self._vertex_conv(self.vc_s, role_node_feature)  # 这一步是节点卷积，将节点特征卷到超边上 (R, N, d) -> (R, d)
            role_weights = F.softmax(rs.mean(dim=1), dim=0).view(-1, 1)  # 这里生成的是每个角色的权重 R*1
            expanded_role_weights = role_weights.unsqueeze(-1).expand(-1, -1, rs.shape[1])  #

            weighted_role_node_features = role_node_feature * expanded_role_weights  # 将角色节点特征矩阵与扩展的角色权重矩阵相乘

            final_node_features = torch.zeros(node_feature.shape[0], role_node_feature.shape[2])  # 创建一个维度为 (节点数, 维度) 的矩阵，用于存储每个节点的特征

            for role_idx, indices in enumerate(role_node_index):
                for idx, node_idx in enumerate(indices):
                    if node_idx != -1:  # 忽略填充值 -1
                        final_node_features[node_idx] = weighted_role_node_features[role_idx, idx]  # 将角色的节点特征赋值给最终的节点特征张量


        if self.Hyper_N == True:
            node_roles_ft = self.hyper_Role_N(G)
            n_feat = self._nearest_select(G, node_roles_ft, node_feature)  # (N, k, d)
            final_node_features = self._vertex_conv(self.vc_n, n_feat)

        final_node_features = final_node_features.to(device)

        return self.activation(self.lin1(F.dropout(final_node_features, p=self.dropout_rate)))


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


class GraphConvolution(nn.Module):
    """
    A GCN layer
    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        # dim_in,
        # dim_out,
        # dropout_rate=0.5,
        # activation
        """
        super().__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=True)
        self.dropout = nn.Dropout(p=kwargs['dropout_rate'])
        self.activation = kwargs['activation']

    def _region_aggregate(self, feats, edge_dict):
        N = feats.size()[0]
        pooled_feats = torch.stack([torch.mean(feats[edge_dict[i]], dim=0) for i in range(N)])  # 对于每个节点i计算其邻居节点特征的平均值，

        return pooled_feats

    def forward(self, G, role_node_idx, node_feature, edge_list, node_role_norm_matrix):
        """
        :param ids: compatible with `MultiClusterConvolution`
        :param feats:
        :param edge_dict:
        :return:
        """
        x = node_feature.float()  # (N, d)
        x = self.dropout(self.activation(self.fc(x)))  # (N, d')
        x = self._region_aggregate(x, edge_list)  # (N, d)
        return x

class HGNN_conv(nn.Module):
    """
    A HGNN layer
    """
    def __init__(self, **kwargs):
        super(HGNN_conv, self).__init__()

        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=True)
        self.dropout = nn.Dropout(p=kwargs['dropout_rate'])
        self.activation = kwargs['activation']


    def forward(self, G, role_node_idx, node_feature, edge_list, node_role_norm_matrix):
        x = node_feature.float()
        node_role_norm_matrix = node_role_norm_matrix.float().to(device)
        x = self.activation(self.fc(x))
        x = node_role_norm_matrix.matmul(x)
        x = self.dropout(x)
        return x
