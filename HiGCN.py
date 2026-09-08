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
from torch_sparse import matmul, SparseTensor
import math
from torch.nn.modules.module import Module
from dgl.nn.pytorch.softmax import edge_softmax
import dgl
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn.conv import hypergraph_conv

from incidence_matrix import get_simplex_idx
from utils import get_node_HL, get_simplex_HL1_fast, get_node_HL_fast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HiGCN_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Order=2, bias=True, **kwargs):
        super(HiGCN_prop, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        self.alpha = alpha
        self.Order = Order
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
        return '{}(Order={}, K={}, filterWeights={})'.format(self.__class__.__name__, self.Order, self.K, self.temp)


class HiGCN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(HiGCN, self).__init__()
        self.Order = kwargs['Order']
        node_feature_shape = kwargs['node_feature_shape']
        self.num_hid = kwargs['num_hid']
        self.K = kwargs['K']
        self.alpha = kwargs['alpha']
        self.dprate = kwargs['dprate']
        self.Init = kwargs['Init']
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()


        for i in range(self.Order):
            self.lin_in.append(Linear(node_feature_shape, self.num_hid))
            self.hgc.append(HiGCN_prop(self.K, self.alpha, self.Init, self.Order))

        self.lin_out1 = Linear(self.num_hid * self.Order, int((self.num_hid * self.Order)/2))
        self.lin_out2 = Linear(int((self.num_hid * self.Order)/2), 1)

        self.fc = nn.Sequential(nn.Linear(self.num_hid, int((self.num_hid)/2)), nn.ReLU(),
                                nn.Linear(int((self.num_hid)/2), 1))  # 权重的卷积维度
        self.fc1 = nn.Linear(self.num_hid, node_feature_shape)

        self.vc_edge = VertexConv(node_feature_shape, 2)  # 边卷积，生成边特征
        self.vc_face = VertexConv(node_feature_shape, 3)  # 面卷积，生成面特征

        self.edge_conv = nn.Sequential(nn.Linear(node_feature_shape, node_feature_shape//2), nn.ReLU(),
                                nn.Linear(node_feature_shape//2, 1))  # 权重的卷积维度


        # self.Hyper_lin1 = Linear(8, 32, bias=True)
        # self.HyperConv1 = HGNN_demo(in_ch=32, n_class=1, n_hid=64)


        # self.Init = args.Init

        # self.a = fuse

    # def reset_parameters(self):
    #     self.hgc.reset_parameters()

    def get_simplex_ft(self, node_feature, edge, faces):

        edge_matrix = torch.tensor(np.array([[key[0], key[1]] for key in edge.keys()]), dtype=torch.long).to(device)  # 获得边的矩阵
        faces_matrix = torch.tensor(np.array(faces), dtype=torch.long).to(device)  # 获得三角形的矩阵

        edge_num = len(edge_matrix)
        faces_num = len(faces_matrix)
        d = node_feature.shape[1]

        edge_node_feature = node_feature[edge_matrix.view(-1)].view(edge_num, 2, d)
        faces_node_feature = node_feature[faces_matrix.view(-1)].view(faces_num, 3, d)

        return edge_node_feature, faces_node_feature

    def get_HL(self, G_single, train_adj_matrices, edge, faces, edge_weights, face_weights):
        nodes = G_single.nodes()

        node_H = get_node_HL_fast(train_adj_matrices, len(nodes))  # 获得0-simplex的稀疏矩阵
        edge_H = get_simplex_HL1_fast(edge_weights, nodes, edge)
        faces_H = get_simplex_HL1_fast(face_weights, nodes, faces)

        HL = {
            1: node_H,
            2: edge_H,
            3: faces_H
        }
        return HL

    def forward(self, node_feature, G_single, train_adj_matrices, edge, faces, edge_node_ft, faces_node_ft):
        node_feature = node_feature.to(device).float()
        edge_node_ft = edge_node_ft.to(device).float()
        faces_node_ft = faces_node_ft.to(device).float()

        edge_ft = self.vc_edge(edge_node_ft)  # 这一步是节点卷积，将节点特征卷到超边上 (R, N, d) -> (R, d),获得边特征
        face_ft = self.vc_face(faces_node_ft)  # 获得面特征

        # edge_weights = F.softmax(edge_ft.mean(dim=1), dim=0).view(-1, 1)  # 每条边的权重，这种计算权重的方式，权重太小
        # face_weights = F.softmax(face_ft.mean(dim=1), dim=0).view(-1, 1)  # 每个面的权重
        edge_weights = self.edge_conv(edge_ft)
        face_weights = self.edge_conv(face_ft)
        # edge_weights = (edge_ft.mean(dim=1) / torch.max(edge_ft.mean(dim=1))).view(-1, 1)
        # face_weights = (face_ft.mean(dim=1) / torch.max(face_ft.mean(dim=1))).view(-1, 1)

        HL = self.get_HL(G_single, train_adj_matrices, edge, faces, edge_weights, face_weights)

        x_concat = torch.tensor([]).to(device)
        for i in range(self.Order):
            xx = F.dropout(node_feature, p=self.dprate, training=self.training)
            xx = self.lin_in[i](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[i](xx, HL[i + 1])

            xx_weight = xx.view(len(xx), 1, xx.size(1))  # 计算权重需要加上这个，不计算删除

            x_concat = torch.cat((x_concat, xx_weight), dim=1)  # 不计算权重将xx_weight换为xx

        scores = []
        n_edges = x_concat.size(1)
        for i in range(n_edges):
            scores.append(self.fc(x_concat[:, i]))
        scores1 = torch.softmax(torch.stack(scores, 1), 1)

        scores2 = (scores1 * x_concat).sum(1)
        scores2 = F.dropout(scores2, p=self.dprate, training=self.training)
        scores3 = self.fc1(scores2)

        ###x_concat = F.dropout(x_concat, p=0.5, training=self.training)
        ###x_concat1 = self.lin_out1(x_concat)
        ###x_concat2 = self.lin_out2(x_concat1)
        ###x_concat3 = F.leaky_relu(x_concat2)

        #Hyper_lin = self.Hyper_lin1(node_feature)
        #hyper_score = self.HyperConv1(feature=Hyper_lin, G=KMeans_adj_norm)

        #DismanScore = torch.sigmoid(torch.add((1-self.a) * hyper_score, self.a * x_concat3))


        return scores3


class OptimizedTransform(nn.Module):
    """
    An optimized Vertex Transformation module using self-attention.
    """

    def __init__(self, dim_in, k):
        super().__init__()
        self.to_q = nn.Linear(dim_in, dim_in, bias=False)
        self.to_k = nn.Linear(dim_in, dim_in, bias=False)
        self.activation = nn.Softmax(dim=-1)
        self.scale = dim_in ** -0.5

    def forward(self, region_feats):
        """
        :param region_feats: (N, k, d)
        :return: (N, k, d)
        """
        # --- THIS IS THE FIX ---
        # Ensure the input tensor is float32, which matches the model's parameters.
        region_feats = region_feats.float()

        q = self.to_q(region_feats)
        k = self.to_k(region_feats)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        multiplier = self.activation(attn_scores)
        transformed_feats = torch.matmul(multiplier, region_feats)

        return transformed_feats


# Your VertexConv class remains the same, just make sure it uses the corrected OptimizedTransform above.

class VertexConv(nn.Module):
    """
    A Vertex Convolution layer
    (R,n,d) -> (R,d)
    """

    def __init__(self, dim_in, k):
        super().__init__()
        self.trans = OptimizedTransform(dim_in, k)
        self.convK1 = nn.Conv1d(k, 1, 1)

    def forward(self, region_feats):
        transformed_feats = self.trans(region_feats)
        pooled_feats = self.convK1(transformed_feats)
        pooled_feats = pooled_feats.squeeze(1)
        return pooled_feats
