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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HiGCN_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Order=2, bias=True, **kwargs):
        super(HiGCN_prop, self).__init__(aggr='add', **kwargs)
        # K=10, alpha=0.1, Init='PPR',
        self.K = K
        self.alpha = alpha
        self.Order = Order
        self.fW = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, HL):

        HL = HL.to(device)

        hidden = x * (self.fW[0])
        for k in range(self.K):
            x = matmul(HL, x)
            # x = torch.matmul(HL, x)
            gamm = self.fW[k + 1]
            hidden = hidden + gamm * x

        return hidden

    # 自定义打印结构
    def __repr__(self):
        return '{}(Order={}, K={}, filterWeights={})'.format(self.__class__.__name__, self.Order, self.K, self.fW)


class HiGCN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(HiGCN, self).__init__()
        self.Order = kwargs['Order']
        node_feature_shape = kwargs['node_feature_shape']
        self.num_hid = kwargs['num_hid']
        self.K = kwargs['K']
        self.alpha = kwargs['alpha']
        self.dprate = kwargs['dprate']
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList()



        for i in range(self.Order):
            self.lin_in.append(Linear(node_feature_shape, self.num_hid))
            self.hgc.append(HiGCN_prop(self.K, self.alpha, self.Order))

        self.lin_out1 = Linear(self.num_hid * self.Order, int((self.num_hid * self.Order)/2))
        self.lin_out2 = Linear(int((self.num_hid * self.Order)/2), 1)

        self.fc = nn.Sequential(nn.Linear(self.num_hid, int((self.num_hid)/2)), nn.ReLU(),
                                nn.Linear(int((self.num_hid)/2), 1))  # 权重的卷积维度
        self.fc1 = nn.Linear(self.num_hid, node_feature_shape)


        # self.Hyper_lin1 = Linear(8, 32, bias=True)
        # self.HyperConv1 = HGNN_demo(in_ch=32, n_class=1, n_hid=64)


        # self.Init = args.Init

        # self.a = fuse

    # def reset_parameters(self):
    #     self.hgc.reset_parameters()

    def forward(self, node_feature, HL, edge, faces, G_single):

        # node_feature, A_adj = node_feature.to(next(self.parameters()).device), [adj.to(next(self.parameters()).device) for adj in A_adj]
        node_feature = node_feature.float()
        # A_adj = A_adj.to_sparse()


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
