import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec, GCNConv, GATConv
from torch_geometric.data import Data
import random
from gensim.models import Word2Vec

# GCN模型定义
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 4)
        self.conv2 = GCNConv(4, num_classes)

    def forward(self, node_feature, edge_index):
        edge_index = edge_index.long()
        node_feature = node_feature.float()

        x = self.conv1(node_feature, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)

    def loss(self, score, adj, gamma: float = 1, mean: bool = True):
        tmp = 1 + torch.mul(score.unsqueeze(1), adj.cuda())
        tmp = torch.prod(tmp.pow(-1), dim=0)
        loss1 = tmp.mean() if mean else tmp.sum()
        loss2 = score.mean() if mean else score.sum()

        return loss1 + gamma * loss2


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 4)
        self.conv2 = GATConv(4, num_classes)

    def forward(self, node_feature, edge_index):
        edge_index = edge_index.long()
        node_feature = node_feature.float()

        x = self.conv1(node_feature, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)

    def loss(self, score, adj, gamma: float = 1, mean: bool = True):
        tmp = 1 + torch.mul(score.unsqueeze(1), adj.cuda())
        tmp = torch.prod(tmp.pow(-1), dim=0)
        loss1 = tmp.mean() if mean else tmp.sum()
        loss2 = score.mean() if mean else score.sum()

        return loss1 + gamma * loss2

