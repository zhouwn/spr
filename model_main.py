import torch
from torch import nn
# from layers import GraphConvolution,DHGLayer,HGNN_conv
from model_layers import RoleLayer, GraphConvolution, HGNN_conv
from HiGCN import HiGCN
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Hyper_GNN(nn.Module):
    """
    Dynamic Hypergraph Convolution Neural Network with a GCN-style input layer
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['node_feature_shape']  # 节点特征维度d
        self.n_role = kwargs['N_roles']               # 角色数量R
        self.role_node_num = kwargs['role_node_num']  # 角色中的节点数量N
        self.layer_spec = kwargs['layer_spec']        # 经过一层GCN/HGNN卷积后的节点维度
        self.n_layers = kwargs['n_layers']            # 层数
        activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])        # 激活函数

        self.role_conv = nn.ModuleList(
            [GraphConvolution(
            dim_in=self.dim_feat,  # 输入节点维度
            dim_out=self.layer_spec,  # 输出节点维度
            dropout_rate=kwargs['dropout_rate'],
            activation=activations[0])] +
            [RoleLayer(
            dim_in=self.layer_spec,  # 经过一层GCN/HGNN卷积后的节点维度
            N_Role=self.n_role,           # 角色数量
            role_node_num=self.role_node_num,  # 角色中的节点数量
            k_dim=kwargs['knn_dim'],    # knn维度
            dropout_rate=kwargs['dropout_rate'],  # 这个是角色卷积的dropout比例
            activation=activations[i]) for i in range(1, self.n_layers)])
        """self.role_conv = RoleLayer(
                dim_in=self.dim_feat,
                N_Role=self.n_role,
                role_node_num=self.role_node_num,
                dropout_rate=kwargs['dropout_rate'],
                k_dim=kwargs['knn_dim'],
                activation=self.activation)"""

        self.simplex_conv = HiGCN(
            node_feature_shape=self.dim_feat,
            num_hid=kwargs['num_hid'],
            Order=kwargs['Order'],
            K=kwargs['K'],
            alpha=kwargs['alpha'],
            dprate=kwargs['dprate'],
            Init=kwargs['Init'])  # 这个是单纯复形卷积的dropout比例

        self.fc = nn.Sequential(nn.Linear(self.dim_feat*2, self.dim_feat), nn.ReLU(), nn.Linear(self.dim_feat, 1))  # contact 后的卷积维度
        # self.fc = nn.Sequential(nn.Linear(self.dim_feat, self.dim_feat//2), nn.ReLU(), nn.Linear(self.dim_feat//2, 1))  # 加和、平均的卷积维度

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :param G:
        :return:
        """
        G = kwargs.get('G')  # 图G，nx
        node_feature = kwargs.get('node_feature')  # 节点特征
        HL = kwargs.get('HL')  # 这个是单纯复形
        role_node_idx = kwargs.get('role_node_idx')  # 这个是每个角色包含的节点list
        edge_list = kwargs.get('edge_list')  # 这个是边列表，用于GCN卷积
        node_role_norm_matrix = kwargs.get('node_role_norm_matrix')  # 这个是归一化后的角色邻接矩阵，用于HGNN卷积



        for i_layer in range(self.n_layers):
            node_feature = self.role_conv[i_layer](G, role_node_idx, node_feature, edge_list, node_role_norm_matrix).to(device)

        simplex_ft = self.simplex_conv(node_feature, HL).to(device)
        # role_ft = self.role_conv(G, role_node_idx, node_feature, edge_list, node_role_norm_matrix).to(device)  # 这里输入的是R*N*d的超图特征矩阵

        combined_ft = torch.cat((node_feature, simplex_ft), dim=1)  # concat
        # combined_ft = role_ft + simplex_ft

        scores = self.fc(combined_ft)

        scores = torch.sigmoid(scores)

        return scores



    def loss(self, score, adj, gamma: float = 1, mean: bool = True):
        tmp = 1 + torch.mul(score.unsqueeze(1), adj.cuda())
        tmp = torch.prod(tmp.pow(-1), dim=0)
        loss1 = tmp.mean() if mean else tmp.sum()
        loss2 = score.mean() if mean else score.sum()

        return loss1 + gamma * loss2

"""
这个是计算权重的卷积方式
        role_ft = role_ft.view(len(role_ft), 1, role_ft.size(1))
        simplex_ft = simplex_ft.view(len(simplex_ft), 1, simplex_ft.size(1))

        two_layer_ft = torch.cat((role_ft, simplex_ft), dim=1)

        scores = []
        n_edges = two_layer_ft.size(1)
        for i in range(n_edges):
            scores.append(self.fc(two_layer_ft[:, i]))
        scores = torch.softmax(torch.stack(scores, 1), 1)

        scores = (scores * two_layer_ft).sum(1)

        scores = self.fc(self.dropout(scores))
"""