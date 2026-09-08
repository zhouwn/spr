import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from torch import nn
# from layers import GraphConvolution,DHGLayer,HGNN_conv
from model_layers import RoleLayer, GraphConvolution, HGNN_conv
from HiGCN import HiGCN
import torch.nn.functional as F
from Role_GCN import Role_GCN
from torch_sparse import SparseTensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


try:
    from torch_scatter import scatter_add
except Exception as e:
    scatter_add = None  # 如果你没有装 torch_scatter，下面会给降级方案（但不建议）

class Hyper_GNN(nn.Module):
    """
    Dynamic Hypergraph Convolution Neural Network with a GCN-style input layer
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.dim_feat = kwargs['node_feature_shape']  # 节点特征维度d
        self.n_role = kwargs['N_roles']               # 角色数量R
        # activations = nn.ModuleList([nn.ReLU() for i in range(self.n_layers - 1)] + [nn.LogSoftmax(dim=-1)])        # 激活函数
        """self.role_conv1 = nn.ModuleList(
            [HiGCN(
                node_feature_shape=self.dim_feat,
                num_hid=kwargs['num_hid'],
                Order=kwargs['Order'],
                K=kwargs['K'],
                alpha=kwargs['alpha'],
                dprate=kwargs['dprate'])] +
            [simplex_Layer(
                dim_in=self.dim_feat,  # 经过一层GCN/HGNN卷积后的节点维度
                edge_num=self.edge_num,
                faces_num=self.faces_num,
                dropout_rate=kwargs['dropout_rate'],  # 这个是角色卷积的dropout比例
                activation=activations[i]) for i in range(1, self.n_layers)])"""

        self.role_conv = Role_GCN(
            node_feature_shape=self.dim_feat,
            role_feature_shape=kwargs['role_feature_shape'],
            num_hid=kwargs['role_num_hid'],
            K=kwargs['role_K'],
            alpha=kwargs['role_alpha'],
            dprate=kwargs['role_dprate'],
            Init=kwargs['role_Init'])  # 这个是单纯复形卷积的dropout比例


        """self.role_conv = RoleLayer(
                dim_in=self.dim_feat,
                N_Role=self.n_role,
                role_node_num=self.role_node_num,
                dropout_rate=kwargs['dropout_rate'],
                k_dim=kwargs['knn_dim'],
                activation=self.activation)"""

        self.simplex_conv = HiGCN(
            node_feature_shape=self.dim_feat,
            num_hid=kwargs['sim_num_hid'],
            Order=kwargs['sim_Order'],
            K=kwargs['sim_K'],
            alpha=kwargs['sim_alpha'],
            dprate=kwargs['sim_dprate'],
            Init=kwargs['sim_Init'])  # 这个是单纯复形卷积的dropout比例

        self.fc = nn.Sequential(nn.Linear(self.dim_feat*2, self.dim_feat), nn.ReLU(), nn.Linear(self.dim_feat, 1))  # contact 后的卷积维度
        # self.fc = nn.Sequential(nn.Linear(self.dim_feat, self.dim_feat//2), nn.ReLU(), nn.Linear(self.dim_feat//2, 1))  # 加和、平均的卷积维度

    def forward(self, **kwargs):
        """
        :param feats:
        :param edge_dict:
        :param G:
        :return:
        """
        G_loop = kwargs.get('G_loop')  # 图G，nx
        G_single = kwargs.get('G_single')

        node_feature = kwargs.get('data')['node_feature']  # 节点特征
        role_feature = kwargs.get('data')['role_feature']  # 这个是角色特征
        node_role_matrix = kwargs.get('data')['node_role_matrix']  # 角色关联矩阵
        train_adj_matrices = kwargs.get('data')['train_adj_matrices']
        edge = kwargs.get('data')['edge']
        faces = kwargs.get('data')['faces']
        edge_node_ft = kwargs.get('data')['edge_node_ft']
        faces_node_ft = kwargs.get('data')['faces_node_ft']

        node_feature = node_feature.to(device).float()
        role_feature = role_feature.to(device).float()
        if isinstance(node_role_matrix, torch.Tensor):
            node_role_matrix = node_role_matrix.to(device)
        if isinstance(train_adj_matrices, torch.Tensor):
            train_adj_matrices = train_adj_matrices.to(device)
        edge_node_ft = edge_node_ft.to(device).float()
        faces_node_ft = faces_node_ft.to(device).float()

        role_ft = self.role_conv(G_loop, node_feature, role_feature, node_role_matrix).to(device)

        simplex_ft = self.simplex_conv(node_feature, G_single, train_adj_matrices, edge, faces, edge_node_ft, faces_node_ft)


        # role_ft = self.role_conv(G, role_node_idx, node_feature, edge_list, node_role_norm_matrix).to(device)  # 这里输入的是R*N*d的超图特征矩阵


        simplex_ft = simplex_ft.to(device)

        combined_ft = torch.cat((role_ft, simplex_ft), dim=1)  # concat
        # combined_ft = role_ft + simplex_ft

        scores = self.fc(combined_ft)

        scores = torch.sigmoid(scores)

        return scores


    def loss(self, score, adj, gamma: float = 1.0, mean: bool = True):
        device = score.device
        s = score.view(-1).to(device)   # [N], 节点得分
        eps = 1e-12

        # --- Case 1: torch_sparse.SparseTensor ---
        if isinstance(adj, SparseTensor):
            # 取 COO 形式
            row, col, val = adj.coo()   # row/col: [E], val: [E] 或 None
            row = row.to(device)
            col = col.to(device)
            if val is None:
                # 无权图默认权重 1
                val = torch.ones(row.size(0), device=device, dtype=s.dtype)
            else:
                val = val.to(device)

            f = 1.0 + s[row] * val            # [E]
            f = torch.clamp(f, min=eps)
            neg_log_f = -torch.log(f)

            if scatter_add is not None:
                vec_log = scatter_add(neg_log_f, col, dim=0,
                                      dim_size=adj.sparse_sizes()[1])
            else:
                vec_log = torch.zeros(adj.sparse_sizes()[1],
                                      device=device, dtype=neg_log_f.dtype)
                vec_log.index_add_(0, col, neg_log_f)

            vec = torch.exp(vec_log)          # 对每列 j：∏_i (1+s_i A_ij)^(-1)

        # --- Case 2: PyTorch 稀疏 COO Tensor ---
        elif isinstance(adj, torch.Tensor) and adj.layout == torch.sparse_coo:
            A = adj.coalesce().to(device)
            idx = A.indices()                  # [2, E]
            row, col = idx[0], idx[1]
            val = A.values().to(device)
            f = 1.0 + s[row] * val
            f = torch.clamp(f, min=eps)
            neg_log_f = -torch.log(f)

            if scatter_add is not None:
                vec_log = scatter_add(neg_log_f, col, dim=0, dim_size=A.size(1))
            else:
                vec_log = torch.zeros(A.size(1), device=device, dtype=neg_log_f.dtype)
                vec_log.index_add_(0, col, neg_log_f)

            vec = torch.exp(vec_log)

        # --- Case 3: SciPy 稀疏矩阵 ---
        elif sp.issparse(adj):
            coo = adj.tocoo()
            row = torch.from_numpy(coo.row.astype(np.int64)).to(device)
            col = torch.from_numpy(coo.col.astype(np.int64)).to(device)
            val = torch.from_numpy(coo.data).to(device=device, dtype=s.dtype)
            f = 1.0 + s[row] * val
            f = torch.clamp(f, min=eps)
            neg_log_f = -torch.log(f)

            if scatter_add is not None:
                vec_log = scatter_add(neg_log_f, col, dim=0, dim_size=coo.shape[1])
            else:
                vec_log = torch.zeros(coo.shape[1], device=device, dtype=neg_log_f.dtype)
                vec_log.index_add_(0, col, neg_log_f)

            vec = torch.exp(vec_log)

        # --- Case 4: 稠密张量（仅小图或你明确允许变稠密） ---
        else:
            A = adj.to(device)
            tmp = 1.0 + s.unsqueeze(1) * A     # 这里才允许标量 + 张量
            tmp = torch.clamp(tmp, min=eps)
            vec = torch.prod(tmp.pow(-1), dim=0)

        loss1 = vec.mean() if mean else vec.sum()
        loss2 = s.mean() if mean else s.sum()
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
