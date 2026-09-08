import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 添加类型转换和设备移动函数
def ensure_float32(tensor):
    """确保张量是float32类型"""
    if tensor is None:
        return None
    if tensor.dtype == torch.float64:
        return tensor.float()
    return tensor


def move_to_device(obj):
    """递归地将对象中的所有张量移动到设备上"""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v) for v in obj)
    else:
        return obj


# 稀疏矩阵工具函数 (兼容PyTorch 1.7)
def build_sparse_incidence_matrix(num_nodes, simplex_list, weights=None, simplex_type="edge"):
    """
    构建稀疏关联矩阵 (兼容PyTorch 1.7)
    """
    rows = []
    cols = []
    values = []

    if simplex_type == "edge":
        # 边关联矩阵
        for simplex_idx, (u, v) in enumerate(simplex_list):
            rows.extend([u, v])
            cols.extend([simplex_idx, simplex_idx])
            values.extend([1.0, 1.0])
    else:
        # 面关联矩阵
        for simplex_idx, face in enumerate(simplex_list):
            for node in face:
                rows.append(node)
                cols.append(simplex_idx)
                values.append(1.0)

    if weights is not None:
        # 应用权重 (适配PyTorch 1.7)
        weights = weights.squeeze()
        for col in cols:
            values[col] = values[col] * weights[col].item()

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)  # 确保float32

    # PyTorch 1.7兼容的稀疏矩阵创建方式
    return torch.sparse.FloatTensor(indices, values, torch.Size([num_nodes, len(simplex_list)])).coalesce()


class OptimizedTransform(nn.Module):
    """优化的变换模块 (兼容PyTorch 1.7)"""

    def __init__(self, dim_in, k):
        super().__init__()
        self.k = k
        self.dim_in = dim_in
        # 使用轻量级变换
        self.linear = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU()
        )
        self.attention = nn.Linear(dim_in * 2, 1)

    def forward(self, region_feats):
        # 确保数据类型正确
        region_feats = ensure_float32(region_feats)

        N, k, d = region_feats.size()

        # 计算注意力权重
        center_feats = region_feats.mean(dim=1, keepdim=True)  # (N, 1, d)
        center_expanded = center_feats.expand_as(region_feats)  # (N, k, d)

        # 计算每个节点与中心的相似度
        attention_input = torch.cat([region_feats, center_expanded], dim=-1)  # (N, k, 2d)
        attention_scores = self.attention(attention_input).squeeze(-1)  # (N, k)
        attention_weights = F.softmax(attention_scores, dim=1)  # (N, k)

        # 应用变换和注意力
        transformed = self.linear(region_feats)  # (N, k, d)
        weighted = transformed * attention_weights.unsqueeze(-1)  # (N, k, d)

        return weighted


class OptimizedVertexConv(nn.Module):
    """优化的顶点卷积模块 (兼容PyTorch 1.7)"""

    def __init__(self, dim_in, k):
        super().__init__()
        self.transform = OptimizedTransform(dim_in, k)
        self.pool = nn.Linear(dim_in * k, dim_in)  # 使用线性层替代大卷积

    def forward(self, region_feats):
        region_feats = ensure_float32(region_feats)
        transformed = self.transform(region_feats)  # (N, k, d)
        flattened = transformed.contiguous().view(transformed.size(0), -1)  # (N, k*d)
        pooled = self.pool(flattened)  # (N, d)
        return pooled


class HiGCN_prop(MessagePassing):
    '''
    优化的传播类，使用稀疏矩阵操作 (兼容PyTorch 1.7)
    '''

    def __init__(self, K, alpha, Init, Order=2, bias=True, **kwargs):
        super(HiGCN_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.Order = Order
        self.Init = Init

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))

        self.temp = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha] = 1.0
        elif self.Init == 'PPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
            self.temp.data[-1] = (1 - self.alpha) ** self.K
        elif self.Init == 'NPPR':
            for k in range(self.K + 1):
                self.temp.data[k] = self.alpha ** k
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3 / (self.K + 1))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data / torch.sum(torch.abs(self.temp.data))

    def forward(self, x, HL):
        """使用稀疏矩阵乘法优化 (兼容PyTorch 1.7)"""
        # 确保数据类型正确
        x = ensure_float32(x)

        if not isinstance(HL, torch.sparse.FloatTensor):
            HL = HL.to_sparse()

        HL = HL.coalesce().to(x.device)

        hidden = x * self.temp[0]
        current = x.clone()

        try:
            # 检查是否是稀疏张量
            if HL.is_sparse:
                for k in range(self.K):
                    # 使用稀疏矩阵乘法
                    current = torch.sparse.mm(HL, current)
                    gamma = self.temp[k + 1]
                    hidden = hidden + gamma * current

                    # 清除中间变量
                    if k < self.K - 1:
                        del current
                        torch.cuda.empty_cache()
                        current = hidden.clone()
        except RuntimeError:
            # 回退到稠密矩阵乘法
            HL_dense = HL.to_dense().float()  # 确保float32
            for k in range(self.K):
                current = torch.mm(HL_dense, current)
                gamma = self.temp[k + 1]
                hidden = hidden + gamma * current
                if k < self.K - 1:
                    del current
                    torch.cuda.empty_cache()
                    current = hidden.clone()

        return hidden

    def __repr__(self):
        return '{}(Order={}, K={}, filterWeights={})'.format(
            self.__class__.__name__, self.Order, self.K, self.temp)


class HiGCN(torch.nn.Module):
    """优化的HiGCN模块 (兼容PyTorch 1.7)"""

    def __init__(self, **kwargs):
        super(HiGCN, self).__init__()
        self.Order = min(kwargs.get('Order', 3), 3)  # 限制最大Order为3
        node_feature_shape = kwargs['node_feature_shape']
        self.num_hid = kwargs['num_hid']
        self.K = kwargs['K']
        self.alpha = kwargs['alpha']
        self.dprate = kwargs['dprate']
        self.Init = kwargs['Init']

        # 使用共享权重减少参数
        self.lin_in = nn.Linear(node_feature_shape, self.num_hid)
        self.hgc = nn.ModuleList([
            HiGCN_prop(self.K, self.alpha, self.Init, self.Order)
            for _ in range(self.Order)
        ])

        # 输出层
        self.fc1 = nn.Linear(self.num_hid, node_feature_shape)

        # 优化的顶点卷积
        self.vc_edge = OptimizedVertexConv(node_feature_shape, 2)
        self.vc_face = OptimizedVertexConv(node_feature_shape, 3)

        # 轻量级权重计算
        self.edge_weight = nn.Linear(node_feature_shape, 1)
        self.face_weight = nn.Linear(node_feature_shape, 1)

        # 注意力机制融合多阶特征
        self.attention = nn.MultiheadAttention(
            embed_dim=self.num_hid,
            num_heads=4
        )

    def get_simplex_ft(self, node_feature, edge, faces):
        """优化特征提取，避免大视图操作"""
        if not edge:
            return None, None

        # 确保数据类型正确
        node_feature = ensure_float32(node_feature)

        # 边特征
        edge_matrix = []
        for u, v in edge.keys():
            edge_matrix.append([u, v])
        edge_matrix = torch.tensor(edge_matrix, dtype=torch.long, device=device)
        edge_node_feature = node_feature[edge_matrix.view(-1)].view(
            len(edge_matrix), 2, node_feature.shape[1])

        # 面特征
        if faces:
            faces_matrix = torch.tensor(faces, dtype=torch.long, device=device)
            faces_node_feature = node_feature[faces_matrix.view(-1)].view(
                len(faces), 3, node_feature.shape[1])
        else:
            faces_node_feature = None

        return edge_node_feature, faces_node_feature

    def get_HL(self, G_single, train_adj_matrices, edge, faces, edge_weights, face_weights):
        """直接构建稀疏关联矩阵 (兼容PyTorch 1.7)"""
        num_nodes = len(G_single.nodes())

        # 节点关联矩阵（0-simplex）
        if isinstance(train_adj_matrices, np.ndarray):
            train_adj_matrices = torch.tensor(train_adj_matrices, device=device).float()
        node_H = train_adj_matrices.to_sparse()

        # 边关联矩阵（1-simplex）
        edge_list = list(edge.keys())
        edge_H = build_sparse_incidence_matrix(
            num_nodes=num_nodes,
            simplex_list=edge_list,
            weights=edge_weights,
            simplex_type="edge"
        )

        # 面关联矩阵（2-simplex）
        face_H = build_sparse_incidence_matrix(
            num_nodes=num_nodes,
            simplex_list=faces,
            weights=face_weights,
            simplex_type="face"
        ) if faces else edge_H  # 如果没有面，使用边矩阵

        return {1: node_H, 2: edge_H, 3: face_H}

    def forward(self, node_feature, G_single, train_adj_matrices, edge, faces, edge_node_ft, faces_node_ft):
        # 确保输入在正确的设备上
        node_feature = node_feature.to(device)
        node_feature = ensure_float32(node_feature)

        # 按需提取特征
        if edge_node_ft is None or faces_node_ft is None:
            edge_node_ft, faces_node_ft = self.get_simplex_ft(
                node_feature, edge, faces)

        # 轻量级权重计算
        if edge_node_ft is not None:
            edge_node_ft = edge_node_ft.to(device)
            edge_node_ft = ensure_float32(edge_node_ft)
            edge_weights = self.edge_weight(edge_node_ft.mean(dim=1))
        else:
            edge_weights = None

        if faces_node_ft is not None:
            faces_node_ft = faces_node_ft.to(device)
            faces_node_ft = ensure_float32(faces_node_ft)
            face_weights = self.face_weight(faces_node_ft.mean(dim=1))
        else:
            face_weights = None

        # 构建稀疏关联矩阵
        HL = self.get_HL(G_single, train_adj_matrices, edge, faces, edge_weights, face_weights)

        # 初始变换
        xx = F.dropout(node_feature, p=self.dprate, training=self.training)
        xx = self.lin_in(xx)

        # 多阶特征存储
        features = []
        for i in range(self.Order):
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)

            # 获取当前阶的关联矩阵
            hl_matrix = HL.get(i + 1, HL[1])  # 默认为节点关联矩阵

            # 使用优化的传播
            xx = self.hgc[i](xx, hl_matrix)
            features.append(xx.unsqueeze(1))  # [N, 1, D]

        # 使用注意力机制融合多阶特征
        features_tensor = torch.cat(features, dim=1)  # [N, Order, D]
        attn_output, _ = self.attention(
            features_tensor, features_tensor, features_tensor
        )
        combined = attn_output.mean(dim=1)  # 平均注意力输出 [N, D]

        # 输出层
        scores = F.dropout(combined, p=self.dprate, training=self.training)
        scores = self.fc1(scores)

        # 及时清除中间变量
        del features, features_tensor, attn_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores


class Role_GCN(torch.nn.Module):
    """优化的Role_GCN模块 (兼容PyTorch 1.7)"""

    def __init__(self, **kwargs):
        super(Role_GCN, self).__init__()
        node_feature_shape = kwargs['node_feature_shape']
        role_feature_shape = kwargs['role_feature_shape']
        self.num_hid = kwargs['num_hid']
        self.K = kwargs['K']
        self.alpha = kwargs['alpha']
        self.dprate = kwargs['dprate']
        self.Init = kwargs['Init']

        self.lin_in = nn.Linear(node_feature_shape, self.num_hid)
        self.hgc = HiGCN_prop(self.K, self.alpha, self.Init)
        self.fc1 = nn.Linear(self.num_hid, node_feature_shape)

        # 轻量级角色权重计算
        self.role_weight = nn.Sequential(
            nn.Linear(role_feature_shape, node_feature_shape // 4),
            nn.ReLU(),
            nn.Linear(node_feature_shape // 4, 1)
        )

    def get_HL(self, role_weights, node_role_matrix):
        """完全稀疏化的HL矩阵构造 (兼容PyTorch 1.7)"""
        # 确保数据类型正确
        role_weights = ensure_float32(role_weights)

        # 转换为稀疏格式
        if not node_role_matrix.is_sparse:
            node_role_matrix = node_role_matrix.to_sparse()

        indices = node_role_matrix._indices()
        values = node_role_matrix._values()

        # 确保值为float32
        if values.dtype != torch.float32:
            values = values.float()

        # 节点度计算
        node_in_simplex = scatter_add(
            values,
            indices[0],
            dim_size=node_role_matrix.size(0)
        )
        r_inv_sqrt_node_in_simplex = 1.0 / torch.sqrt(node_in_simplex).clamp(min=1e-10)
        r_inv_sqrt_node_in_simplex[torch.isinf(r_inv_sqrt_node_in_simplex)] = 0.0

        # 角色度计算
        simplex_dim = scatter_add(
            values,
            indices[1],
            dim_size=node_role_matrix.size(1)
        )
        r_inv_sqrt_simplex_dim = 1.0 / torch.sqrt(simplex_dim).clamp(min=1e-10)
        r_inv_sqrt_simplex_dim[torch.isinf(r_inv_sqrt_simplex_dim)] = 0.0

        # 应用权重
        scaled_vals = values * role_weights.squeeze()[indices[1]]

        # 应用节点度缩放
        scaled_vals = scaled_vals * r_inv_sqrt_node_in_simplex[indices[0]]

        # 应用角色度缩放
        scaled_vals = scaled_vals * r_inv_sqrt_simplex_dim[indices[1]]

        # 创建最终稀疏矩阵
        HL = torch.sparse.FloatTensor(
            indices,
            scaled_vals,
            node_role_matrix.size()
        ).to(device)

        # 转置并再次相乘
        HL_T = torch.sparse.transpose(HL, 0, 1)
        HL = torch.sparse.mm(HL, HL_T)

        return HL.coalesce()

    def forward(self, G_loop, node_feature, role_feature, node_role_matrix):
        # 确保输入在正确的设备上
        node_feature = node_feature.to(device)
        role_feature = role_feature.to(device)
        node_role_matrix = node_role_matrix.to(device)

        # 确保数据类型为float32
        node_feature = ensure_float32(node_feature)
        role_feature = ensure_float32(role_feature)

        role_weights = self.role_weight(role_feature)
        role_weights = F.softmax(role_weights, dim=0)

        # 使用优化的HL计算
        role_HL = self.get_HL(role_weights, node_role_matrix)

        xx = F.dropout(node_feature, p=self.dprate, training=self.training)
        xx = self.lin_in(xx)
        xx = F.dropout(xx, p=self.dprate, training=self.training)
        xx = self.hgc(xx, role_HL)
        scores = F.dropout(xx, p=self.dprate, training=self.training)
        scores = self.fc1(scores)

        return scores


class Hyper_GNN(nn.Module):
    """优化的主模型 (兼容PyTorch 1.7)"""

    def __init__(self, **kwargs):
        super().__init__()
        self.dim_feat = kwargs['node_feature_shape']
        self.n_role = kwargs['N_roles']

        # 使用优化的子模块
        self.role_conv = Role_GCN(
            node_feature_shape=self.dim_feat,
            role_feature_shape=kwargs['role_feature_shape'],
            num_hid=kwargs['role_num_hid'],
            K=kwargs['role_K'],
            alpha=kwargs['role_alpha'],
            dprate=kwargs['role_dprate'],
            Init=kwargs['role_Init'])

        self.simplex_conv = HiGCN(
            node_feature_shape=self.dim_feat,
            num_hid=kwargs['sim_num_hid'],
            Order=kwargs['sim_Order'],
            K=kwargs['sim_K'],
            alpha=kwargs['sim_alpha'],
            dprate=kwargs['sim_dprate'],
            Init=kwargs['sim_Init'])

        # 轻量级融合层
        self.fc = nn.Sequential(
            nn.Linear(self.dim_feat * 2, self.dim_feat),
            nn.ReLU(),
            nn.Linear(self.dim_feat, 1)
        )

    def forward(self, **kwargs):
        # 只提取必要的变量
        data = kwargs.get('data', {})
        G_loop = kwargs.get('G_loop', None)
        G_single = kwargs.get('G_single', None)

        node_feature = data.get('node_feature')
        role_feature = data.get('role_feature')
        node_role_matrix = data.get('node_role_matrix')

        # 确保输入在正确的设备上
        if node_feature is not None:
            node_feature = node_feature.to(device)
        if role_feature is not None:
            role_feature = role_feature.to(device)
        if node_role_matrix is not None:
            node_role_matrix = node_role_matrix.to(device)

        # 递归移动所有嵌套数据结构
        data = move_to_device(data)

        # 条件计算角色特征
        if role_feature is not None and node_role_matrix is not None:
            role_ft = self.role_conv(
                G_loop=G_loop,
                node_feature=node_feature,
                role_feature=role_feature,
                node_role_matrix=node_role_matrix
            )
        else:
            role_ft = torch.zeros_like(node_feature).to(device)

        # 计算单纯形特征
        simplex_ft = self.simplex_conv(
            node_feature=node_feature,
            G_single=G_single,
            train_adj_matrices=data.get('train_adj_matrices'),
            edge=data.get('edge', {}),
            faces=data.get('faces', []),
            edge_node_ft=data.get('edge_node_ft'),
            faces_node_ft=data.get('faces_node_ft')
        )

        # 特征融合
        combined_ft = torch.cat((role_ft, simplex_ft), dim=1)
        scores = self.fc(combined_ft)
        scores = torch.sigmoid(scores)

        return scores

    def origin_loss(self, score, adj, gamma: float = 1, mean: bool = True):
        # 高效损失计算
        score = ensure_float32(score)
        adj = ensure_float32(adj)

        # 确保在相同设备上
        adj = adj.to(device)
        score = score.to(device)

        prod_term = (1 + score.unsqueeze(1) * adj).prod(dim=0)
        loss1 = prod_term.mean() if mean else prod_term.sum()
        loss2 = score.mean() if mean else score.sum()
        return loss1 + gamma * loss2