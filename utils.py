import networkx as nx
import numpy as np
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
import igraph
from sklearn.preprocessing import normalize
import pickle
import dgl
import yaml
import numpy as np
import torch
import os
from torch import nn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from collections import defaultdict
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import degree, add_self_loops
from torch_sparse import SparseTensor


def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def save_to_file(path, data, network):
    txt_path = os.path.join(path, f"{network}.txt")

    with open(txt_path, 'a') as file:  # 'a' 模式表示追加内容到文件末尾
        file.write(data + '\n')  # 添加换行符以便于阅读

def save_config_to_txt(config, file_path, network):
    # 确保文件夹路径存在，如果不存在则创建它
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    # 完整的文本文件路径
    txt_path = os.path.join(file_path, f"{network}.txt")

    with open(txt_path, 'w') as file:  # 'w' 模式表示写入内容，如果文件已存在则覆盖
        for key in config:
            # 检查键是否是您想要保存的超参数
            if key in ['seed', 'num_feats', 'n_role', 'activation', 'gamma', 'learning_rate', 'weight_decay', 'patience', 'sim_num_hidden', 'sim_order', 'sim_k',
                       'sim_alpha', 'sim_dprate', 'knn_dim', 'sim_Init', 'role_num_hidden', 'role_k', 'role_alpha', 'role_dprate', 'role_Init']:
                file.write(f"{key}: {config[key]}\n")  # 将超参数及其值写入文件
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, logger, network, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.logging = logger
        self.network = network
    def __call__(self, val_loss, model,epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'Validation decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            save_to_file(path=self.path,
                         data=f"Validation decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...",
                         network=self.network)
        # torch.save(model.state_dict(), os.path.join(self.path, f"checkpoint_{epoch}.pt"))
        self.val_loss_min = val_loss


def get_G(network):
    datadir = os.path.join('.', 'data', 'data', 'realworld', f'{network}.txt')

    G = nx.Graph()

    for line in open(datadir):
        strlist = line.split()
        n1 = int(strlist[0])
        n2 = int(strlist[1])
        G.add_edges_from([(n1, n2)])
    nodes = G.nodes()
    G.add_edges_from([(node, node) for node in nodes])
    return G

def cos_dis(X):
    """
    cosine distance
    :param X: (N, d)
    :return: (N, N)
    """
    X = nn.functional.normalize(X)
    XT = X.transpose(0, 1)
    return torch.matmul(X, XT)

def save_dict_to_yaml(dict_value: dict, save_path: str):
    with open(save_path, 'w', encoding="utf-8") as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))

def counts_high_order_nodes(G, depth=2):
    NODES_LIST = list(G.nodes)
    output = {}
    output = output.fromkeys(NODES_LIST)
    for node in NODES_LIST:
        layers = dict(nx.bfs_successors(G, source=node, depth_limit=depth))
        high_order_nodes = 0
        for i in layers.keys():
            high_order_nodes += len(layers[i])
        # high_order_nodes = sum([len(i) for i in layers.values()])
        output[node] = high_order_nodes

    return output


def LoadFivefeature(G):
    """简化的节点特征生成，避免昂贵的中心性计算"""
    # 获取有序节点列表
    nodes = sorted(G.nodes())
    node_count = len(nodes)

    # 1. 度特征 (归一化)
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    degree_list = np.array([degrees[node] / max_degree for node in nodes])[:, None]

    # 2. 局部聚类系数 (归一化)
    clustering = nx.clustering(G)
    max_clustering = max(clustering.values()) if clustering.values() else 1
    clustering_list = np.array([clustering.get(node, 0) / max_clustering for node in nodes])[:, None]

    # 3. PageRank (归一化)
    try:
        pagerank = nx.pagerank(G, max_iter=50)  # 减少迭代次数
        max_pagerank = max(pagerank.values()) if pagerank.values() else 1
        pagerank_list = np.array([pagerank.get(node, 0) / max_pagerank for node in nodes])[:, None]
    except:
        pagerank_list = np.zeros((node_count, 1))

    # 4. 常数特征
    constant_list = np.ones((node_count, 1))

    # 合并特征 (移除昂贵的中心性特征)
    node_features = np.concatenate((degree_list, clustering_list, pagerank_list, constant_list), axis=1)

    # 处理NaN值
    node_features[np.isnan(node_features)] = 0

    return node_features

def LoadOnefeature(num_nodes, num_feats):

    return torch.ones(num_nodes, num_feats)

def LoadDegreefeature(G):

    NODES_LIST = list(G.nodes)
    degrees = list(G.degree())
    # 确定最大度数
    max_degree = max(dict(degrees).values())

    # 计算最大度数的二进制表示的长度
    max_length = len(bin(max_degree)[2:])

    # 初始化一个空列表来存储所有节点的特征向量
    features_list = []

    for node, degree in degrees:
        # 将度数转换为二进制表示
        binary_feature = bin(degree)[2:]
        # 前面用0填充以保持一致的长度
        binary_feature = binary_feature.zfill(max_length)
        # 将二进制字符串转换为数字列表
        feature = [int(b) for b in binary_feature]
        # 将特征向量添加到列表中
        features_list.append(feature)

    # 使用vstack将特征列表转换为矩阵
    binary_degree_features_matrix = np.vstack(features_list)

    return binary_degree_features_matrix


def sparse_adjacency_matrix(G):
    """创建稀疏邻接矩阵表示"""
    # 创建节点到索引的映射
    nodes = sorted(G.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    # 构建COO格式的稀疏邻接矩阵
    rows, cols = [], []
    for u, v in G.edges():
        idx_u = node_index[u]
        idx_v = node_index[v]
        # 无向图，添加两个方向的边
        rows.append(idx_u)
        cols.append(idx_v)
        if u != v:  # 避免自环重复添加
            rows.append(idx_v)
            cols.append(idx_u)

    # 值为1表示存在边
    data = np.ones(len(rows))

    # 创建稀疏矩阵
    sparse_adj = coo_matrix((data, (rows, cols)), shape=(n, n))
    return sparse_adj, node_index


def LoadRealworldData(network, g):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datadir = os.path.join('.', 'data', 'data', 'realworld')
    ig_g = igraph.Graph().Read_Edgelist(os.path.join(datadir, f'{network}.txt'), directed=False)

    # 1. 创建节点索引映射（确保连续索引）
    nodes = sorted(g.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    # 2. 生成节点特征（简化版）
    node_feature = LoadFivefeature(g)
    node_feature = torch.from_numpy(node_feature).to(device)

    # 3. 创建稀疏邻接矩阵
    sparse_adj, _ = sparse_adjacency_matrix(g)

    return node_feature, sparse_adj, ig_g

""" 放置于LoadRealworldData函数最后
    # 生成Kmeans的超图关联矩阵
    KMeans_adj = generate_hg_KMeans_adj(node_feature)
    KMeans_adj_norm = generate_G_from_H(KMeans_adj)
    KMeans_adj_norm = torch.Tensor(KMeans_adj_norm).to(device)

    # 生成knn的超图关联矩阵
    Knn_adj = generate_hg_Knn_adj(node_feature)
    Knn_adj_norm = generate_G_from_H(Knn_adj)
    Knn_adj_norm = torch.Tensor(Knn_adj_norm).to(device)

    # 生成k跳邻居的超图关联矩阵
    ThreeHop_adj = generate_hg_hop_adj(dense_adj_matrix)
    ThreeHop_adj_norm = generate_G_from_H(ThreeHop_adj)
    ThreeHop_adj_norm = torch.Tensor(ThreeHop_adj_norm).to(device)
"""



def generate_hg_KMeans_adj(feature):
    num_clusters = 20

    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(feature.cpu().numpy())

    # 创建图
    G = dgl.DGLGraph()

    # 添加节点
    G.add_nodes(len(feature))

    # 添加边，根据KMeans聚类结果连接同一类别的节点
    src, dst = [], []
    for i in range(len(feature)):
        for j in range(len(feature)):
            if clusters[i] == clusters[j]:
                src.append(i)
                dst.append(j)

    G.add_edges(src, dst)

    # 将图转换为邻接矩阵
    adjacency_matrix = G.adjacency_matrix().to_dense()

    return adjacency_matrix

def generate_hg_Knn_adj(feature):
    k_neighbors = 20

    # 使用 k-NN 方法进行聚类
    knn = NearestNeighbors(n_neighbors=k_neighbors)
    knn.fit(feature.cpu().numpy())
    distances, indices = knn.kneighbors(feature.cpu().numpy())

    # 创建图
    G = dgl.DGLGraph()

    # 添加节点
    G.add_nodes(len(feature))

    # 添加边，根据 k-NN 结果连接节点
    src = torch.tensor([i for i in range(len(feature)) for _ in range(k_neighbors)])
    dst = torch.tensor([j for indices_i in indices for j in indices_i])

    # 使用一次性添加边的方式
    G.add_edges(src, dst)

    # 将图转换为邻接矩阵
    adjacency_matrix = G.adjacency_matrix().to_dense()

    return adjacency_matrix

def generate_hg_hop_adj(dense_adj_matrix):

    # 一跳邻居关联矩阵
    one_hop_adjacency_matrix = (dense_adj_matrix >= 1).float()

    # 二跳邻居关联矩阵
    two_hop_adjacency_matrix = (torch.matmul(dense_adj_matrix, dense_adj_matrix) >= 1).float()

    # 三跳邻居关联矩阵
    three_hop_adjacency_matrix = (torch.matmul(two_hop_adjacency_matrix, dense_adj_matrix) >= 1).float()

    # 合并一跳、二跳和三跳邻居关联矩阵
    combined_adjacency_matrix = one_hop_adjacency_matrix + two_hop_adjacency_matrix + three_hop_adjacency_matrix

    combined_adjacency_matrix = (combined_adjacency_matrix >= 1).float()

    return combined_adjacency_matrix


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = H.cpu().numpy()
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))

    # 这一部分是为了消除0的影响
    non_zero_indices = np.where(DV != 0)  # 找出非零元素的索引
    DV_inverse_sqrt = np.zeros_like(DV)
    DV_inverse_sqrt[non_zero_indices] = np.power(DV[non_zero_indices], -0.5)  # 仅对非零元素计算负平方根
    DV2 = np.mat(np.diag(DV_inverse_sqrt))
    # DV2 = np.mat(np.diag(np.power(DV, -0.5)))

    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = np.linalg.pinv(np.diag(DE)) * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * np.linalg.pinv(np.diag(DE)) * HT * DV2
        return G

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_simplex_HL1(role_weights, node_list, simplices):
    """
    role_weights: 角色权重张量，形状为 [n_roles, 1]
    node_list: 节点列表，包含所有节点索引
    simplices: 角色（单纯形）列表，每个元素是包含节点索引的元组
    """
    device = role_weights.device
    n_nodes = len(node_list)
    n_simplices = len(simplices)

    # 创建节点索引映射
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    # ==== 构建节点-角色关联矩阵（稀疏格式）====
    row_indices = []
    col_indices = []

    # 同时记录每个单纯形包含的节点（索引形式）
    simplex_node_indices = []

    for simplex_idx, simplex_nodes in enumerate(simplices):
        current_simplex_indices = []
        for node in simplex_nodes:
            if node in node_to_index:  # 确保节点在给定列表中
                node_idx = node_to_index[node]
                row_indices.append(node_idx)
                col_indices.append(simplex_idx)
                current_simplex_indices.append(node_idx)
        simplex_node_indices.append(current_simplex_indices)

    # ==== 处理空关联矩阵的情况 ====
    if len(row_indices) == 0:
        # 如果没有非零元素，直接返回空矩阵
        return SparseTensor(
            row=torch.empty(0, dtype=torch.long, device=device),
            col=torch.empty(0, dtype=torch.long, device=device),
            value=torch.empty(0, dtype=torch.float32, device=device),
            sparse_sizes=(n_nodes, n_nodes)
        )

    # 构建稀疏矩阵（用于计算度数）
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values = torch.ones(len(row_indices), dtype=torch.float32, device=device)

    node_role_matrix = torch.sparse_coo_tensor(
        indices,
        values,
        size=(n_nodes, n_simplices)
    ).coalesce()

    # ==== 计算度数 ====
    # 节点度数 (节点所属的单纯形数量)
    # 处理稀疏矩阵求和可能出现的空矩阵情况
    if node_role_matrix._nnz() == 0:
        node_in_simplex = torch.zeros(n_nodes, device=device)
    else:
        node_in_simplex = torch.sparse.sum(node_role_matrix, dim=1).to_dense()

    r_inv_sqrt_node_in_simplex = 1.0 / torch.sqrt(node_in_simplex)
    r_inv_sqrt_node_in_simplex[torch.isinf(r_inv_sqrt_node_in_simplex)] = 0.0

    # 单纯形度数 (单纯形包含的节点数量)
    # 使用之前记录的每个单纯形的节点数，避免稀疏求和
    simplex_dim = torch.zeros(n_simplices, dtype=torch.float32, device=device)
    for s_idx, node_indices in enumerate(simplex_node_indices):
        simplex_dim[s_idx] = len(node_indices)

    r_inv_sqrt_simplex_dim = 1.0 / torch.sqrt(simplex_dim)
    r_inv_sqrt_simplex_dim[torch.isinf(r_inv_sqrt_simplex_dim)] = 0.0

    # 获取每个角色（单纯形）的权重
    w_s = role_weights.squeeze() * r_inv_sqrt_simplex_dim

    # 初始化存储节点对的列表
    all_rows = []
    all_cols = []
    all_vals = []

    # 遍历每个单纯形
    for s_idx in range(n_simplices):
        # 当前单纯形包含的节点索引
        node_indices = simplex_node_indices[s_idx]
        num_nodes = len(node_indices)

        if num_nodes == 0:  # 跳过空单纯形
            continue

        # 获取当前单纯形的权重
        weight = w_s[s_idx]

        # 为当前单纯形中的所有节点对添加边（包括自环）
        for i in range(num_nodes):
            for j in range(num_nodes):
                all_rows.append(node_indices[i])
                all_cols.append(node_indices[j])
                # 如果权重是标量，直接使用；如果是张量，提取值
                if isinstance(weight, torch.Tensor) and weight.numel() > 1:
                    # 如果权重是向量，取对应元素
                    all_vals.append(weight[s_idx].item())
                else:
                    # 标量或单元素张量
                    all_vals.append(weight.item() if hasattr(weight, 'item') else weight)

    # 如果没有节点对，返回空矩阵
    if len(all_rows) == 0:
        return SparseTensor(
            row=torch.empty(0, dtype=torch.long, device=device),
            col=torch.empty(0, dtype=torch.long, device=device),
            value=torch.empty(0, dtype=torch.float32, device=device),
            sparse_sizes=(n_nodes, n_nodes)
        )

    # 转换为张量
    all_rows = torch.tensor(all_rows, dtype=torch.long, device=device)
    all_cols = torch.tensor(all_cols, dtype=torch.long, device=device)
    all_vals = torch.tensor(all_vals, dtype=torch.float32, device=device)

    # 创建未缩放的HL矩阵（包含权重但未应用节点度缩放）
    M = SparseTensor(
        row=all_rows,
        col=all_cols,
        value=all_vals,
        sparse_sizes=(n_nodes, n_nodes)
    ).coalesce()

    # 应用度缩放：HL = Dn^{-1/2} * M * Dn^{-1/2}
    rows = M.storage.row()
    cols = M.storage.col()
    vals = M.storage.value()

    scale_factors = (
            r_inv_sqrt_node_in_simplex[rows] *
            r_inv_sqrt_node_in_simplex[cols]
    )
    scaled_vals = vals * scale_factors

    # 创建最终的HL稀疏张量
    sparse_tensor_HL = SparseTensor(
        row=rows,
        col=cols,
        value=scaled_vals,
        sparse_sizes=(n_nodes, n_nodes)
    ).coalesce()

    return sparse_tensor_HL


def get_simplex_HL1_fast(role_weights, node_list, simplices):
    device = role_weights.device
    # 保持输出矩阵尺寸与节点特征一致；不在这里扩充节点集合，避免传播时维度不匹配。
    node_list = list(node_list)
    n_nodes = len(node_list)
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    # 构建入射矩阵索引
    row_indices = []
    col_indices = []
    simplex_node_counts = []  # 保存每个超边包含的有效节点数
    for simplex_idx, simplex_nodes in enumerate(simplices):
        count = 0
        for node in simplex_nodes:
            if node in node_to_index:
                row_indices.append(node_to_index[node])
                col_indices.append(simplex_idx)
                count += 1
        simplex_node_counts.append(count)

    if len(row_indices) == 0:
        return SparseTensor(
            row=torch.empty(0, dtype=torch.long, device=device),
            col=torch.empty(0, dtype=torch.long, device=device),
            value=torch.empty(0, dtype=torch.float32, device=device),
            sparse_sizes=(n_nodes, n_nodes)
        )

    # 将索引张量化
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

    # 计算每个超边的 1/sqrt(|e|)
    simplex_dim = torch.tensor(simplex_node_counts, dtype=torch.float32, device=device)
    r_inv_sqrt_simplex_dim = 1.0 / torch.sqrt(simplex_dim)
    r_inv_sqrt_simplex_dim[torch.isinf(r_inv_sqrt_simplex_dim)] = 0.0  # 超边大小为0的情况置0

    # 构造加权入射矩阵 H_w
    node_role_matrix = torch.sparse_coo_tensor(indices, values, size=(n_nodes, len(simplices))).coalesce()
    H_idx = node_role_matrix.indices()  # [2, nnz]
    H_val = node_role_matrix.values()  # 全为1
    col_idx = H_idx[1]

    # 将 1/sqrt(|e|) 因子应用到入射值上（对称地作用在H的两端）
    H_val = H_val * r_inv_sqrt_simplex_dim[col_idx]

    # 应用超边权重（仅作用在左乘侧）
    weights_for_entries = role_weights.flatten()[col_idx]  # 长度与nnz相同
    H_w_val = H_val * weights_for_entries
    H_w_idx = H_idx

    # 构建转置入射矩阵索引和值 (右乘侧)，包含 1/sqrt(|e|) 因子但不含权重
    H_t_idx = torch.stack([col_idx, H_idx[0]], dim=0)
    H_t_val = H_val

    # 稀疏矩阵乘法计算 M = H_w * H^T
    M_idx, M_val = spspmm(H_w_idx, H_w_val, H_t_idx, H_t_val, n_nodes, len(simplices), n_nodes)

    # 计算度并规避除零
    row, col = M_idx
    deg = degree(col, n_nodes, dtype=M_val.dtype).clamp(min=1e-6)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.0)

    # 归一化 M 得到 HL
    norm_val = deg_inv_sqrt[row] * M_val * deg_inv_sqrt[col]
    HL_idx, HL_val = coalesce(M_idx, norm_val, n_nodes, n_nodes)
    return SparseTensor(row=HL_idx[0], col=HL_idx[1], value=HL_val, sparse_sizes=(n_nodes, n_nodes))


def get_node_HL(train_adj_matrices):

    node_adj = torch.tensor(train_adj_matrices, device=device)

    node_in_simplex = torch.sum(node_adj, dim=1)
    r_inv_sqrt_node_in_simplex = 1.0 / torch.sqrt(node_in_simplex).flatten()
    r_inv_sqrt_node_in_simplex[torch.isinf(r_inv_sqrt_node_in_simplex)] = 0.
    diag_node_in_simplex = torch.diag(r_inv_sqrt_node_in_simplex)

    simplex_dim = torch.sum(node_adj, dim=0)
    r_inv_sqrt_simplex_dim = 1.0 / simplex_dim.flatten()
    r_inv_sqrt_simplex_dim[torch.isinf(r_inv_sqrt_simplex_dim)] = 0.
    diag_simplex_dim = torch.diag(r_inv_sqrt_simplex_dim)


    HL = diag_node_in_simplex @ node_adj @ diag_simplex_dim @ node_adj.T @ diag_node_in_simplex

    num_rows, num_cols = HL.shape
    nonzero_indices = torch.nonzero(HL)
    nonzero_values = HL[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    sparse_tensor = torch.sparse_coo_tensor(nonzero_indices.t(), nonzero_values, size=(num_rows, num_cols))
    sparse_tensor_HL = SparseTensor.from_torch_sparse_coo_tensor(sparse_tensor)

    return sparse_tensor_HL


def get_node_HL_fast(adj_matrix_sparse, num_nodes):
    """
    Optimized function to compute the standard GCN normalization for the node adjacency matrix.

    Args:
        adj_matrix_sparse (torch.sparse_coo_tensor): The sparse adjacency matrix [N, N].
        num_nodes (int): The total number of nodes.
        device: The torch device to use.
    """

    if sp.issparse(adj_matrix_sparse):
        coo = adj_matrix_sparse.tocoo()
        indices = torch.tensor(
            np.vstack((coo.row, coo.col)),
            dtype=torch.long,
            device=device
        )
        values = torch.tensor(coo.data, dtype=torch.float32, device=device)
        node_role_matrix = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
    elif not isinstance(adj_matrix_sparse, torch.Tensor):
        node_role_matrix = torch.tensor(adj_matrix_sparse, dtype=torch.float32, device=device)
    else:
        node_role_matrix = adj_matrix_sparse.to(device)

    # 关键修正：确保张量是稀疏格式
    # 如果它还不是一个稀疏张量，就地将其转换。
    if not node_role_matrix.is_sparse:
        adj_matrix_sparse = node_role_matrix.to_sparse()
    else:
        adj_matrix_sparse = node_role_matrix

    adj_matrix_sparse = adj_matrix_sparse.coalesce().to(device)
    adj_indices, adj_values = adj_matrix_sparse.indices(), adj_matrix_sparse.values()

    # Add self-loops to handle nodes with degree 0 and improve stability
    from torch_geometric.utils import add_self_loops
    adj_indices_sl, adj_values_sl = add_self_loops(adj_indices, adj_values, fill_value=1.0, num_nodes=num_nodes)

    # Calculate D^(-1/2)
    row, col = adj_indices_sl
    deg = degree(col, num_nodes, dtype=adj_values_sl.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    # Apply normalization: D^(-1/2) * A * D^(-1/2)
    norm_values = deg_inv_sqrt[row] * adj_values_sl * deg_inv_sqrt[col]

    HL = SparseTensor(row=row, col=col, value=norm_values, sparse_sizes=(num_nodes, num_nodes))

    return HL

def get_node_HL_fast1(adj_matrix_sparse, num_nodes, device=None, dtype=torch.float32):
    """
    Compute normalized adjacency (GCN-style): HL = D^{-1/2} (A + I) D^{-1/2}
    and return as torch_sparse.SparseTensor.

    Args:
        adj_matrix_sparse: one of
            - scipy.sparse matrix (coo/csr/csc/...)
            - torch.Tensor (dense or sparse coo)
            - numpy.ndarray / list (dense)
        num_nodes (int)
        device: torch.device or None -> auto
        dtype: torch dtype for values
    """

    # 1) 选择设备
    if device is None:
        if isinstance(adj_matrix_sparse, torch.Tensor):
            device = adj_matrix_sparse.device
        else:
            device = torch.device('cpu')

    # 2) 统一转换为 torch 稀疏 COO 张量 A
    A = None
    try:
        import scipy.sparse as sp
        is_scipy = sp.issparse(adj_matrix_sparse)
    except Exception:
        is_scipy = False

    if is_scipy:
        coo = adj_matrix_sparse.tocoo()
        indices = torch.tensor(
            np.vstack([coo.row, coo.col]),
            dtype=torch.long, device=device
        )
        values = torch.tensor(coo.data, dtype=dtype, device=device)
        A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    elif isinstance(adj_matrix_sparse, torch.Tensor):
        if adj_matrix_sparse.is_sparse:
            A = adj_matrix_sparse.coalesce().to(device)
            if A.dtype != dtype:
                A = torch.sparse_coo_tensor(A.indices(), A.values().to(dtype), A.size(), device=device).coalesce()
        else:
            # 稠密 -> 稀疏
            A = adj_matrix_sparse.to(dtype=dtype, device=device)
            A = A.to_sparse().coalesce()

    elif isinstance(adj_matrix_sparse, (np.ndarray, list)):
        dense = torch.as_tensor(adj_matrix_sparse, dtype=dtype, device=device)
        A = dense.to_sparse().coalesce()

    else:
        raise TypeError(f"Unsupported type for adj_matrix_sparse: {type(adj_matrix_sparse)}")

    # 3) 取出边与权重
    edge_index = A.indices()   # [2, E] long
    edge_weight = A.values()   # [E]     dtype

    # 4) 加自环（A + I），对 0 度节点稳定一些
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
    )

    # 5) 计算 D^{-1/2}
    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes, dtype=edge_weight.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0)

    # 6) 归一化
    norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # 7) 构建 torch_sparse 的 SparseTensor 返回
    HL = SparseTensor(row=row, col=col, value=norm_values,
                      sparse_sizes=(num_nodes, num_nodes))
    return HL

def ensure_sparse_tensor(A, device=None, dtype=torch.float32):
    """
    将输入 A（可能是 SciPy 稀疏、PyTorch 稀疏/稠密、NumPy）统一转成 torch_sparse.SparseTensor。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 已经是 SparseTensor
    if isinstance(A, SparseTensor):
        return A.to(device)

    # 2) SciPy 稀疏
    try:
        import scipy.sparse as ssp
        if ssp.issparse(A):
            coo = A.tocoo()
            row = torch.from_numpy(coo.row.astype(np.int64)).to(device)
            col = torch.from_numpy(coo.col.astype(np.int64)).to(device)
            val = torch.from_numpy(coo.data).to(device=device, dtype=dtype)
            return SparseTensor(row=row, col=col, value=val, sparse_sizes=coo.shape)
    except Exception:
        pass

    # 3) PyTorch 张量（稀疏/稠密）
    if isinstance(A, torch.Tensor):
        if A.layout == torch.sparse_coo:
            A = A.coalesce().to(device)
            row, col = A.indices()
            val = A.values().to(dtype)
            return SparseTensor(row=row, col=col, value=val, sparse_sizes=A.shape)
        elif A.layout == torch.sparse_csr:
            A = A.to_sparse_coo().coalesce().to(device)
            row, col = A.indices()
            val = A.values().to(dtype)
            return SparseTensor(row=row, col=col, value=val, sparse_sizes=A.shape)
        else:
            A = A.to(device=device, dtype=dtype).to_sparse_coo().coalesce()
            row, col = A.indices()
            val = A.values()
            return SparseTensor(row=row, col=col, value=val, sparse_sizes=A.shape)

    # 4) 其它（NumPy/可转数组）
    arr = np.asarray(A)
    dense = torch.from_numpy(arr).to(device=device, dtype=dtype)
    A = dense.to_sparse_coo().coalesce()
    row, col = A.indices()
    val = A.values()
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=A.shape)
#
# def LoadGraph(input_dim, TRAIN_Graph, directed_train=False, feature_type='1', use_cuda= True):
#     path_to_train = os.path.join( f"..\\data\\realworld\\{TRAIN_Graph}"+'.txt')
#     g = igraph.Graph().Read_Edgelist(
#         path_to_train, directed=directed_train)
#     dg, feature_dim = get_rev_dgl( network=TRAIN_Graph, graph=g, feature_type=feature_type, feature_dim=input_dim, use_cuda=use_cuda)
#
#
#
#     return g, dg, feature_dim

