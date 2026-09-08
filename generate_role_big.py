import torch
from sklearn.preprocessing import RobustScaler
from torch.nn.utils.rnn import pad_sequence
from graphrole import RecursiveFeatureExtractor, RoleExtractor
from incidence_matrix import get_G
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from gen_HoHLaplacian import *
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import warnings


def generate_role_matrix(G, n_Role):
    """
    改进版本的角色矩阵生成函数，支持大规模图处理
    """
    # 自动选择计算方法（大图用快速方法，小图用原方法）
    if len(G.nodes) > 100000:
        return generate_role_matrix_fast(G, n_Role)
    else:
        return generate_role_matrix_original(G, n_Role)


def generate_role_matrix_fast(G, n_Role):
    """增强版特征提取方法，支持1M-10M节点"""
    # 1. 计算多维度结构特征
    nodes = sorted(G.nodes)
    num_nodes = len(nodes)

    # 1. 初始化特征矩阵
    feature_matrix = np.zeros((num_nodes, 10))

    # 2. 计算基础高效特征
    degrees = dict(G.degree())
    log_degrees = {node: np.log1p(deg) for node, deg in degrees.items()}

    # 3. 并行计算邻居统计特征
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 忽略空除警告

        # 邻居平均度
        avg_neighbor_deg = nx.average_neighbor_degree(G)

        # 邻居度标准差
        neighbor_deg_std = {}
        for node in nodes:
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor_degs = [degrees[neigh] for neigh in neighbors]
                neighbor_deg_std[node] = np.std(neighbor_degs)
            else:
                neighbor_deg_std[node] = 0

    # 4. 选择性计算高级特征（基于图大小）
    clustering_coeffs = {}
    core_numbers = {}
    pagerank = {}

    if num_nodes <= 500000:  # 中等图计算聚类系数
        clustering_coeffs = nx.clustering(G)

    if num_nodes <= 200000:  # 较小图计算k-core和PageRank
        core_numbers = nx.core_number(G)
        pagerank = nx.pagerank(G, max_iter=20)

    # 5. 构建特征矩阵
    for i, node in enumerate(nodes):
        feature_matrix[i, 0] = degrees.get(node, 0)  # 节点度
        feature_matrix[i, 1] = log_degrees.get(node, 0)  # 对数变换度
        feature_matrix[i, 2] = avg_neighbor_deg.get(node, 0)  # 邻居平均度
        feature_matrix[i, 3] = neighbor_deg_std.get(node, 0)  # 邻居度标准差

        # 二阶邻居统计
        neighbors = list(G.neighbors(node))
        if neighbors:
            second_neighbors = set()
            for neighbor in neighbors:
                second_neighbors |= set(G.neighbors(neighbor))
            second_neighbors -= set([node] + neighbors)
            feature_matrix[i, 4] = len(second_neighbors)  # 二阶邻居数
        else:
            feature_matrix[i, 4] = 0

        # 选择性特征
        feature_matrix[i, 5] = clustering_coeffs.get(node, 0)  # 聚类系数
        feature_matrix[i, 6] = core_numbers.get(node, 0)  # k-core数
        feature_matrix[i, 7] = pagerank.get(node, 0)  # PageRank值

        # 额外的鲁棒特征
        feature_matrix[i, 8] = len(neighbors) * log_degrees.get(node, 0)  # 度组合特征
        feature_matrix[i, 9] = 1 if degrees.get(node, 0) > np.median(list(degrees.values())) else 0  # 高中介节点

    # 6. 特征缩放（使用鲁棒缩放处理异常值）
    scaler = RobustScaler()
    norm_features = scaler.fit_transform(feature_matrix)

    # 7. K-Means聚类
    kmeans = MiniBatchKMeans(
        n_clusters=n_Role,
        batch_size=min(10000, num_nodes // 10),
        max_iter=100,
        n_init=5,
        random_state=42
    )
    labels = kmeans.fit_predict(norm_features)

    # 8. 构建角色矩阵
    role_list = [f"role{i}" for i in range(n_Role)]
    node_role_matrix = np.zeros((num_nodes, n_Role))

    for i, label in enumerate(labels):
        node_role_matrix[i, label] = 1

    # 9. 角色特征矩阵
    role_feature = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)

    return node_role_matrix, role_feature

def generate_role_matrix_original(G, n_Role):
    """
    原始graphrole方法（用于小型图）
    """

    # 特征提取
    feature_extractor = RecursiveFeatureExtractor(G)
    features = feature_extractor.extract_features()

    # 角色提取
    role_extractor = RoleExtractor(n_roles=n_Role)
    role_extractor.extract_role_factors(features)

    # 节点角色映射
    node_roles = role_extractor.roles

    # 角色列表
    role_list = role_extractor.node_role_factor.columns.tolist()

    # 角色特征
    role_feature = role_extractor.role_feature_factor.values

    # 构建关联矩阵
    num_nodes = len(node_roles)
    num_roles = len(role_list)
    node_role_matrix = np.zeros((num_nodes, num_roles))

    for node, role in node_roles.items():
        role_index = role_list.index(role)
        node_role_matrix[node, role_index] = 1

    return node_role_matrix, torch.tensor(role_feature, dtype=torch.float)

def hyper_list(node_roles):
    # 创建一个空的超边索引张量
    hyperedge_index = torch.zeros((2, len(node_roles)), dtype=torch.long)

    # 填充超边索引张量
    for i, role in enumerate(node_roles.values()):
        # 解析角色字符串中的数字
        role_number = int(role.split('_')[1])
        hyperedge_index[0, i] = i
        hyperedge_index[1, i] = role_number

    return hyperedge_index



def generate_role_node_list(node_roles, role_list):
    matrix = defaultdict(list)

    for node, role in node_roles.items():
        matrix[role].append(node)

    matrix.update({role: [] for role in role_list if role not in matrix})  # 确保每个角色都有矩阵，即使是空矩阵


    desired_order = sorted(set(role_list), key=lambda x: int(x.split('_')[1]))

    # 转换为按顺序存储的列表
    role_node_idx = [matrix[role] for role in desired_order]

    max_role_len = max(role_node_idx, key=len)

    return role_node_idx, max_role_len


def generate_role_graph(role_H):
    role_G = nx.Graph()

    # 添加节点
    num_roles = role_H.shape[0]
    role_G.add_nodes_from(range(num_roles))

    # 遍历邻接矩阵，添加边
    for i in range(num_roles):
        for j in range(num_roles):
            if role_H[i][j] == 1:
                role_G.add_edge(i, j)

    return role_G

def extra_role(G, n_Role):

    feature_extractor = RecursiveFeatureExtractor(G)
    features = feature_extractor.extract_features()

    # n_roles = 10
    role_extractor = RoleExtractor(n_roles=n_Role)  # 这是设置角色数量时
    # role_extractor = RoleExtractor()  # 自然生成角色数量
    role_extractor.extract_role_factors(features)

    node_roles = role_extractor.roles

    # 获取roles的列表
    role_list = role_extractor.node_role_factor.columns.tolist()

    # 获取roles的特征
    role_feature = role_extractor.role_feature_factor

    return node_roles, role_list, role_feature

def extra_indic(node_roles, role_list):

    num_roles = len(node_roles)  # 节点数量
    num_nodes = len(role_list)  # 角色数量

    # 初始化关联矩阵
    node_role_matrix = np.zeros([num_roles, num_nodes])

    # 填充关联矩阵
    for node, role in node_roles.items():
        if role in role_list:
            role_index = role_list.index(role)
            node_role_matrix[node][role_index] = 1

    return node_role_matrix

def extra_feature(role_node_idx, node_feature):
    role_node_idx = pad_sequence([torch.tensor(sublist) for sublist in role_node_idx],
                                 batch_first=True, padding_value=-1).to(device)

    role_node_feature = torch.zeros(role_node_idx.size(0), role_node_idx.size(1), node_feature.size(1),
                                  device=device, dtype=torch.float64).to(device)

    # 将节点特征赋予给每个角色对应的节点，并处理编号为-1的节点特征表示为0
    for i in range(role_node_idx.size(0)):
        for j in range(role_node_idx.size(1)):
            node_index = role_node_idx[i][j].item()
            if node_index != -1:
                role_node_feature[i][j] = node_feature[node_index]

    return role_node_feature, role_node_idx


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = torch.tensor(H).float()
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = torch.ones(n_edge)
    # the degree of the node
    DV = torch.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = torch.sum(H, axis=0)

    invDE = torch.diag(DE.pow(-1))
    invDE[torch.isinf(invDE)] = 0.  # 避免前面有0值
    DV2 = torch.diag(DV.pow(-0.5))
    DV2[torch.isinf(DV2)] = 0.  # 避免前面有0值
    W = torch.diag(W)
    HT = H.t()

    if variable_weight:
        DV2_H = DV2 @ H
        invDE_HT_DV2 = invDE @ HT @ DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 @ H @ W @ invDE @ HT @ DV2
        return G