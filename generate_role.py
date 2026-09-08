import torch
from torch.nn.utils.rnn import pad_sequence
from graphrole import RecursiveFeatureExtractor, RoleExtractor
from incidence_matrix import get_G
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from gen_HoHLaplacian import *
from collections import defaultdict


def generate_role_matrix(G, n_Role):

    node_roles, role_list, role_feature = extra_role(G, n_Role)
    role_feature = torch.tensor(role_feature.values)

    node_role_matrix = extra_indic(node_roles, role_list)  # 生成节点角色关联矩阵

    # node_role_norm_matrix = _generate_G_from_H(node_role_matrix)
    # role_node_idx, max_role_len = generate_role_node_list(node_roles, role_list)

    # hyperedge_index = hyper_list(node_roles)
    ####以下代码是构建新的角色图，但是如果边过多，计算量过大，后续再考虑
    # role_H = np.dot(node_role_matrix, node_role_matrix.T)  # 构建邻接矩阵
    # role_H = np.where(role_H > 0, 1, role_H)
    # role_G = generate_role_graph(role_H)  # 构建角色图

    return node_role_matrix, role_feature

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