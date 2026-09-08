import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
import os
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch
import math
import itertools
import torch.sparse as sps
from gen_HoHLaplacian import *
from torch_sparse import SparseTensor, fill_diag, matmul, mul
import itertools
from itertools import combinations
from scipy.sparse import lil_matrix, triu



def gen_coo_simplex_A(g):
    gen_HL = creat_L_SparseTensor(g, maxCliqueSize=3)

    return gen_HL

def get_simplex_idx(G_single):
    edge_to_idx = {edge: i for i, edge in enumerate(G_single.edges)}

    faces, tetrahedra, pentachora = get_higher_order_simplices(G_single)
    # faces = get_faces(G_single)

    return edge_to_idx, faces, tetrahedra, pentachora

def get_simplex_ft(node_feature, edge, faces, tetrahedra, pentachora):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_matrix = torch.tensor(np.array([[key[0], key[1]] for key in edge.keys()]), dtype=torch.long).to(device)  # 获得边的矩阵
    faces_matrix = torch.tensor(np.array(faces), dtype=torch.long).to(device)  # 获得三角形的矩阵
    tetrahedra_matrix = torch.tensor(np.array(tetrahedra), dtype=torch.long).to(device)
    pentachora_matrix = torch.tensor(np.array(pentachora), dtype=torch.long).to(device)

    edge_num = len(edge_matrix)
    faces_num = len(faces_matrix)
    tetra_num = len(tetrahedra)
    penta_num = len(pentachora)

    d = node_feature.shape[1]

    edge_node_feature = node_feature[edge_matrix.view(-1)].view(edge_num, 2, d)
    faces_node_feature = node_feature[faces_matrix.view(-1)].view(faces_num, 3, d)
    tetrahedra_node_feature = node_feature[tetrahedra_matrix.view(-1)].view(tetra_num, 4, d)
    pentachora_node_feature = node_feature[pentachora_matrix.view(-1)].view(penta_num, 5, d)

    return edge_node_feature, faces_node_feature, tetrahedra_node_feature, pentachora_node_feature


def compute_simple_complex_matrix(g):

    simplex = get_simplex(g, max_petal_dim=3)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}
    node_to_edge, node_to_triangle, edge_to_triangle = incidence_matrices(g, sorted(g.nodes), sorted(g.edges), get_faces(g), edge_to_idx)

    return node_to_edge, node_to_triangle, edge_to_triangle

def normalize_sparse_simple_complex_symmetric(H_matrix0_0, H_matrix0_1, H_matrix0_2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H_matrix0_0 = torch.tensor(H_matrix0_0, dtype=torch.float32, device=device)
    H_matrix0_1 = torch.tensor(H_matrix0_1, dtype=torch.float32, device=device)
    H_matrix0_2 = torch.tensor(H_matrix0_2, dtype=torch.float32, device=device)

    #  计算A_hat
    #  0-0 单纯复形
    Z_00 = H_matrix0_0.sum(1)

    r_inv_sqrt_Z_00 = 1.0 / torch.sqrt(Z_00).flatten()
    r_inv_sqrt_Z_00[torch.isinf(r_inv_sqrt_Z_00)] = 0.
    Re_Z_00 = torch.diag(r_inv_sqrt_Z_00)

    A_Hat_00 = Re_Z_00 @ H_matrix0_0 @ H_matrix0_0.T @ Re_Z_00  # 式2

    #  0-1 单纯复形
    Z_01 = H_matrix0_1.sum(1)

    r_inv_sqrt_Z_01 = 1.0 / torch.sqrt(Z_01).flatten()
    r_inv_sqrt_Z_01[torch.isinf(r_inv_sqrt_Z_01)] = 0.
    Re_Z_01 = torch.diag(r_inv_sqrt_Z_01)

    A_Hat_01 = Re_Z_01 @ H_matrix0_1 @ H_matrix0_1.T @ Re_Z_01  # 式2

    #  0-2 单纯复形
    Z_02 = H_matrix0_2.sum(1)

    r_inv_sqrt_Z_02 = 1.0 / torch.sqrt(Z_02).flatten()
    r_inv_sqrt_Z_02[torch.isinf(r_inv_sqrt_Z_02)] = 0.
    Re_Z_02 = torch.diag(r_inv_sqrt_Z_02)

    A_Hat_02 = Re_Z_02 @ H_matrix0_2 @ H_matrix0_2.T @ Re_Z_02  # 式2

    A_adj = H_matrix0_1 @ H_matrix0_1.T + H_matrix0_2 @ H_matrix0_2.T

    return A_Hat_01, A_Hat_02, A_adj


def normalize_sparse_simple_complex_symmetric1(H_matrix0_1, H_matrix0_2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H_matrix0_1 = torch.tensor(H_matrix0_1, dtype=torch.float32, device=device)
    H_matrix0_2 = torch.tensor(H_matrix0_2, dtype=torch.float32, device=device)

    #  计算A_hat
    #  0-1 单纯复形
    Z_01 = H_matrix0_1.sum(1)
    Z_10 = H_matrix0_1.sum(0)

    r_inv_sqrt_Z_01 = 1.0 / torch.sqrt(Z_01).flatten()
    r_inv_sqrt_Z_01[torch.isinf(r_inv_sqrt_Z_01)] = 0.
    Re_Z_01 = torch.diag(r_inv_sqrt_Z_01)

    r_inv_sqrt_Z_10 = 1.0 / torch.sqrt(Z_10).flatten()
    r_inv_sqrt_Z_10[torch.isinf(r_inv_sqrt_Z_10)] = 0.
    Re_Z_10 = torch.diag(r_inv_sqrt_Z_10)

    A_Hat_01 = Re_Z_01 @ H_matrix0_1 @ Re_Z_10  # 式2

    #  0-2 单纯复形
    Z_02 = H_matrix0_2.sum(1)
    Z_20 = H_matrix0_2.sum(0)

    r_inv_sqrt_Z_02 = 1.0 / torch.sqrt(Z_02).flatten()
    r_inv_sqrt_Z_02[torch.isinf(r_inv_sqrt_Z_02)] = 0.
    Re_Z_02 = torch.diag(r_inv_sqrt_Z_02)

    r_inv_sqrt_Z_20 = 1.0 / torch.sqrt(Z_20).flatten()
    r_inv_sqrt_Z_20[torch.isinf(r_inv_sqrt_Z_20)] = 0.
    Re_Z_20 = torch.diag(r_inv_sqrt_Z_20)

    A_Hat_02 = Re_Z_02 @ H_matrix0_2 @ Re_Z_20  # 式2

    #  计算卷积
    #  设定权重
    # Omega_0 = torch.eye(Re_Z_01.shape[0], dtype=torch.float32, device=device)  # 节点权重
    Omega_1 = torch.eye(Re_Z_10.shape[0], dtype=torch.float32, device=device)  # 边权重
    Omega_2 = torch.eye(Re_Z_20.shape[0], dtype=torch.float32, device=device)  # 三角形权重

    #  0-单纯复形
    Norm_A_0_01 = A_Hat_01 @ Omega_1 @ A_Hat_01.T  # 式4
    Norm_A_0_02 = A_Hat_02 @ Omega_2 @ A_Hat_02.T  # 式4
    Norm_A_0 = Norm_A_0_01 + Norm_A_0_02
    Norm_A_0_10 = A_Hat_01 @ Omega_1 @ Re_Z_10  # 式6
    Norm_A_0_20 = A_Hat_02 @ Omega_2 @ Re_Z_20   # 式9

    return Norm_A_0, Norm_A_0_10, Norm_A_0_20

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


def get_triangles_sparse(G):
    """使用稀疏矩阵乘法计算三角形，确保没有重复节点"""
    # 创建节点映射
    nodes = sorted(G.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    # 构建稀疏邻接矩阵（排除自环）
    A = lil_matrix((n, n), dtype=np.int8)
    for u, v in G.edges():
        # 只添加不同节点之间的边
        if u != v:
            idx_u = node_index[u]
            idx_v = node_index[v]
            A[idx_u, idx_v] = 1
            A[idx_v, idx_u] = 1

    # 转换为CSR格式以提高性能
    A_csr = A.tocsr()

    # 计算A²：识别共享两个邻居的节点对
    A_squared = A_csr @ A_csr

    # 提取三角形：只考虑三个不同节点形成的闭合三角形
    triangles = []
    for i in range(n):
        # 获取节点i的邻居
        neighbors_i = set(A_csr[i].nonzero()[1])
        # 只考虑索引大于i的邻居（避免重复计数）
        for j in neighbors_i:
            if j <= i:
                continue
            # 获取节点j的邻居
            neighbors_j = set(A_csr[j].nonzero()[1])
            # 找到共同邻居中索引大于j的节点
            common = neighbors_i & neighbors_j
            for k in common:
                if k > j:
                    # 检查(i,k)和(j,k)是否真的有边（避免误报）
                    if A_csr[i, k] and A_csr[j, k]:
                        triangles.append((i, j, k))

    # 转换回原始节点ID
    index_node = {idx: node for node, idx in node_index.items()}
    return [tuple(sorted((index_node[i], index_node[j], index_node[k]))) for i, j, k in triangles]


def get_triangles_optimized(G):
    """优化后的三角形计数算法，时间复杂度O(|E| * <d>)"""
    # 构建邻居索引
    neighbors = {}
    for node in G.nodes():
        neighbors[node] = set(G.neighbors(node))

    triangles = []
    visited = set()

    # 按度数排序节点（度数小的优先处理）
    nodes_sorted = sorted(G.nodes(), key=lambda x: G.degree(x))

    for u in nodes_sorted:
        # 只考虑度数大于u的邻居，避免重复计算
        for v in [n for n in G.neighbors(u) if G.degree(n) > G.degree(u)]:
            common = neighbors[u] & neighbors[v]
            for w in common:
                # 避免重复：只记录排序后顺序一致的三角形
                if u < v < w:
                    triangles.append((u, v, w))

    return triangles

def get_higher_order_simplices(G):
    adj_matrix = nx.to_numpy_array(G)
    nodes = list(G.nodes())
    n = len(nodes)

    faces = []
    tetrahedra = []
    pentachora = []

    # 预计算每个节点的邻居
    neighbors = {i: set(np.where(adj_matrix[i] > 0)[0]) for i in range(n)}

    # 寻找三角形（面）
    for i in range(n):
        for j in neighbors[i]:
            if j > i:
                common = neighbors[i] & neighbors[j]
                for k in common:
                    if k > j:
                        faces.append(tuple(sorted([nodes[i], nodes[j], nodes[k]])))

    # 寻找四面体
    for face in faces:
        i, j, k = [nodes.index(node) for node in face]
        common = neighbors[i] & neighbors[j] & neighbors[k]
        for l in common:
            if l > k:
                tetrahedra.append(tuple(sorted([nodes[i], nodes[j], nodes[k], nodes[l]])))

    # 寻找五面体（如果需要的话）
    if len(tetrahedra) > 0:
        for tetra in tetrahedra:
            i, j, k, l = [nodes.index(node) for node in tetra]
            common = neighbors[i] & neighbors[j] & neighbors[k] & neighbors[l]
            for m in common:
                if m > l:
                    pentachora.append(tuple(sorted([nodes[i], nodes[j], nodes[k], nodes[l], nodes[m]])))

    return list(set(faces)), list(set(tetrahedra)), list(set(pentachora))
def get_faces(G):
    """
    Returns a list of the faces in an undirected graph
    """

    # 获取图的邻接矩阵
    adj_matrix = nx.to_numpy_array(G)
    nodes = list(G.nodes())  # 获取节点列表

    # 初始化一个空的列表来存储面
    faces = []

    # 遍历图中的所有节点对
    for i in range(adj_matrix.shape[0]):
        for j in range(i+1, adj_matrix.shape[1]):
            # 如果两个节点之间有边
            if adj_matrix[i, j] == 1:
                # 找出与这两个节点都相连的所有其他节点
                common_neighbors = np.where((adj_matrix[i, :] == 1) & (adj_matrix[j, :] == 1))[0]
                # 将这些节点与当前的节点对组合成面
                faces.extend([tuple(sorted([nodes[i], nodes[j], nodes[k]])) for k in common_neighbors])

    return list(set(faces))   # 使用 set 来去除重复的面

def get_tetrahedra(G):
    """Returns a list of tetrahedra in an undirected graph"""
    adj_matrix = nx.to_numpy_array(G)
    nodes = list(G.nodes())
    tetrahedra = []

    for i, j, k, l in combinations(range(len(nodes)), 4):
        if (adj_matrix[i, j] and adj_matrix[i, k] and adj_matrix[i, l] and
            adj_matrix[j, k] and adj_matrix[j, l] and adj_matrix[k, l]):
            tetrahedra.append(tuple(sorted([nodes[i], nodes[j], nodes[k], nodes[l]])))

    return list(set(tetrahedra))

def get_pentachora(G):
    """Returns a list of pentachora (4-simplices) in an undirected graph"""
    adj_matrix = nx.to_numpy_array(G)
    nodes = list(G.nodes())
    pentachora = []

    for a, b, c, d, e in combinations(range(len(nodes)), 5):
        if all(adj_matrix[x, y] for x, y in combinations([a, b, c, d, e], 2)):
            pentachora.append(tuple(sorted([nodes[a], nodes[b], nodes[c], nodes[d], nodes[e]])))

    return list(set(pentachora))

def incidence_matrices(G, V, E, faces, edge_to_idx):
    """
    Returns incidence matrices B1 and B2

    :param G: NetworkX DiGraph
    :param V: list of nodes
    :param E: list of edges
    :param faces: list of faces in G

    Returns edge_inci (|V| x |E|) and triangle_inci (|E| x |faces|)
    """
    node_to_edge = np.array(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=False).todense())  # 节点到边的的关联矩阵

    node_to_triangle = np.zeros([len(V), len(faces)])  # 节点到面的关联矩阵
    for f_idx, face in enumerate(faces):  # face is sorted
        for node_index in face:
            # 在关联矩阵中标记连接的节点
            node_to_triangle[node_index, f_idx] = 1

    edge_to_triangle = np.zeros([len(E), len(faces)])  # 边到面的关联矩阵
    for face_idx, face in enumerate(faces):
        for edge in itertools.combinations(face, 2):  # 生成面中的所有边的组合
            if edge in E:
                edge_idx = E.index(edge)  # 获取边在1阶单纯复形中的索引
                edge_to_triangle[edge_idx, face_idx] = 1
            else:
                reversed_edge = (edge[1], edge[0])
                edge_idx = E.index(reversed_edge)  # 获取边在1阶单纯复形中的索引
                edge_to_triangle[edge_idx, face_idx] = 1

    return node_to_edge, node_to_triangle, edge_to_triangle

def compute_simple_complex_feature(H_matrix0_1, H_matrix0_2, node_feature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_feature = torch.tensor(node_feature, dtype=torch.float32).to(device)

    edge_norm = torch.tensor(H_matrix0_1 / np.sum(H_matrix0_1, axis=0), dtype=torch.float32).to(device)
    edge_feature = edge_norm.T @ node_feature

    triangle_norm = torch.tensor(H_matrix0_2 / np.sum(H_matrix0_2, axis=0), dtype=torch.float32).to(device)
    triangle_feature = triangle_norm.T @ node_feature

    return edge_feature, triangle_feature


"""以下是原代码，计算内存过大
def normalize_sparse_simple_complex_symmetric(network):

    H_matrix0_1, H_matrix0_2 = compute_hodge_matrix(network)


    #  0-1 单纯复形
    Z_01 = np.array(H_matrix0_1.sum(1))
    Z_10 = np.array(H_matrix0_1.sum(0))

    r_inv_sqrt_Z_01 = np.power(Z_01, -0.5).flatten()
    r_inv_sqrt_Z_01[np.isinf(r_inv_sqrt_Z_01)] = 0.
    Re_Z_01 = sp.diags(r_inv_sqrt_Z_01)

    r_inv_sqrt_Z_10 = np.power(Z_10, -0.5).flatten()
    r_inv_sqrt_Z_10[np.isinf(r_inv_sqrt_Z_10)] = 0.
    Re_Z_10 = sp.diags(r_inv_sqrt_Z_10)

    A_Hat_01 = Re_Z_01.dot(H_matrix0_1).dot(Re_Z_10)  # 式2

    Omega_01 = sp.eye(Re_Z_10.shape[0])
    Norm_A_01 = A_Hat_01.dot(Omega_01).dot(A_Hat_01.transpose())  # 式4

    Norm_A_10 = A_Hat_01.dot(Omega_01).dot(Re_Z_10)  # 式6

    #  0-2 单纯复形
    Z_02 = np.array(H_matrix0_2.sum(1))
    Z_20 = np.array(H_matrix0_2.sum(0))

    r_inv_sqrt_Z_02 = np.power(Z_02, -0.5).flatten()
    r_inv_sqrt_Z_02[np.isinf(r_inv_sqrt_Z_02)] = 0.
    Re_Z_02 = sp.diags(r_inv_sqrt_Z_02)

    r_inv_sqrt_Z_20 = np.power(Z_20, -0.5).flatten()
    r_inv_sqrt_Z_20[np.isinf(r_inv_sqrt_Z_20)] = 0.
    Re_Z_20 = sp.diags(r_inv_sqrt_Z_20)

    A_Hat_02 = Re_Z_02.dot(H_matrix0_2).dot(Re_Z_20)  # 式2

    Omega_02 = sp.eye(Re_Z_20.shape[0])
    Norm_A_02 = A_Hat_02.dot(Omega_02).dot(A_Hat_02.transpose())  # 式7

    Norm_A_20 = A_Hat_02.dot(Omega_02).dot(Re_Z_20)  # 式9


    return Norm_A_01, Norm_A_10, Norm_A_02, Norm_A_20
"""