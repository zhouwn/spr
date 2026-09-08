# ======= incidence_matrix.py (替换整个文件) =======

import networkx as nx
import numpy as np
import torch


def _build_order_and_neighbors(G):
    """
    为前向枚举建立：
    - order_rank: 节点 -> 排名（度，id）升序
    - Nf[u]: 仅包含比 u 排名更大的邻居（forward 邻居）
    """
    # 保证节点是稳定的可比较类型（int）；如不是，可以 map 一次
    nodes = list(G.nodes())
    degs = dict(G.degree())
    # 排序：先度，再 id
    nodes_sorted = sorted(nodes, key=lambda x: (degs[x], x))
    order_rank = {u: i for i, u in enumerate(nodes_sorted)}

    # forward 邻接
    Nf = {}
    for u in nodes:
        ru = order_rank[u]
        Nf[u] = {v for v in G.neighbors(u) if order_rank[v] > ru}

    return order_rank, Nf


def _triangles_forward(G, order_rank, Nf, max_triangles=None):
    """
    前向三角形枚举：u<v<w（按 order_rank）
    返回：list[tuple(u,v,w)]
    """
    tri = []
    for u in G.nodes():
        Nu = Nf[u]
        if not Nu:
            continue
        for v in Nu:
            # 交集只在 forward 邻接中求，保证 w > v
            common = Nu & Nf[v]
            if common:
                for w in common:
                    tri.append((u, v, w))
                    if (max_triangles is not None) and (len(tri) >= max_triangles):
                        return tri
    return tri


def _tetrahedra_from_triangles(triangles, Nf, max_tetra=None):
    """
    在已得三角形基础上枚举四面体：u<v<w<x
    """
    tetra = []
    for (u, v, w) in triangles:
        common = Nf[u] & Nf[v] & Nf[w]   # x > w
        for x in common:
            tetra.append((u, v, w, x))
            if (max_tetra is not None) and (len(tetra) >= max_tetra):
                return tetra
    return tetra


def _penta_from_tetra(tetrahedra, Nf, max_penta=None):
    """
    在已得四面体基础上枚举五面体：u<v<w<x<y
    """
    penta = []
    for (u, v, w, x) in tetrahedra:
        common = Nf[u] & Nf[v] & Nf[w] & Nf[x]  # y > x
        for y in common:
            penta.append((u, v, w, x, y))
            if (max_penta is not None) and (len(penta) >= max_penta):
                return penta
    return penta


def get_higher_order_simplices(
    G,
    enable_tetra_penta=True,
    max_triangles=20_000_000,   # 保护上限：按你机器内存酌情调整
    max_tetra=2_000_0000,
    max_penta=20000_0000,
    hi_order_enable_nmax=900_000  # 节点数超过该阈值时，自动禁用 tetra/penta
):
    """
    仅用邻接表的内存安全版本。默认在超大图上跳过四面体/五面体。
    返回：faces(list[3]), tetrahedra(list[4]), pentachora(list[5])
    """
    n = G.number_of_nodes()
    order_rank, Nf = _build_order_and_neighbors(G)

    # 1) triangles
    faces = _triangles_forward(G, order_rank, Nf, max_triangles)

    # 2) tetra/penta：视规模与开关决定
    tetrahedra, pentachora = [], []
    if enable_tetra_penta and (n <= hi_order_enable_nmax):
        if faces:
            tetrahedra = _tetrahedra_from_triangles(faces, Nf, max_tetra)
            if tetrahedra:
                pentachora = _penta_from_tetra(tetrahedra, Nf, max_penta)
    else:
        # 大图上直接跳过，以免组合爆炸
        tetrahedra = []
        pentachora = []

    # 去重（理论上 forward 算法已无重复；这里稳妥起见）
    faces = list(set(tuple(sorted(t)) for t in faces))
    tetrahedra = list(set(tuple(sorted(t)) for t in tetrahedra))
    pentachora = list(set(tuple(sorted(t)) for t in pentachora))

    return faces, tetrahedra, pentachora


def get_simplex_idx(G_single):
    """
    返回：
      edge_to_idx: {(u,v)（有序） -> idx}
      faces / tetrahedra / pentachora : list of tuples（节点 id）
    """
    # 规范化边的有序表示（小的在前），避免 (u,v) / (v,u) 重复
    edge_to_idx = {}
    i = 0
    for u, v in G_single.edges():
        a, b = (u, v) if u < v else (v, u)
        if (a, b) not in edge_to_idx:
            edge_to_idx[(a, b)] = i
            i += 1

    # 高阶单纯形（内存安全版本）
    faces, tetrahedra, pentachora = get_higher_order_simplices(G_single)

    return edge_to_idx, faces, tetrahedra, pentachora


def get_simplex_ft(node_feature, edge, faces, tetrahedra, pentachora):
    """
    将节点特征切片为边/面/四面体/五面体的节点特征张量。
    输入:
      node_feature: [N, d] (CPU 或 CUDA 均可)
      edge: dict{(u,v)->idx}，键是节点 id（必须能直接索引 node_feature）
      faces/tetra/penta: list[tuple]，元素是节点 id
    输出:
      edge_node_feature: [E, 2, d]
      faces_node_feature: [F, 3, d]（若 F=0，返回空张量）
      tetrahedra_node_feature: [T, 4, d]
      pentachora_node_feature: [P, 5, d]
    """
    device = node_feature.device
    d = node_feature.shape[1]

    # 边
    if len(edge) > 0:
        edge_keys = list(edge.keys())
        edge_matrix = torch.tensor(np.array(edge_keys), dtype=torch.long, device=device)  # [E,2]
        edge_node_feature = node_feature[edge_matrix.reshape(-1)].reshape(len(edge_keys), 2, d)
    else:
        edge_node_feature = torch.empty((0, 2, d), dtype=node_feature.dtype, device=device)

    # 三角形
    if len(faces) > 0:
        faces_matrix = torch.tensor(np.array(faces), dtype=torch.long, device=device)     # [F,3]
        faces_node_feature = node_feature[faces_matrix.reshape(-1)].reshape(len(faces), 3, d)
    else:
        faces_node_feature = torch.empty((0, 3, d), dtype=node_feature.dtype, device=device)

    # 四面体
    if len(tetrahedra) > 0:
        tetrahedra_matrix = torch.tensor(np.array(tetrahedra), dtype=torch.long, device=device)  # [T,4]
        tetrahedra_node_feature = node_feature[tetrahedra_matrix.reshape(-1)].reshape(len(tetrahedra), 4, d)
    else:
        tetrahedra_node_feature = torch.empty((0, 4, d), dtype=node_feature.dtype, device=device)

    # 五面体
    if len(pentachora) > 0:
        pentachora_matrix = torch.tensor(np.array(pentachora), dtype=torch.long, device=device)  # [P,5]
        pentachora_node_feature = node_feature[pentachora_matrix.reshape(-1)].reshape(len(pentachora), 5, d)
    else:
        pentachora_node_feature = torch.empty((0, 5, d), dtype=node_feature.dtype, device=device)

    return edge_node_feature, faces_node_feature, tetrahedra_node_feature, pentachora_node_feature
# ======= 以上替换结束 =======
