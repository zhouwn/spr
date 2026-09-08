import os
import igraph
import networkx as nx
import numpy as np
import random
from gensim.models import Word2Vec


def calculate_pagerank(G):
    # 计算图G的PageRank
    pagerank_scores = nx.pagerank(G)
    return pagerank_scores

def calculate_degree(G):
    # 计算图G的degree
    degree_scores = nx.degree_centrality(G)
    return degree_scores

def calculate_collective_influence(G, depth=2):
    # 计算图G的Collective Influence
    ci_scores = {}
    for node in G.nodes():
        # 计算节点的邻居数量（度数减去1，因为要排除节点自身）
        degree = G.degree(node) - 1
        # 计算节点的邻居的邻居数量的总和
        ball_size = sum(G.degree(n) for n in nx.single_source_shortest_path_length(G, node, cutoff=depth).keys()) - degree
        # 计算CI值
        ci_scores[node] = degree * ball_size
    # 标准化CI值
    max_ci = max(ci_scores.values())
    ci_scores = {k: v / max_ci for k, v in ci_scores.items()}
    return ci_scores

def calculate_eigenvector(G):
    # 注意：对于有多个组件的图，可能需要特殊处理
    try:
        eigenvector_scores = nx.eigenvector_centrality(G, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        print("Eigenvector centrality did not converge, returning degree centrality as fallback.")
        eigenvector_scores = nx.degree_centrality(G)
    return eigenvector_scores

def random_walk(G, node, walk_length):
    walk = [node]
    for i in range(walk_length - 1):
        # 在尝试访问邻居之前检查节点是否存在于图中
        if not G.has_node(node):
            raise ValueError(f"节点 {node} 不在图中")
        neighbors = list(G.neighbors(walk[-1]))
        if neighbors:
            walk.append(random.choice(neighbors))
    return walk

def generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(G, str(node), walk_length))
    return walks

def Dismantling(graph, seed_nodes, threshold):
    A = graph.get_edgelist()
    G = nx.Graph(A)  # In case you graph is undirected
    original_largest_cc = G.number_of_nodes()

    TAS_NUM = 0
    TAS_CON = dict.fromkeys(seed_nodes)

    for node in seed_nodes:
        G.remove_node(node)
        if len(G) == 0:
            residual_largest_cc = 0
        else:
            residual_largest_cc = len(max(nx.connected_components(G), key=len)) / original_largest_cc
        TAS_CON[node] = residual_largest_cc

    for node in seed_nodes:
        if float(TAS_CON[node]) > threshold:
            TAS_NUM += 1
        else:
            break

    return TAS_NUM

from typing import List, Tuple, Dict, Union

class DSU:
    __slots__ = ("parent", "size", "max_size")
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.max_size = 1

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.size[ra] > self.max_size:
            self.max_size = self.size[ra]

def dismantling_fast_linear(g_or_tuple: Union["igraph.Graph", Tuple[int, List[Tuple[int,int]]]],
    seed_nodes: List[int],
    threshold: float
) -> Tuple[int, Dict[int, float]]:
    """
    线性时间计算: 删除 seed_nodes 的前 k 个后，最大连通分量占比。
    返回:
      TAS: 最小 k 使得 max_comp_size / n <= threshold
      TAS_CON: {seed_nodes[i]: residual_ratio_after_removal_of_that_node}
    """
    # 1) 取 n 与无向边集
    try:
        import igraph as ig
        if isinstance(g_or_tuple, ig.Graph):
            n = g_or_tuple.vcount()
            edges = g_or_tuple.get_edgelist()  # list[(u,v)]
        else:
            n, edges = g_or_tuple
    except Exception:
        # 没装 igraph 或传的是 tuple
        n, edges = g_or_tuple

    # 2) 构建邻接表（无向）
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)

    # 3) 标记将被删除的节点
    removed_flag = [False] * n
    for v in seed_nodes:
        if 0 <= v < n:
            removed_flag[v] = True
        else:
            raise ValueError(f"seed node id {v} out of range [0,{n})")

    # 4) DSU 仅在“未删除”的基底节点上初始化
    dsu = DSU(n)
    active = [False] * n
    for v in range(n):
        if not removed_flag[v]:
            active[v] = True
    # 基底合并
    for u, v in edges:
        if active[u] and active[v]:
            dsu.union(u, v)
    # 计算基底的最大连通分量（可能全空）
    base_active_count = sum(active)
    dsu.max_size = 0 if base_active_count == 0 else dsu.max_size

    # 5) 反向把 seed_nodes 加回来，得到每一步的最大分量大小
    L = len(seed_nodes)
    # size_after_k: 删除前 k 个节点后的最大分量大小
    size_after_k = [0] * (L + 1)
    size_after_k[L] = dsu.max_size  # 删除完全部 seed_nodes 后（即仅基底） 的最大分量

    # 逆序添加
    for t in range(L - 1, -1, -1):
        v = seed_nodes[t]
        if not active[v]:
            active[v] = True
            dsu.max_size = max(dsu.max_size, 1)
            # 与所有已激活的邻居合并
            for u in adj[v]:
                if active[u]:
                    dsu.union(v, u)
        size_after_k[t] = dsu.max_size

    # 6) 得到 TAS 与 TAS_CON
    TAS = L
    TAS_CON: Dict[int, float] = {}
    for i, v in enumerate(seed_nodes):
        residual_ratio = size_after_k[i+1] / float(n)  # 移除第 i 个后
        TAS_CON[v] = residual_ratio
        if TAS == L and residual_ratio <= threshold:
            TAS = i + 1

    return TAS, TAS_CON

def get_loop_G(network):
    datadir = os.path.join('.', 'data', 'data', 'realworld',  f'{network}.txt')

    G = nx.Graph()

    for line in open(datadir):
        strlist = line.split()
        n1 = int(strlist[0])
        n2 = int(strlist[1])
        G.add_edges_from([(n1, n2)])
    nodes = G.nodes()
    G.add_edges_from([(node, node) for node in nodes])  # 添加自环
    return G

def save_results_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(str(data))


def dismantle_network(G, graph, thresholds, network):
    node_num = len(G.nodes())

    # 计算Collective Influence分数
    scores = calculate_collective_influence(G)

    # deepwalk
    # walks = generate_walks(G, num_walks=10, walk_length=80)
    # model = Word2Vec(walks, vector_size=1, window=5, min_count=0, sg=1, workers=2)
    # scores = {node: model.wv[str(node)] for node in G.nodes()}

    sorted_nodes = sorted(scores, key=scores.get, reverse=True)

    # 指定保存文件的路径
    results_dir = 'dismantling_results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"CI_{network}_results.txt")

    # 针对每个阈值计算拆解分数并保存结果
    with open(results_file, 'a') as file:
        for threshold in thresholds:
            # TAS = Dismantling(graph, np.array(sorted_nodes), threshold=threshold)
            TAS, TAS_CON = dismantling_fast_linear(graph, seed_nodes=sorted_nodes, threshold=threshold)
            TAS_ratio = TAS / node_num
            file.write(f"Threshold {threshold}: {TAS_ratio}\n")


# 主程序
# WS网络
# List = ['WS_size466_avgdeg4', 'WS_size454_avgdeg4', 'WS_size406_avgdeg4', 'WS_size444_avgdeg4', 'WS_size402_avgdeg4', 'WS_size933_avgdeg4', 'WS_size922_avgdeg4', 'WS_size971_avgdeg4','WS_size989_avgdeg4', 'WS_size1000_avgdeg4', 'WS_size1477_avgdeg4', 'WS_size1493_avgdeg4', 'WS_size1451_avgdeg4', 'WS_size1960_avgdeg4', 'WS_size1970_avgdeg4', 'WS_size1931_avgdeg4']
# List = ['WS_size1000_avgdeg3-1', 'WS_size1000_avgdeg3-2', 'WS_size1000_avgdeg4-1', 'WS_size1000_avgdeg4-2', 'WS_size1000_avgdeg5-1', 'WS_size1000_avgdeg5-2', 'WS_size1000_avgdeg6-1', 'WS_size1000_avgdeg6-2']
List = ['DIMACS10'] # 'Douban',
# BA网络
#List = ['BA_size451_avgdeg4', 'BA_size412_avgdeg4', 'BA_size909_avgdeg4', 'BA_size989_avgdeg4', 'BA_size1407_avgdeg4', 'BA_size1465_avgdeg4', 'BA_size1498_avgdeg4', 'BA_size1951_avgdeg4', 'BA_size1958_avgdeg4']
# List = ['BA_size1000_avgdeg3-1', 'BA_size1000_avgdeg3-2', 'BA_size1000_avgdeg4-1', 'BA_size1000_avgdeg5-1', 'BA_size1000_avgdeg5-2', 'BA_size1000_avgdeg6-1', 'BA_size1000_avgdeg6-2']

# PLC
# List = ['PLC_size449_avgdeg4', 'PLC_size490_avgdeg4', 'PLC_size975_avgdeg4', 'PLC_size954_avgdeg4', 'PLC_size940_avgdeg4', 'PLC_size1440_avgdeg4', 'PLC_size1460_avgdeg4', 'PLC_size1901_avgdeg4', 'PLC_size1979_avgdeg4']
# List = ['Crime', 'NetScience', 'FilmTrust', 'DNCEmails', 'Figeys', 'RoviraVirgili', 'PPI']
# thresholds = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]  # 不同的阈值
thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.4]
for network in List:
    G = get_loop_G(network)
    datadir = os.path.join('.', 'data', 'data', 'realworld')
    graph = igraph.Graph().Read_Edgelist(os.path.join(datadir, f'{network}.txt'), directed=False)
    dismantle_network(G, graph, thresholds, network)



