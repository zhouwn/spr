import copy
from pathlib import Path

import psutil
import torch, gc
import time
import logging
import pickle
import logging
import os
import igraph
import networkx as nx
from glob import glob
import argparse
import random
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from yaml import SafeLoader
from utils import *
from incidence_matrix import *
from generate_role import *
from model_main_v2 import Hyper_GNN
from hypergraph_conv import HypergraphConv


def get_loop_G(network):
    datadir = os.path.join('.', 'data', 'data', 'realworld', f'{network}.txt')

    G = nx.Graph()

    for line in open(datadir):
        strlist = line.split()
        n1 = int(strlist[0])
        n2 = int(strlist[1])
        G.add_edges_from([(n1, n2)])
    nodes = G.nodes()
    G.add_edges_from([(node, node) for node in nodes])  # 添加自环
    return G

def get_no_loop_G(network):
    datadir = os.path.join('.', 'data', 'data', 'realworld', f'{network}.txt')

    G = nx.Graph()

    for line in open(datadir):
        strlist = line.split()
        n1 = int(strlist[0])
        n2 = int(strlist[1])
        G.add_edges_from([(n1, n2)])
    G.remove_edges_from(nx.selfloop_edges(G))  # 去除数据集中本身存在自环的节点

    return G


def save_model(net, Network, model_name):
    checkpoint = {"model_state_dict": net.state_dict()}
    path_checkpoint = os.path.join('log', model_name + '_' + FEATURE_TYPE + '_' + OPTIMIOZER, f'{Network}',f'{DATE}', f'debug-{DATE}_{TIME}', f"{model_name}.pt")
    torch.save(checkpoint, path_checkpoint)

def load_processed_data(network):
    file_path = os.path.join('data', 'data', 'realworld', f'{network}.pt')
    return torch.load(file_path)


def run_greedy(graph: nx, nodes_id, threshold):
    """
    :param graph: networkx.Graph()
    :param nodes_id: 节点排序
    :param threshold:
    :return: the id of nodes that should be removed after reinsertion
    """
    N = graph.number_of_nodes()
    nseed = len(nodes_id)
    seed = [0] * N

    for i in nodes_id:
        seed[i] = 1

    nodes = []  # store the id of nodes that should be removed after reinsertion

    rank = [0] * N
    parent = [i for i in range(N)]
    present = [0] * N  # flag: the node in the network or not
    size_comp = [0] * N


    def find_set(i):
        if parent[i] != i:
            parent[i] = find_set(parent[i])
        return parent[i]

    def union_set(i, j):
        ri = find_set(i)
        rj = find_set(j)
        if ri != rj:
            if rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[ri] = rj
                if rank[ri] == rank[rj]:
                    rank[rj] += 1

    ngiant = 0  # nodes num in gcc

    num_comp = N
    nedges = 0

    def compute_comp(i, present, size_comp, N):
        mask = [0] * N
        compos = []
        nc = 1
        ncomp = 0

        for j in graph.neighbors(i):
            if present[j]:
                c = find_set(j)
                if not mask[c]:
                    compos.append(c)
                    mask[c] = 1
                    nc += size_comp[c]
                    ncomp += 1

        for k in compos:
            mask[k] = 0

        return nc, ncomp

    for i in range(N):
        if seed[i]:
            continue

        nc, ncomp = compute_comp(i, present, size_comp, N)
        present[i] = 1
        num_comp += 1 - ncomp

        for j in graph.neighbors(i):
            if present[j]:
                union_set(i, j)
                nedges += 1

        size_comp[find_set(i)] = nc

        if nc > ngiant:
            ngiant = nc

    compos = []

    for t in range(nseed, 0, -1):  # not a seed?
        nbest = N  # the new size after this reinsertion? see line 212
        ibest = 0
        ncompbest = 0

        for i in range(N):  # 遍历所有非种子节点，将它们添加到网络中
            if present[i]:  # node i is in the network
                continue

            nc, ncomp = compute_comp(i, present, size_comp, N)

            if nc < nbest:
                ibest = i
                nbest = nc
                ncompbest = ncomp

        present[ibest] = 1
        num_comp += 1 - ncompbest

        for j in graph.neighbors(ibest):
            if present[j]:
                union_set(ibest, j)
                nedges += 1

        size_comp[find_set(ibest)] = nbest

        if nbest > ngiant:
            ngiant = nbest

        if nbest >= threshold:
            break

        seed[ibest] = 0

    for i in range(N):
        if seed[i]:
            nodes.append(i)  # here is not i+1

    return nodes
def get_gcc(g:nx):
    """ Calculate the size of  giant component component size (GCC) """

    if len(g) == 0:
        return 0

    return max([len(c) for c in nx.connected_components(g)])

def Dismantling(graph, seed_nodes):
    A = graph.get_edgelist()
    G = nx.Graph(A)  # In case you graph is undirected
    original_largest_cc = G.number_of_nodes()
    threshold = 0.01
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

    return TAS_NUM, TAS_CON

def reinsertion(graph: nx, nodes_id, dismantling_threshold=0.01, metric='SPR'):
    """
    :param graph:
    :param nodes_id:
    :param dismantling_threshold:
    :param rel_path: reinsertion 结果存储的路径，为None则不存储
    :param metric: reinsertion结果的名称
    :return:
    """
    G = copy.deepcopy(graph)
    N = G.number_of_nodes()
    target_size = dismantling_threshold * N
    if target_size <= 1:
        target_size = 2

    # Two isolated nodes were added for the program to work properly. 为了程序正常运行添加了两个孤立节点
    G.add_node(0)
    G.add_node(N)
    attacked_nodes = run_greedy(G, nodes_id, target_size)  # 选择攻击节点

    def sort_nodes(N, nodes_id, attacked_nodes):
        remove_list = []
        exist = [0] * (N + 1)

        for v in attacked_nodes:
            exist[v] = 1

        for v in nodes_id:
            if exist[v]:
                remove_list.append(v)

        return remove_list

    # Sort the order of node removal. 对节点移除的顺序进行排序
    remove_list = sort_nodes(N, nodes_id, attacked_nodes)
    # print("Number of attacked nodes after reinsertion:", len(remove_list))


    return remove_list

# 高效版：线性时间动态连通性（反向加点）
# 兼容 igraph.Graph 或 (n, edges) 形式，edges 为 0-based 无向边列表

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


def save_scores(order_weight, log_path, network, order):
    file_path = os.path.join(log_path, f'{network}_best_TAS_order{order}.txt')
    # 将张量移动到 CPU 并转换为 numpy 数组
    scores_np = order_weight.cpu().detach().numpy()

    with open(file_path, 'w') as f:
        for i, sample in enumerate(scores_np):
            f.write(f"Node {i}:\n")
            for j, values in enumerate(sample):
                f.write(f"  Order {j}: {', '.join([f'{v:.4f}' for v in values])}\n")
            f.write("\n")  # 在每个样本之间添加一个空行
    print(f"Scores saved to {file_path}")


def save_best_TAS_CON_rein(TAS_CON, log_path, network):
    file_path = os.path.join(log_path, f'{network}_best_TAS.txt')
    with open(file_path, 'w') as f:
        for value in TAS_CON:
            f.write(f"{value}\n")
    print(f"Best TAS_CON saved to {file_path}")


def Train(Network, log_path, config):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 初始内存使用量（MB）

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(int(config['seed']))  # reproducibility
        patience = 24
        path_checkpoint = log_path
        early_stopping = EarlyStopping(logging, patience=patience, verbose=True, path=path_checkpoint, network=Network)

        G_loop = get_loop_G(Network)  # 这里获得的是添加了自环的图
        G_single = get_no_loop_G(Network)

        data = load_processed_data(Network)

        model = Hyper_GNN(N_roles=config['n_role'],  # 角色数量
                          node_feature_shape=data['node_feature'].shape[1],  # 节点特征维度
                          role_feature_shape=data['role_feature'].shape[1],  # 角色特征维度
                          role_num_hid=config['role_num_hidden'],
                          role_K=config['role_k'],
                          role_alpha=config['role_alpha'],
                          role_dprate=config['role_dprate'],
                          role_Init=config['role_Init'],
                          sim_num_hid=config['sim_num_hidden'],
                          sim_Order=config['sim_order'],
                          sim_K=config['sim_k'],
                          sim_alpha=config['sim_alpha'],
                          sim_dprate=config['sim_dprate'],
                          sim_Init=config['sim_Init']).to(device)

        save_config_to_txt(config=config, file_path=path_checkpoint, network=Network)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        # print(f"Network: {Network})
        print(f"Network: {Network}")

        best_Tperf = 1
        best_epoch = 0
        best_TAS_CON = []  # 添加这行来跟踪最佳的TAS_CON

        for epoch in range(500):
            optimizer.zero_grad()

            score = model(**{
                'G_loop': G_loop,
                'G_single': G_single,
                'data': data})
            score = score.squeeze(1)
            loss = model.loss(score, data['train_adj_matrices'], gamma=config['gamma'], mean=False)

            _, TAS_indices = torch.topk(score, score.numel())

            TAS, TAS_CON = dismantling_fast_linear(data['train_g'], seed_nodes=TAS_indices.cpu().tolist(), threshold=0.01)

            # rein_TAS= reinsertion(G_loop, list(TAS_CON.keys())[:TAS], dismantling_threshold=0.01, metric='SPR')

            TAS_ratio = TAS / score.numel()

            if best_Tperf > TAS_ratio:
                best_Tperf = TAS_ratio
                best_epoch = epoch
                # best_TAS_CON = rein_TAS  # 更新最佳TAS_CON，重插
                best_TAS_CON = list(TAS_CON.keys())[:TAS]  # 这是如果不采用重插的输出结果


            early_stopping(TAS_ratio, model, epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            print(f"Train Epoch {epoch} | Loss: {loss:.4f} | TAS: {TAS}/{score.numel()}={TAS_ratio:.6f} ")
            save_to_file(path=path_checkpoint,
                         data=f"Train Epoch {epoch} | Loss: {loss:.4f} | TAS: {TAS}/{score.numel()}={TAS_ratio:.6f}",
                         network=Network)

            loss.backward()
            optimizer.step()

        save_best_TAS_CON_rein(best_TAS_CON, log_path, Network) # 这里是重插的最好结果输出

        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 结束时内存使用量（MB）
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        return {best_epoch: best_Tperf}, execution_time, memory_used


if __name__ == '__main__':

    model_name = 'DCRS'
    total_time = 0
    total_memory = 0
    List = ['Github_Contest']  #DIMACS10， Github_Contest
    # List = ['0_ER']
    # List = ['Crime']
    #List = ['RoviraVirgili', 'KKI', 'EconPoli', 'NetScience', 'FilmTrust', 'Figeys', 'DNCEmails', 'PPI', 'Crime']  # LastFM太大
    # List = ['LastFM']
    # List = ['Genefusion']
    # List = ['Flickr', 'Gnutella', 'HI-II-14', 'Figeys', 'Ca-GrQc', 'Vidal']
    # List = ['Figeys', 'Ca-GrQc']
    for network in List:
        config = yaml.load(open('config_realworld.yaml'), Loader=SafeLoader)[network]

        seed = int(config['seed'])
        patience = 24
        title = os.path.join('log', model_name)
        DATE = time.strftime('%m-%d', time.localtime())
        TIME = time.strftime('%H.%M.%S', time.localtime())
        log_path = os.path.join(title, f'{network}', f'{DATE}', f'debug-{DATE}_{TIME}')
        os.makedirs(log_path, exist_ok=True)
        Train(network, log_path, config)




