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

def save_best_TAS_CON(TAS_CON, log_path, network):
    file_path = os.path.join(log_path, f'{network}_best_TAS_CON.txt')
    with open(file_path, 'w') as f:
        f.write('{\n')
        for key, value in TAS_CON.items():
            f.write(f'{key}: {value},\n')
        f.write('}')
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

        data = load_processed_data(network)

        #n_Role = config['n_role']
        #node_role_matrix, role_feature = generate_role_matrix(G_loop, n_Role)  # 这里输出的是节点和角色的关联矩阵和角色特征
        #node_feature, train_adj_matrices, train_g, edge_data, edge_list = LoadRealworldData(Network, G_loop)

        #edge, faces, tetrahedra, pentachora = get_simplex_idx(G_single)
        #edge_node_ft, faces_node_ft, tetrahedra_node_ft, pentachora_node_ft = get_simplex_ft(node_feature, edge, faces, tetrahedra, pentachora)



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
        best_TAS_CON = None  # 添加这行来跟踪最佳的TAS_CON

        for epoch in range(500):
            optimizer.zero_grad()

            # score = model(node_feature, edge_data, train_adj_matrices, A_Hat_01, A_Hat_02, A_adj, beta=0.7).squeeze(1)  # 这里输入的是边的连接，节点特征和
            # score = model(node_feature, HL, edge_feature, edge_HL, node_to_edge_incide)
            score, order_weight= model(**{
                'G_loop': G_loop,
                'G_single': G_single,
                'data': data})
            score = score.squeeze(1)
            loss = model.loss(score, data['train_adj_matrices'].to(device), gamma=config['gamma'], mean=False)

            _, TAS_indices = torch.topk(score, score.numel())

            TAS, TAS_CON = Dismantling(data['train_g'], TAS_indices.cpu().numpy())
            TAS_ratio = TAS / score.numel()

            if best_Tperf > TAS_ratio:
                best_Tperf = TAS_ratio
                best_epoch = epoch
                best_TAS_CON = TAS_CON  # 更新最佳TAS_CON


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

        save_best_TAS_CON(best_TAS_CON, log_path, Network)
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # 结束时内存使用量（MB）
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        return {best_epoch: best_Tperf}, execution_time, memory_used


if __name__ == '__main__':

    model_name = 'DCRS'
    total_time = 0
    total_memory = 0
    List = ['DIMACS10']
    for network in List:
        config = yaml.load(open('config_realworld.yaml'), Loader=SafeLoader)[network]

        seed = int(config['seed'])
        patience = 24
        title = os.path.join('log', model_name)
        DATE = time.strftime('%m-%d', time.localtime())
        TIME = time.strftime('%H.%M.%S', time.localtime())
        log_path = os.path.join(title, f'{network}', f'{DATE}', f'debug-{DATE}_{TIME}')
        os.makedirs(log_path, exist_ok=True)
        _, execution_time, memory_used = Train(network, log_path, config)
        total_time += execution_time
        total_memory += memory_used

        print(f"{network}_Total execution time for all networks: {total_time:.2f} seconds")
        print(f"{network}_Total memory used for all networks: {total_memory:.2f} MB")




