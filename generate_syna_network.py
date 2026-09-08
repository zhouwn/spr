import networkx as nx
import matplotlib.pyplot as plt
import os
import random


def generate_synthetic_network(network_type, size, avg_degree):
    if network_type == 'ER':
        # Erdös-Rényi model
        # For a given average degree <k>, the probability p is <k> / (n-1)
        p = avg_degree / (size - 1)
        G = nx.erdos_renyi_graph(size, p)
    elif network_type == 'BA':
        # Barabási-Albert model
        # For a given average degree <k>, the number of edges to attach from a new node to existing nodes (m) is <k> / 2
        m = avg_degree // 2
        G = nx.barabasi_albert_graph(size, m)
    elif network_type == 'PLC':
        # Powerlaw Cluster model
        # For a given average degree <k>, the number of edges to attach from a new node to existing nodes (m) is <k> / 2
        # Let's assume a default clustering coefficient (triangle formation probability) of 0.05
        m = avg_degree // 2
        G = nx.powerlaw_cluster_graph(size, m, 0.05)
    elif network_type == 'WS':
        # Watts-Strogatz model
        # For a given average degree <k>, each node is connected to <k> nearest neighbors in ring topology
        # Let's assume a default rewiring probability of 0.1
        k = avg_degree
        G = nx.watts_strogatz_graph(size, k, 0.1)
    else:
        raise ValueError("Unsupported network type. Choose from 'ER', 'BA', 'PLC', or 'WS'.")

    return G

def ensure_all_nodes_present(G, size):
    # 找到所有孤立的节点
    isolated_nodes = list(nx.isolates(G))

    while isolated_nodes:
        # 从孤立节点列表中取出一个节点
        node = isolated_nodes.pop()

        # 随机选择一个非孤立的节点
        non_isolated_node = random.choice(list(set(G.nodes()) - set(isolated_nodes)))

        # 将孤立节点连接到非孤立节点
        G.add_edge(node, non_isolated_node)

    return G

def ensure_triangle_exists(G):
    # 确保网络中至少存在一个三角形
    if len(G.nodes()) < 3:
        raise ValueError("Network must have at least 3 nodes to form a triangle.")

    # 随机选择三个不同的节点
    nodes = random.sample(G.nodes(), 6)
    # 连接这三个节点，以形成三角形
    G.add_edge(nodes[0], nodes[1])
    G.add_edge(nodes[1], nodes[2])
    G.add_edge(nodes[0], nodes[2])
    # 形成第二个三角形
    G.add_edge(nodes[3], nodes[4])
    G.add_edge(nodes[4], nodes[5])
    G.add_edge(nodes[3], nodes[5])

    return G

def save_network(G, network_type, avg_degree, folder_path):
    # 根据网络类型和平均节点度生成文件名
    filename = f"{network_type}_size{len(G.nodes())}_avgdeg{avg_degree}.txt"

    # 确保文件夹存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 完整的文件路径
    file_path = os.path.join(folder_path, filename)

    # 保存网络到文本文件
    with open(file_path, 'w') as file:
        for edge in G.edges():
            file.write(f"{edge[0]} {edge[1]}\n")

    print(f"Network saved to {file_path}")


# Example usage:
network_type = 'BA'  # Choose from 'ER', 'BA', 'PLC', 'WS'
size = random.randint(1400, 1500)  # 生成随机节点数量的网络Random number of nodes between 50 and 100
# size = 1000  # 生成固定节点数量的网络Number of nodes
avg_degree = 4  # Average degree
folder_path = r'D:\experiment\0_high_order_dis\0_last_model\0_syn_network\data\data\syn_network\BA_Scale_N'  # 指定文件夹路径

# Generate the network
G = generate_synthetic_network(network_type, size, avg_degree)
G = ensure_all_nodes_present(G, size)
# G = ensure_triangle_exists(G)  # 确保存在三角形


save_network(G, network_type, avg_degree, folder_path)