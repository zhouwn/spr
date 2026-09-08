
import time
import logging
import pickle
import logging

import pandas as pd
import scipy.sparse
from utils import *
#from incidence_matrix import *
#from generate_role import *
from generate_role_big import *
from model_main_v2 import Hyper_GNN
from hypergraph_conv import HypergraphConv
from get_simplex import *

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


def data_process(Network, path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G_loop = get_loop_G(Network)  # 这里获得的是添加了自环的图
    G_single = get_no_loop_G(Network)


    node_feature, train_adj_matrices, train_g = LoadRealworldData(Network, G_loop)
    #node_role_matrix, role_feature = extra_role(G_loop, n_Role=8)###这两行生成的是小规模的
    #node_role_matrix, role_node_idx, max_role_len, node_role_norm_matrix = generate_role_matrix(G_loop, n_Role)  # 输出一个R*N的列表，R是角色，N是属于该角色的节点编号
    #node_role_matrix, role_feature = generate_role_matrix(G_single, n_Role=8)  # 输出的是节点和角色的关联矩阵和角色特征，这里是处理大规模的，用的是k-means
    # --- 从CSV文件加载角色信息 ---
    # 2. 定义您的CSV文件名（请确保文件名与您的Network变量对应）
    # 假设您的CSV文件与此脚本在同一目录下，或者提供完整路径
    node_role_file = os.path.join(path, f'{Network}_node_roles_one_hot.csv')
    role_feature_file = os.path.join(path, f'{Network}_role_features.csv')

    print(f"Loading role data from {node_role_file} and {role_feature_file}...")

    # 3. 读取 node_role_matrix
    try:
        df_node_roles = pd.read_csv(node_role_file)
        # 将 pandas DataFrame 转换为 PyTorch Tensor
        node_role_matrix = torch.from_numpy(df_node_roles.values).float()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {node_role_file}！请检查文件名和路径。")
        return  # 如果文件不存在，则停止执行

    # 4. 读取 role_feature
    try:
        # index_col=0 将第一列 'feature_name' 作为行索引
        df_role_features = pd.read_csv(role_feature_file, index_col=0)
        # 您的文件中，列是角色，行是特征。我们需要 [角色数 x 特征数] 的格式，所以需要转置。
        # 使用 .T 进行转置
        df_role_features_transposed = df_role_features.T
        # 将转置后的 DataFrame 转换为 PyTorch Tensor
        role_feature = torch.from_numpy(df_role_features_transposed.values).float()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {role_feature_file}！请检查文件名和路径。")
        return  # 如果文件不存在，则停止执行

    print("Role data loaded successfully.")
    print(f"Shape of node_role_matrix: {node_role_matrix.shape}")
    print(f"Shape of role_feature: {role_feature.shape}")
    # ------------------------------------
    edge, faces, tetrahedra, pentachora = get_simplex_idx(G_single)

    edge_node_ft, faces_node_ft, tetrahedra_node_ft, pentachora_node_ft = get_simplex_ft(node_feature, edge, faces, tetrahedra, pentachora)

    print(f"edge:{len(edge)}, faces:{len(faces)}, tetrahedra:{len(tetrahedra)}, pentachora:{len(pentachora)}" )

    ## 把node_role_matrix转成稀疏格式
    scipy_sparse_matrix = scipy.sparse.coo_matrix(node_role_matrix)
    indices = torch.from_numpy(np.vstack((scipy_sparse_matrix.row, scipy_sparse_matrix.col))).long()
    values = torch.from_numpy(scipy_sparse_matrix.data)
    shape = torch.Size(scipy_sparse_matrix.shape)
    node_role_matrix = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    # 创建保存路径
    save_path = os.path.join(path, f'{Network}.pt')

    # 将所有数据保存到一个字典中
    data_to_save = {
        'node_feature': node_feature,
        'node_role_matrix': node_role_matrix,
        'role_feature': role_feature,
        'train_adj_matrices': train_adj_matrices,
        'train_g': train_g,
        'edge': edge,
        'faces': faces,
        'tetrahedra': tetrahedra,
        'pentachora': pentachora,
        'edge_node_ft': edge_node_ft,
        'faces_node_ft': faces_node_ft,
        'tetrahedra_node_ft': tetrahedra_node_ft,
        'pentachora_node_ft': pentachora_node_ft
    }

    # 使用 torch.save() 保存数据
    torch.save(data_to_save, save_path)

    print(f"Data for {Network} has been processed and saved to {save_path}")



if __name__ == '__main__':
    List = ['citeseer']  #Github_Contest, DIMACS10, citeseer, TwitterFollows, Github_Contest, Douban
    # List = ['0-demo', 'DNCEmails', 'Figeys', 'FilmTrust', 'NetScience', 'EconPoli', 'KKI', 'RoviraVirgili']  # LastFM太大
    # List = ['DNCEmails', 'Genefusion', 'Bible', 'Infectious', 'AirTraffic', 'PPI', 'LastFM']  # LastFM太大
    # List = ['LastFM']
    # List = ['Flickr', 'Gnutella', 'HI-II-14', 'Figeys', 'Ca-GrQc', 'Vidal']
    # List = ['Figeys', 'Ca-GrQc']
    for network in List:

        path = os.path.join('data', 'data', 'realworld')
        os.makedirs(path, exist_ok=True)
        data_process(network, path)