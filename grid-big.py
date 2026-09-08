import logging
import os
from datetime import time

import optuna
import torch
import yaml
import time
from yaml import SafeLoader

from generate_role import generate_role_matrix
from incidence_matrix import gen_coo_simplex_A
from model_main_v2 import Hyper_GNN
# from train_realworld_v1 import get_loop_G, get_no_loop_G, Dismantling
from train_realworld_Norder import *
from utils import EarlyStopping, LoadRealworldData, save_to_file, save_config_to_txt, ensure_sparse_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'DCRS'

network = 'DIMACS10'
config = yaml.load(open('config_realworld.yaml'), Loader=SafeLoader)[network]

def train_model(Network, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(int(config['seed']))  # reproducibility
    patience = 24

    DATE = time.strftime('%m-%d', time.localtime())
    TIME = time.strftime('%H.%M.%S', time.localtime())
    title = os.path.join('log', model_name)
    log_path = os.path.join(title, f'{Network}', f'{DATE}', f'debug-{DATE}_{TIME}')
    os.makedirs(log_path, exist_ok=True)

    path_checkpoint = log_path
    early_stopping = EarlyStopping(logging, patience=patience, verbose=True, path=path_checkpoint, network=Network)

    G_loop = get_loop_G(Network)  # 这里获得的是添加了自环的图
    G_single = get_no_loop_G(Network)

    data = load_processed_data(network)

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

        A = ensure_sparse_tensor(data['train_adj_matrices'], device=device)
        loss = model.loss(score, A, gamma=config['gamma'], mean=False)

        _, TAS_indices = torch.topk(score, score.numel())

        TAS, TAS_CON = Dismantling(data['train_g'], TAS_indices.cpu().numpy())

        #rein_TAS= reinsertion(G_loop, list(TAS_CON.keys())[:TAS], dismantling_threshold=0.01, metric='SPR')

        TAS_ratio = TAS / score.numel()

        if best_Tperf > TAS_ratio:
            best_Tperf = TAS_ratio
            best_epoch = epoch
            #best_TAS_CON = rein_TAS  # 更新最佳TAS_CON，重插
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

    save_best_TAS_CON_rein(best_TAS_CON, log_path, Network)  # 这里是重插的最好结果输出
    # save_scores(order_weight, log_path, Network, config['sim_order'])  # 这里保存每一阶order的权重
    return best_Tperf

def objective(trial, network, config):
    # 提取超参数
    config['n_role'] = trial.suggest_categorical('n_role', [2, 3, 4, 5, 6, 7, 8, 9, 10])
    config['sim_num_hidden'] = trial.suggest_categorical('sim_ num_hidden', [8, 12, 16, 24, 32])
    config['sim_order'] = trial.suggest_categorical('sim_order', [2])
    config['sim_k'] = trial.suggest_categorical('sim_k', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    config['sim_alpha'] = trial.suggest_float('sim_alpha', 0.1, 1.0)
    config['sim_dprate'] = trial.suggest_float('sim_dprate', 0.1, 1.0)
    config['sim_Init'] = trial.suggest_categorical('sim_Init', ['PPR', 'NPPR', 'Random'])
    config['role_num_hidden'] = trial.suggest_categorical('role_num_hidden', [8, 12, 16, 24, 32])
    config['role_k'] = trial.suggest_categorical('role_k', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    config['role_alpha'] = trial.suggest_float('role_alpha', 0.1, 1.0)
    config['role_dprate'] = trial.suggest_float('role_dprate', 0.1, 1.0)
    config['role_Init'] = trial.suggest_categorical('role_Init', ['PPR', 'NPPR', 'Random'])
    config['gamma'] = trial.suggest_float('gamma', 0.1, 1.0)  # 添加gamma参数
    config['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)  # 添加learning_rate参数
    config['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)  # 添加weight_decay参数


    # 训练模型并获取结果
    best_Tperf = train_model(network, config)

    # 返回最佳结果
    return best_Tperf

# 创建一个study对象并开始优化
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, network, config), n_trials=500)
# 打印最佳参数
print(study.best_params)
print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
print(f"Best trial number: {study.best_trial.number}")