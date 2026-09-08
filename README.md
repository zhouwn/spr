# SPR Code

本项目引入了节点角色信息、边/三角形等高阶单纯形结构，并使用稀疏矩阵与动态连通性评估来适配较大规模真实网络。

核心目标是：对输入网络中的节点进行重要性排序，尽量用更少的攻击节点使最大连通分量下降到给定阈值以下。训练过程是无监督的，模型输出每个节点的拆解分数，再按照分数从高到低评估 TAS（Targeted Attack Set）比例。

## 代码整体逻辑

当前主流程分为两步：

1. 预处理真实网络数据，生成训练缓存 `.pt`。
2. 读取 `.pt` 缓存与 YAML 配置，训练高阶 DCRS 模型并输出最佳拆解序列。

运行链路如下：

```text
data/data/realworld/<Network>.txt
        |
        |  data_process.py
        v
data/data/realworld/<Network>.pt
        |
        |  train_realworld_big.py
        v
Hyper_GNN(model_main_v2.py)
        |
        +-- Role_GCN(Role_GCN.py): 角色分支
        |
        +-- HiGCN(HiGCN.py): 高阶单纯形分支
        |
        v
节点拆解分数 -> TAS 评估 -> log/DCRS/<Network>/...
```

## 环境依赖

原始 `requirement.txt` 中记录了基础依赖：

```text
igraph==0.9.9
networkx==2.6.3
numpy==1.19.5
PyGSP==0.5.1
PyYAML==6.0
scikit-learn==0.24.2
scipy==1.5.4
torch==1.7.1+cu101
dgl==0.7.0+cu101
```

当前代码还实际使用了以下包：

```text
pandas
psutil
tqdm
optuna
matplotlib
graphrole
torch-geometric
torch-sparse
torch-scatter
```

建议使用 Python 3.7 和 CUDA 10.1 对齐当前 `venv` 与依赖版本。`torch-geometric`、`torch-sparse`、`torch-scatter` 需要安装与 PyTorch/CUDA 匹配的 wheel，否则模型中的稀疏传播会导入失败。

安装基础依赖：

```bash
pip install -r requirement.txt
```

如果缺少 PyG 相关依赖，请按本机 CUDA 与 PyTorch 版本补装兼容版本。

## 数据目录

真实网络默认放在：

```text
data/data/realworld/
```

每个网络至少需要一个边列表文件：

```text
<Network>.txt
```

文件格式为无向边列表，每行两个节点编号：

```text
0 1
1 2
2 3
```

当前代码多处直接用节点 id 索引张量，因此建议节点编号使用从 `0` 开始的连续整数。如果输入网络不是连续编号，需要先重编号。

对于大图，`data_process.py` 默认从 CSV 读取预先生成好的角色信息：

```text
<Network>_node_roles_one_hot.csv
<Network>_role_features.csv
```

其中：

- `<Network>_node_roles_one_hot.csv` 是节点-角色 one-hot 矩阵，形状约为 `[N, R]`。
- `<Network>_role_features.csv` 是角色特征表，读取时第一列作为索引，随后转置为 `[R, F]`。

预处理成功后会生成：

```text
<Network>.pt
```

该文件是训练脚本直接读取的缓存，包含节点特征、角色矩阵、角色特征、稀疏邻接矩阵、igraph 图对象、边、三角形以及对应的单纯形节点特征。

## 数据预处理

打开 `data_process.py`，在文件底部修改需要处理的数据集：

```python
List = ['citeseer']
```

然后运行：

```bash
python data_process.py
```

预处理做的事情包括：

- 读取 `data/data/realworld/<Network>.txt` 构建 NetworkX 图。
- 生成带自环图 `G_loop` 和不带自环图 `G_single`。
- 通过 `LoadRealworldData` 生成节点结构特征和稀疏邻接矩阵。
- 从角色 CSV 中读取节点-角色矩阵和角色特征。
- 使用 `get_simplex_idx` 枚举边、三角形、四面体、五面体。
- 使用 `get_simplex_ft` 将节点特征切片成边/三角形等单纯形特征。
- 将所有对象保存到 `data/data/realworld/<Network>.pt`。

注意：当前训练主模型实际使用的是边和三角形信息；四面体、五面体会被缓存，但在 `model_main_v2.py` 的默认 `Hyper_GNN` 中暂未参与前向传播。

## 训练真实网络

当前推荐入口是：

```bash
python train_realworld_big.py
```

运行前需要在 `train_realworld_big.py` 底部修改数据集列表：

```python
List = ['Github_Contest']
```

脚本会从 `config_realworld.yaml` 中读取同名配置：

```python
config = yaml.load(open('config_realworld.yaml'), Loader=SafeLoader)[network]
```

因此，新增数据集时需要同时满足：

- `data/data/realworld/<Network>.txt` 存在。
- `data/data/realworld/<Network>.pt` 已经通过预处理生成。
- `config_realworld.yaml` 中存在 `<Network>` 对应配置。

训练输出目录格式为：

```text
log/DCRS/<Network>/<MM-DD>/debug-<MM-DD>_<HH.MM.SS>/
```

主要输出文件：

```text
<Network>.txt
<Network>_best_TAS.txt
```

其中 `<Network>.txt` 记录每个 epoch 的配置、loss、TAS 和早停信息；`<Network>_best_TAS.txt` 保存当前最优 epoch 下的节点攻击序列。

## 模型结构

主模型是 `model_main_v2.py` 中的 `Hyper_GNN`，由两个分支组成。

### 角色分支

实现文件：`Role_GCN.py`

输入包括节点特征、角色特征和节点-角色关联矩阵。模型先通过角色特征学习每个角色的权重，再构造加权角色超图传播矩阵，最后通过 GPR/APPNP 风格的多步传播得到角色增强后的节点表示。

### 高阶单纯形分支

实现文件：`HiGCN.py`

输入包括节点特征、普通边、三角形和它们对应的节点特征。该分支会：

- 使用标准邻接矩阵建模 0/1 阶节点邻接传播。
- 对边和三角形做顶点卷积，学习单纯形权重。
- 构造归一化高阶传播矩阵。
- 对不同阶结构分别传播，再通过权重融合得到高阶结构表示。

### 融合与打分

两个分支输出后会拼接：

```python
combined_ft = torch.cat((role_ft, simplex_ft), dim=1)
```

再经过 MLP 和 sigmoid 输出每个节点的拆解分数。分数越高，节点越优先被攻击。

## 损失函数与评估

训练使用 DCRS 风格的无监督损失：

```python
loss = loss1 + gamma * loss2
```

其中：

- `loss1` 基于邻接关系约束相邻节点的拆解影响。
- `loss2` 约束节点分数整体大小。
- `gamma` 来自 YAML 配置，用于平衡两项。

每个 epoch 中，脚本会按模型分数从高到低排序节点，并调用 `dismantling_fast_linear` 做线性时间拆解评估。默认阈值为 `0.01`，即最大连通分量降到原始节点数的 1% 以下时停止，记录所需攻击节点数量 TAS。

早停逻辑由 `utils.py` 中的 `EarlyStopping` 控制，默认 patience 为 `24`。




## 原始论文引用

如果使用原始 DCRS 代码或论文思想，请引用：

```bibtex
@article{zhou2026dismantling,
  title={Dismantling complex networks based on higher-order graph neural network},
  author={Zhou Wennan, Tan Suoyi, Fang Yang, Lu Xin and Zhao Xiang},
  journal={Communications Physics},
  year={2026},
  publisher={Nature Publishing Group UK London}
```
