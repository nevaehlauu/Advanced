# 测井数据深度学习实验代码

本仓库是一组面向测井序列分类任务的研究代码，主要围绕数据增强、神经网络架构搜索（NAS）、Transformer、原型网络（ProtoNet）和领域自适应（DANN）展开。代码以一维测井曲线切片为输入，完成地层或储层相关分类实验，并提供训练日志、模型保存、混淆矩阵和 t-SNE 可视化等辅助能力。

仓库中包含部分预训练权重和实验图片，但不包含完整数据集。代码和 YAML 中的默认文件路径已经统一改为相对路径。运行前需要先按下文目录结构准备本地数据。

## 目录结构

```text
code/
├── assets/
│   └── fonts/                # 可选：绘图脚本使用的中文字体
├── data/                     # 本地数据目录，不随仓库提供
│   ├── h5/                   # HDF5 数据和 param.json
│   ├── well_228_old/         # 按井拆分的 train/、test/ 文本数据
│   └── well_data/            # 其他测井文本数据
├── data_augmentation/         # 数据增强、HDF5 数据训练、CPC、DARTS 和辅助分析
├── nas_transformer/           # CNN/Transformer 基线、Transformer NAS 搜索与搜索后训练
├── protonet_word_embedding/   # ProtoNet、标签词嵌入修正、测试和混淆矩阵
├── transfer_learning/         # DANN 领域自适应训练与 t-SNE 可视化
└── README.md
```

四个目录是相对独立的实验工程。部分脚本依赖当前工作目录和相对导入，建议先进入对应模块目录，再执行脚本。

## 环境配置

### 1. 基础环境

建议使用 Linux、NVIDIA GPU 和 CUDA 版本的 PyTorch。多数主要训练脚本直接构造 `cuda:<gpu_id>` 设备，未提供完整的 CPU 回退逻辑。Windows 可以用于阅读代码和部分 HDF5 流程，但完整训练更适合在 Linux 环境运行。

代码未锁定 Python 和依赖版本。建议使用 Python 3.9 或 3.10 创建隔离环境：

```bash
conda create -n well-log python=3.10 -y
conda activate well-log
```

### 2. 安装 PyTorch

请根据本机 CUDA 版本，从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择安装命令。例如：

```bash
pip install torch torchvision
```

安装后检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3. 安装其余依赖

仓库没有提供 `requirements.txt`。根据全部源码中的导入，可安装：

```bash
pip install numpy pandas scipy scikit-learn matplotlib PyWavelets h5py PyYAML yacs tqdm tsaug altair torchsummary
```

说明：

- `torchvision` 主要用于 DARTS 相关脚本。
- `h5py`、`yacs` 和 `PyYAML` 用于 HDF5 数据与 YAML 配置流程。
- `tsaug`、`altair`、`torchsummary` 只在部分辅助或实验脚本中使用。
- `tkinter` 属于 Python 图形界面组件；如果系统未自带，请通过操作系统包管理器安装。

## 数据准备

仓库支持两类数据读取方式。

### TXT/CSV 风格测井文件

大部分基线、NAS、ProtoNet 和 DANN 脚本读取按井拆分的文本文件。文件由 `pandas.read_csv(..., header=None)` 加载，因此实际内容应是逗号分隔的表格。训练集和验证集建议分别放置：

```text
code/data/well_228_old/
├── train/
│   ├── W1424.txt
│   └── ...
└── test/
    ├── W1615.txt
    └── ...
```

不同读取器对列的选择略有差异。例如常用读取器会从表格中提取测井曲线列和标签列，再进行标准化、滑动窗口切片与可选的数据增强。井名筛选列表直接写在各模块的 `dataset.py`、`dataloader.py` 或 `readtxt_*.py` 中，因此本地文件名需要与列表匹配。

### HDF5 数据

`data_augmentation/data_h5/` 使用 `.h5` 数据和描述文件 `param.json`。训练和验证文件路径配置在：

```text
data_augmentation/config/training_data_config.yaml
```

至少需要修改：

```yaml
model:
  log_dir_path: "log/data_h5/data_aug_log"

datasets:
  train:
    filepath: "../data/h5/train.h5"
    desc_filepath: "../data/h5/param.json"
  val:
    filepath: "../data/h5/val.h5"
    desc_filepath: "../data/h5/param.json"
```

### 相对路径约定

代码中的默认路径以“先进入模块目录，再执行脚本”为基准。例如执行 `data_augmentation` 实验时：

```bash
cd data_augmentation
python data_228/train.py
```

常用相对目录：

| 路径 | 用途 |
| --- | --- |
| `../data/well_228_old/train/` | TXT 训练集 |
| `../data/well_228_old/test/` | TXT 验证集 |
| `../data/h5/` | HDF5 数据与 `param.json` |
| `pretrain_model/` | 当前模块的预训练权重 |
| `log/` | 当前模块的训练输出 |
| `../assets/fonts/` | 可选中文字体 |

如果使用不同的数据目录，可以通过命令行参数或 YAML 配置覆盖默认值。

## 模块说明

### `data_augmentation`

用于比较不同数据增强策略以及训练基础分类器。

主要内容：

- `data_228/`：基于 TXT 数据的 1D CNN/SENet 训练与数据读取。
- `data_h5/`：基于 HDF5 数据的 ResNet 训练流程。
- `cpc/`：对比预测编码（CPC）预训练与分类器训练。
- `darts/`：DARTS 搜索和搜索后网络训练。
- `ClassNumber/`、`pic_well/`：类别统计、聚类和绘图辅助脚本。

常用命令：

```bash
cd data_augmentation

# TXT 数据基础训练
python data_228/train.py \
  --train_dir ../data/well_228_old/train \
  --val_dir ../data/well_228_old/test \
  --gpu_id 0

# HDF5 数据训练
python data_h5/train_well.py \
  --model_config_file config/training_data_config.yaml

# CPC 预训练与分类器训练
python cpc/cpc_train.py --train_dir ../data/well_228_old/train --gpu_id 0
python cpc/classifier_train.py --train_dir ../data/well_228_old/train --val_dir ../data/well_228_old/test --gpu_id 0

# DARTS 搜索与搜索后训练
python darts/train_search.py --train_dir ../data/well_228_old/train --val_dir ../data/well_228_old/test --gpu 0
python darts/train.py --train_dir ../data/well_228_old/train --val_dir ../data/well_228_old/test --gpu 0
```

### `nas_transformer`

用于训练 CNN/SENet 基线、Transformer 基线，并搜索 Transformer 编码器中的结构参数。

主要入口：

- `train_cnn.py`：SENet/CNN 基线。
- `train_transformer.py`：Transformer 基线。
- `train_transformer_search.py`：Transformer 架构搜索。
- `train_transformer_searched.py`：使用搜索得到的结构继续训练。
- `test.py`、`tsne_feature.py`、`tsne_cluster.py`：测试和可视化。

运行示例：

```bash
cd nas_transformer

python train_cnn.py --train_dir ../data/well_228_old/train --val_dir ../data/well_228_old/test --gpu_id 0
python train_transformer.py --train_dir ../data/well_228_old/train --val_dir ../data/well_228_old/test --gpu_id 0
python train_transformer_search.py --train_dir ../data/well_228_old/train --val_dir ../data/well_228_old/test --gpu 0
python train_transformer_searched.py --train_dir ../data/well_228_old/train --val_dir ../data/well_228_old/test --gpu_id 0
```

搜索后的固定结构定义在 `nas_transformer/model/genotypes.py` 中。替换搜索结果时，需要同步更新该文件中的 genotype。

### `protonet_word_embedding`

用于少样本分类实验。每个 episode 按 `N-way K-shot` 形式采样 support 和 query 样本，再通过原型距离完成分类。

主要入口：

- `train_protonet.py`：标准 ProtoNet。
- `train_protonet_label.py`：使用标签词嵌入修正类别原型。
- `test_protonet.py`：测试。
- `confusion_matrix.py`、`tsne_well.py`：结果分析。

运行示例：

```bash
cd protonet_word_embedding

python train_protonet.py \
  --train_dir ../data/well_228_old/train \
  --val_dir ../data/well_228_old/test \
  --gpu_id 0 \
  --n_cls 10 \
  --support 10 \
  --query 2

python train_protonet_label.py \
  --train_dir ../data/well_228_old/train \
  --val_dir ../data/well_228_old/test \
  --gpu_id 0
```

仓库已包含部分 SENet 预训练权重：

```text
protonet_word_embedding/pretrain_model/
```

命令行默认值仍指向原作者机器路径，使用仓库内权重时请显式传入 `--pretrained_filepath`。

### `transfer_learning`

用于基于 DANN 的领域自适应训练。模型由特征提取器、类别分类器和域分类器组成，训练时同时优化源域分类损失与域判别损失。

主要入口：

- `dann_train.py`：DANN 训练。
- `tsne_well.py`：特征可视化。
- `lmmd.py`：LMMD 损失实现，目前未接入主训练入口。

运行示例：

```bash
cd transfer_learning

python dann_train.py \
  --train_dir ../data/well_228_old/train \
  --val_dir ../data/well_228_old/test \
  --gpu_id 0 \
  --src_categorize_id 2 \
  --tgt_categorize_id 1
```

仓库已包含部分预训练权重：

```text
transfer_learning/pretrain_model/
```

## 常用参数

多数 TXT 数据训练脚本共享以下参数：

| 参数 | 说明 |
| --- | --- |
| `--train_dir` | 训练井文件目录 |
| `--val_dir` | 验证井文件目录 |
| `--gpu_id` 或 `--gpu` | CUDA 设备编号 |
| `--epochs` | 训练轮数 |
| `--batchsize` | 批大小 |
| `--slice_length` | 测井序列切片长度 |
| `--slice_step` | 滑动窗口步长 |
| `--train_well_num` | 训练使用的井数量 |
| `--val_well_num` | 验证使用的井数量 |
| `--frequency_aug` | 频域或小波增强方式 |
| `--noise_ration` | 高斯噪声幅度；代码中参数名保留了原拼写 |
| `--pretrained_filepath` | 预训练权重路径 |

ProtoNet 额外使用 `--n_batch`、`--n_cls`、`--support` 和 `--query` 控制 episode 采样。DANN 使用 `--src_*` 与 `--tgt_*` 参数分别设置源域和目标域。

## 输出文件

训练脚本通常会在 `log_dir_path` 下创建带时间戳的目录，并保存：

```text
best_epoch_model.pth       # 验证集表现最好的模型参数
logging.json               # loss、accuracy 等训练记录
train_val_acc.png          # 准确率曲线
train_val_loss.png         # 损失曲线
confusion_matrix.png       # 部分流程生成
```

HDF5 流程保存的 checkpoint 后缀为 `.ckpt`，其中可包含模型对象、`state_dict` 和配置。

## 已知注意事项

- 多数训练脚本默认使用 CUDA；没有 GPU 时需要自行补充 CPU 兼容逻辑。
- 部分中文注释和 YAML 文本存在编码错位，但 Python 文件仍可解析。
- 相对路径默认以模块目录作为当前工作目录；从仓库根目录直接执行时需要同步调整路径。
- 不同实验对输入通道数要求不同，常见值为 5、6 或 7。修改数据列后要同步调整 `--in_channel` 或 YAML 中的 `features_name`。
- `data_augmentation/data_h5/train_well.py` 默认配置参数中的相对路径以执行时的工作目录为基准，推荐显式传入 `--model_config_file`。
- 辅助分析脚本中有部分历史实验代码，不一定能在新数据集上直接运行。

## 推荐上手顺序

1. 准备 `train/` 和 `test/` 目录中的测井文本文件。
2. 进入 `data_augmentation/`，运行 `data_228/train.py` 验证数据读取和基础分类流程。
3. 使用 `nas_transformer/train_cnn.py` 或 `nas_transformer/train_transformer.py` 训练基线。
4. 根据实验目标继续运行 Transformer NAS、ProtoNet 或 DANN。
5. 使用各模块中的 t-SNE、混淆矩阵和绘图脚本分析结果。
