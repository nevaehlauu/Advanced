from typing import Union
from collections import OrderedDict
import torch
import torch.nn as nn
from pathlib import Path
import json
import os
import random
import sys
import numpy as np
from enum import Enum
import shutil
import logging
import time
import glob

def load_model_params(net, loaded_model: Union[str, OrderedDict] = None):
    """
    这个有点小问题，先放着
    :param net: 网络
    :param loaded_model: 文件或者是，预加载参数
    :return: 无需返回
    """
    # 为空，抛弃
    assert loaded_model is not None, "loaded_model 为 None"

    # 若为文件路径，下载并加载
    if isinstance(loaded_model, str):
        assert Path(loaded_model).exists(), "该权重文件不存在"
        loaded_model = torch.load(loaded_model, map_location=torch.device("cpu"))

    net_dict = net.state_dict()

    # 直接判断同名model的尺寸是否一样就可以了，不一样的不加载
    pretrained_dict = {k: v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}

    net_dict.update(pretrained_dict)  # 更新一下。。
    net.load_state_dict(net_dict, strict=False)
    return net

def get_normalization_1d(input: np.ndarray, a=-1, b=1):
    """
    input 1维的  xxx*1也行
    归一化到 a，b 之间
    """
    if len(input) == 0:
        return input

    value_max = input.max()
    value_min = input.min()

    k = (b - a) / (value_max - value_min + 1e-5)

    output = a + k * (input - value_min)
    return output

def set_seeds(seed=43):
    """
    Set Python random seeding and PyTorch seeds.
    固定随机数种子

    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 哈希随机初始化
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
    torch.use_deterministic_algorithms(True)  # 检测是否用了非原子性算法

def read_json(json_cfg_filepath):
    with open(json_cfg_filepath, encoding="utf-8") as f:
        content = json.load(f)
    return content

def sample_to_device(data, device="cpu"):
    """
    将数据放到指定设备中的函数
    :param data: 原始数据
    :param device: "cuda" 或者 "cpu"
    :return:
    """
    """
    将数据 放到cuda 里
    """
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        data_cuda = {}
        for key in data.keys():
            data_cuda[key] = sample_to_device(data[key], device)
        return data_cuda
    elif isinstance(data, list):
        data_cuda = []
        for key in data:
            data_cuda.append(sample_to_device(key, device))
        return data_cuda
    else:
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        return data.to(device)
    
def save_fig(values, title, fig_path):
    # 使用agg而不是默认的qt5agg
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    t = np.arange(0, len(values))
    plt.plot(t, values)
    plt.title(title)
    plt.savefig(fig_path)
    plt.close()

def get_features_tokenizer(tokenizer_pretrained_dict, model, device="cpu"):
    
    ####### 将预训练的vqvae加载到transformer的embedding部分（这里是vqvae里面encoder+codebook部分）
    for key, value in tokenizer_pretrained_dict.items():
        # 加载模型
        loaded_model = torch.load(value, map_location=torch.device("cpu"))['state_dict']
        model = load_model_params(model, loaded_model)
        tokenizer_dict[key] = model # 将模型实例存储在字典中，键为曲线名
    tokenizer_dict = nn.ModuleDict(tokenizer_dict)
    if device is not "cpu":
        tokenizer_dict = tokenizer_dict.to(device)
    return tokenizer_dict

class Bcolors(Enum):
    """
    颜色枚举体定义，和shell脚本的一致

    echo -e "\e[90m 黑底黑字 \e[0m"
    echo -e "\e[91m 黑底红字 \e[0m"
    echo -e "\e[92m 黑底绿字 \e[0m"
    echo -e "\e[93m 黑底黄字 \e[0m"
    echo -e "\e[94m 黑底蓝字 \e[0m"
    echo -e "\e[95m 黑底紫字 \e[0m"
    echo -e "\e[96m 黑底青字 \e[0m"
    echo -e "\e[97m 黑底白字 \e[0m"
    """

    HEADER = '\033[95m'  # 紫色
    OKBLUE = '\033[94m'  # 蓝色 --> 两个交替着来
    OKGREEN = '\033[92m'  # 绿色 --> 两个交替着来
    WARNING = '\033[93m'  # 黄色 --> 警告
    TIPS = '\033[96m'  # 青色 --> 提示(这个太亮了)
    FAIL = '\033[91m'  # 深红 --> 失败
    ENDC = '\033[0m'  # 关闭所有属性，即属性结束标识
    BOLD = '\033[1m'  # 设置高亮度，即字体加粗、文体强调
    UNDERLINE = '\033[4m'  # 下划线


def printcolor(message, color=Bcolors.ENDC):
    """
    打印彩色内容，规则和print一样，就是颜色是彩色的
    Print a message in a certain color (only rank 0)
    grey, red, green, yellow, blue, magenta, cyan, white.
    """

    # print(colored(message, color))
    # 因为 termcolor 没用，所以我自己弄了个
    if color not in Bcolors:
        color = Bcolors.ENDC
    print(color.value + str(message) + Bcolors.ENDC.value)


def print_block(message, title='', color=Bcolors.ENDC):
    """
    打印区块信息，就是加个title
    :param message:
    :param title:
    :param color:
    :return:
    """
    print('')
    printcolor('-' * 25 + ' ' + title + ' ' + '-' * 25, color)
    printcolor(message, color)
    print('')

def get_classes_map(desc_filepath: Path):
    """
    从数据集描述文件中获取标签的映射关系
    :param desc_filepath: 数据集描述文件路径
    :return: output，key: 类别名，value: 类别id
    """
    output = {}
    data_desc = read_json(desc_filepath)
    for cur_label in data_desc["label"]:
        cur_classes = [int(i) for i in cur_label["vallist"][1]]
        cur_classes_name = cur_label["vallist"][0]
        output[cur_label["name"]] = dict(zip(cur_classes, cur_classes_name)) #将两个列表合成一个字典
    return output

# 统计参数量M
def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

# 创建文件夹，copy的一些操作
def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

def save_train_val_fig(train_values, val_values, train_label, val_label, title, x_title, y_title, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    x = np.arange(0, max(len(train_values), len(val_values)))
    plt.plot(x, train_values, label=train_label, linewidth=1.5)
    plt.plot(x, val_values, label=val_label, linewidth=1.5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_title, fontsize=14, fontweight='bold')
    plt.ylabel(y_title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc=0, numpoints=1)
    plt.savefig(fig_path)
    plt.close()

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

# 随机丢弃路径，来自FractalNet，至少保证有一条路径是连接输入和输出的
def drop_path(x, drop_prob):
  # if drop_prob > 0.:
  #   keep_prob = 1.-drop_prob
  #   # 从伯努利分布中抽取0或1,1的概率为keep_prob，返回的mask是个0/1的tensor，其中1的比例约为keep_prob
  #   mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob))
  #   # 没有丢弃之前的期望为E[x]，加入drop path之后的期望为p*E[x]，需要除以p保持期望不变
  #   x.div_(keep_prob)
  #   x.mul_(mask)
  return x

# 创建文件夹，copy的一些操作
def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def save_log(title, args):
    """
    保存一下文件信息
    """
    args.save = '{}-{}-{}-{}'.format(title, args.save, args.classification_name, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return args.save