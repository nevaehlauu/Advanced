"""
调用该库，进行配置的加载
使用 parse_train_file 解析配置
"""
from pathlib import Path

import torch
import sys
from yacs.config import CfgNode

from configs import base_config

from copy import deepcopy


def cfg2dict(cfg: CfgNode):
    """
    不对原cfg造成任何影响
    @param cfg:
    @return:
    """
    output = {}
    for key in cfg.keys():
        if isinstance(cfg[key], CfgNode):
            output[key] = cfg2dict(cfg[key])
        else:
            output[key] = deepcopy(cfg[key])

    return output


def parse_train_file(cfg_file: str, cfg_default: CfgNode):
    """
    解析训练参数，要求 cfg_file 一定为 .yaml 文件
    :param cfg_file: 配置文件 .yaml
    :param cfg_default: CfgNode 配置类
    :return:
    """
    if cfg_file is not None and cfg_file.endswith('yaml') and Path(cfg_file).exists():
        cfg_default.merge_from_file(cfg_file)  # 合并文件 cfg_file 中的参数到 cfg_default 中
        cfg_default.merge_from_list(['configs', cfg_file])  # 更新 cfg_default 中的配置字段为 cfg_file
        return cfg_default
    else:
        raise ValueError('You need to provide a .yaml to train')


def get_train_cfg(cfg_file: str):
    cfg = parse_train_file(cfg_file, base_config.get_cfg_defaults())
    # 检测能否使用cpu
    cfg.arch.device = cfg.arch.device if torch.cuda.is_available() else 'cpu'
    # 获取平台信息，windows平台进程数限制为0
    cfg.arch.platform = sys.platform
    cfg.datasets.train.num_workers = cfg.datasets.train.num_workers if sys.platform.startswith('linux') else 0
    cfg.datasets.val.num_workers = cfg.datasets.train.num_workers if sys.platform.startswith('linux') else 0
    return cfg
