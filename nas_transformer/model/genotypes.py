import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import torch.nn as nn
from model.attention_model import MultiHeadedAttention, MultiHeadedAttention_qkv, FeedForward
from collections import namedtuple

Genotype = namedtuple('Genotype', 'self_attention mlp')

# 激活函数的选择：swish，leaky-relu，relu还有空
LEAKY_RELU_ACTIVATION_KEY = "leaky_relu"
NONE_ACTIVATION_KEY = "none"
RELU_ACTIVATION_KEY = "relu"
SWISH_ACTIVATION_KEY = "swish"


ACTIVATION_MAP = {
    SWISH_ACTIVATION_KEY: nn.SiLU(),
    LEAKY_RELU_ACTIVATION_KEY: nn.LeakyReLU(),
    RELU_ACTIVATION_KEY: nn.ReLU(),
    NONE_ACTIVATION_KEY: None,
}

# 是否需要搜索左右分支如何结合：相加还是在通道维度Cat，感觉相加更符合意思，不确定是否需要搜索，如果需要在nas_model.py中COMBINER_FUNCTIONS

# 直接搜索self_attention的话，搜索空间大小 4(num_head个数) * 3（1/2. 1. 2）* 3 = 36，搜索空间太大
# 这里可以由于内存限制，先搜索num_head，得到最好的num_head后，在搜索q, k, v维度，其中qk和v可以分别d_model / 2, d_model, d_model * 2
# Attention_MAP = {
#     # 'None': lambda d_model: Zero(), # 直接去掉这条分支
#     'skip_connection': lambda d_model: nn.Identity(), # 恒等映射，相当于跳跃连接操作
#     'MultiHeadedAttention_h1': lambda d_model: MultiHeadedAttention(1, d_model),
#     'MultiHeadedAttention_h2': lambda d_model: MultiHeadedAttention(2, d_model),
#     'MultiHeadedAttention_h4': lambda d_model: MultiHeadedAttention(4, d_model),
#     'MultiHeadedAttention_h4': lambda d_model: MultiHeadedAttention(8, d_model),
#     'MultiHeadedAttention_qkv_h2_22': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model / 2, d_model / 2), 
#     'MultiHeadedAttention_qkv_h2_22': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model / 2, d_model / 2), 
#     'MultiHeadedAttention_qkv_h4_22': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model / 2, d_model / 2), 
#     'MultiHeadedAttention_qkv_h8_22': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model / 2, d_model / 2), 
#     'MultiHeadedAttention_qkv_h2_21': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model / 2, d_model), 
#     'MultiHeadedAttention_qkv_h2_21': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model / 2, d_model), 
#     'MultiHeadedAttention_qkv_h4_21': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model / 2, d_model), 
#     'MultiHeadedAttention_qkv_h8_21': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model / 2, d_model),
#     'MultiHeadedAttention_qkv_h2_2_2': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model / 2, d_model * 2), 
#     'MultiHeadedAttention_qkv_h2_2_2': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model / 2, d_model * 2), 
#     'MultiHeadedAttention_qkv_h4_2_2': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model / 2, d_model * 2), 
#     'MultiHeadedAttention_qkv_h8_2_2': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model / 2, d_model * 2), 
#     'MultiHeadedAttention_qkv_h2_12': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model, d_model / 2), 
#     'MultiHeadedAttention_qkv_h2_12': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model, d_model / 2), 
#     'MultiHeadedAttention_qkv_h4_12': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model, d_model / 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model, d_model / 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model, d_model * 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model, d_model * 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model, d_model * 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model, d_model * 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model * 2, d_model * 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model * 2, d_model * 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model * 2, d_model * 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model * 2, d_model * 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model * 2, d_model / 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model * 2, d_model / 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model * 2, d_model / 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model * 2, d_model / 2), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model * 2, d_model), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model * 2, d_model), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model * 2, d_model), 
#     'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model * 2, d_model), 
# }

# 是否需要搜索左右分支如何结合：相加还是在通道维度Cat，感觉相加更符合意思，不确定是否需要搜索，如果需要在nas_model.py中COMBINER_FUNCTIONS
Attention_MAP_1 = {
    # 'None': lambda d_model: Zero(), # 直接去掉这条分支
    'skip_connection': lambda d_model: nn.Identity(), # 恒等映射，相当于跳跃连接操作
    'MultiHeadedAttention_h2': lambda d_model: MultiHeadedAttention(2, d_model),
    'MultiHeadedAttention_h4': lambda d_model: MultiHeadedAttention(4, d_model),
    'MultiHeadedAttention_h8': lambda d_model: MultiHeadedAttention(8, d_model),
}

# Attention_OPERATION = [
#     'skip_connection',
#     'MultiHeadedAttention_h2',
#     'MultiHeadedAttention_h4',
#     'MultiHeadedAttention_h8',
#     'MultiHeadedAttention_qkv_h2_22',
#     'MultiHeadedAttention_qkv_h4_22',
#     'MultiHeadedAttention_qkv_h8_22',
#     'MultiHeadedAttention_qkv_h2_21',
#     'MultiHeadedAttention_qkv_h4_21',
#     'MultiHeadedAttention_qkv_h8_21',
#     'MultiHeadedAttention_qkv_h2_2_2',
#     'MultiHeadedAttention_qkv_h4_2_2',
#     'MultiHeadedAttention_qkv_h8_2_2',
#     'MultiHeadedAttention_qkv_h2_12',
#     'MultiHeadedAttention_qkv_h4_12',
#     'MultiHeadedAttention_qkv_h8_12'
# ]

"""
q-k-v维度保持相同的情况
"""
Attention_MAP = {
    # 'None': lambda d_model: Zero(), # 直接去掉这条分支
    # 'skip_connection': lambda d_model: nn.Identity(), # 恒等映射，相当于跳跃连接操作
    'MultiHeadedAttention_h1': lambda d_model: MultiHeadedAttention(1, d_model),
    'MultiHeadedAttention_h2': lambda d_model: MultiHeadedAttention(2, d_model),
    'MultiHeadedAttention_h4': lambda d_model: MultiHeadedAttention(4, d_model),
    'MultiHeadedAttention_h8': lambda d_model: MultiHeadedAttention(8, d_model),
    'MultiHeadedAttention_qkv_h1_12': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model / 2, d_model / 2), 
    'MultiHeadedAttention_qkv_h2_12': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model / 2, d_model / 2), 
    'MultiHeadedAttention_qkv_h4_12': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model / 2, d_model / 2), 
    'MultiHeadedAttention_qkv_h8_12': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model / 2, d_model / 2), 
    'MultiHeadedAttention_qkv_h1_22': lambda d_model: MultiHeadedAttention_qkv(1, d_model, d_model * 2, d_model * 2), 
    'MultiHeadedAttention_qkv_h2_22': lambda d_model: MultiHeadedAttention_qkv(2, d_model, d_model * 2, d_model * 2), 
    'MultiHeadedAttention_qkv_h4_22': lambda d_model: MultiHeadedAttention_qkv(4, d_model, d_model * 2, d_model * 2), 
    'MultiHeadedAttention_qkv_h8_22': lambda d_model: MultiHeadedAttention_qkv(8, d_model, d_model * 2, d_model * 2), 
}

Attention_OPERATION = [
    # 'skip_connection',
    'MultiHeadedAttention_h1',
    'MultiHeadedAttention_h2',
    'MultiHeadedAttention_h4',
    'MultiHeadedAttention_h8',
    'MultiHeadedAttention_qkv_h1_12',
    'MultiHeadedAttention_qkv_h2_12',
    'MultiHeadedAttention_qkv_h4_12',
    'MultiHeadedAttention_qkv_h8_12',
    'MultiHeadedAttention_qkv_h1_22',
    'MultiHeadedAttention_qkv_h2_22',
    'MultiHeadedAttention_qkv_h4_22',
    'MultiHeadedAttention_qkv_h8_22'
]


# 这里可以先搜索完MPL_radio，再搜索激活函数
FeedForward_MAP = {
    # 'None': lambda d_model: Zero(), # 直接去掉这条分支
    # 'skip_connection': lambda d_model: nn.Identity(), # 恒等映射，相当于跳跃连接操作
    'FeedForward_ratio_2': lambda d_model: FeedForward(d_model, 2),
    'FeedForward_ratio_2.5': lambda d_model: FeedForward(d_model, 2.5),
    'FeedForward_ratio_3': lambda d_model: FeedForward(d_model, 3),
    'FeedForward_ratio_3.5': lambda d_model: FeedForward(d_model, 3.5),
    'FeedForward_ratio_4': lambda d_model: FeedForward(d_model, 4),
    'FeedForward_ratio_8': lambda d_model: FeedForward(d_model, 8),
}

FeedForward_OPERATION = [
    # 'skip_connection',
    'FeedForward_ratio_2',
    'FeedForward_ratio_2.5',
    'FeedForward_ratio_3',
    'FeedForward_ratio_3.5',
    'FeedForward_ratio_4',
    'FeedForward_ratio_8'
]


###### 没有senet
# 92口井情况下
#Transformer_Encoder = Genotype(self_attention=[[('MultiHeadedAttention_qkv_h4_22', 11)]], mlp=[[('FeedForward_ratio_8', 6)]])
# Transformer_Encoder = Genotype(self_attention=[[('MultiHeadedAttention_h4', 2)]], mlp=[[('FeedForward_ratio_4', 4)]])

# 92口井，cell=8
Transformer_Encoder = Genotype(self_attention=[[('MultiHeadedAttention_qkv_h2_22', 9)]], mlp=[[('FeedForward_ratio_2.5', 1)]])

# 50口井情况下

