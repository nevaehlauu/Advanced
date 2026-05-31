"""
搜索得到的Transformer_Encoder搭建的模型
"""

import torch.nn
import torch

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from model.genotypes import *
from model.attention_model import LayerNorm, Generator, PositionalEncoding
import torch.nn.functional as F
from model.senet import SENet18


class Cell(nn.Module):
    def __init__(self, genotype, d_model):
        super(Cell, self).__init__()
        # 直接将genotype中4个部分拿出来加入模型
        left_attention_name = genotype.self_attention[0][0][0]
        left_mlp_name = genotype.mlp[0][0][0]

        self.left_attention =  Attention_MAP[left_attention_name](d_model)
        self.left_mlp = FeedForward_MAP[left_mlp_name](d_model)
        self.bn = LayerNorm(d_model)
    
    def forward(self, x):
        left_ops = self.left_attention(x)
        x = self.bn(left_ops + x)
        left_ops = self.left_mlp(x)
        x = left_ops + x
        return self.bn(x)

class Network(nn.Module):
    def __init__(self, d_model, cell_num, num_classes, device, genotype):
        super(Network, self).__init__()
        self.cell_num = cell_num # 堆叠层数
        self.num_classes = num_classes

        # self.embedding = nn.Conv1d(in_channels=5, out_channels=d_model, kernel_size=8, stride=8) #senet每个patch的尺寸为12，因此这里滑动时以8为单位
        self.embedding = SENet18(in_channels=5, classes=8) # inputsize是输入的曲线条数，这里outsize不起作用，SENet将曲线映射到512维度，并且映射之后每个patch的尺寸为

        self.generate = Generator(d_model, vocab=num_classes)
        self.position_embed = PositionalEncoding(d_model, dropout=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pre_logits = nn.Identity()

        self.d_model = d_model
        self.device = device
        self.genotype = genotype

        self.cells = nn.ModuleList()
        for i in range(cell_num):
            cell = Cell(genotype, d_model)
            self.cells += [cell]
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(-1, -2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.position_embed(x)
        for i, cell in enumerate(self.cells):
            x = cell(x)
        x = self.pre_logits(x[:, 0])
        x = self.generate(x)
        return x