import torch
import torch.nn as nn

import os
import sys
from pathlib import Path
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(str(Path("./").resolve()))

from darts.operations import *
from torch.autograd import Variable

def drop_path(x, drop_prob):
  return x

class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    #对输入节点定义
    if reduction_prev: 
      self.preprocess0 = FactorizedReduce(C_prev_prev, C) #进行因式化降采样
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0) #完成一个1*1的激活+卷积+BN操作
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce) #对这个列表进行解压，op_names为操作名称，表示在某个特定位置要执行操作名称，indices为索引值，表示特定位置上要操作的输入张量索引
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  # def forward(self, s0, s1, drop_prob):
  def forward(self, s0, s1):
  
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      # if self.training and drop_prob > 0.:
      #   if not isinstance(op1, Identity):
      #     h1 = drop_path(h1, drop_prob)
      #   if not isinstance(op2, Identity):
      #     h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, genotype): 
    super(NetworkCIFAR, self).__init__()
    self._layers = layers #网络层数

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential( #in_channer=3,out_channel=3*C
      nn.Conv1d(C, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm1d(C_curr)
    )
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C #前两个为3*C，C_curr为C
    self.cells = nn.ModuleList() #用于存储和管理多个nn.Module模块
    reduction_prev = False
    for i in range(layers):
      # 通道尺寸变化，每个Cell后通道都*2
      C_curr *= 2
      reduction = False
      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
    self.global_pooling = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      # s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      s0, s1 = s1, cell(s0, s1)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits
