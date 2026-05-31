"""
关系网络的度量模块：用于计算两个拼接在一起的特征的相关性系数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationNetwork(nn.Module):
    """
    将支持集和查询集特征拼接之后计算相似度，这里先经过两层卷积，在通过Linear输出相似度
    """
    def __init__(self, in_channels,  out_channels):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(), # 输入特征拼接
        self.fc2 = nn.Linear(), # 输出一个关系分数
    
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out