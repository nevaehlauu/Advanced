"""
对比预测编码: 编码器+自回归模型
这里采用的编码器模块可以直接是ResNet，也可以是NAS搜索得到的卷积神经网络模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
from pathlib import Path
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(str(Path("./").resolve()))
# from resnet import Net
from data_228.network import Net_1d as ResNet
from darts.model_search import Network
from darts.model import NetworkCIFAR


class CPCModel(nn.Module):
    """
    resnet+LSTM的对比预测编码器
    """
    def __init__(self, in_channels, code_size, anchor_num, is_nas, layers, genotype):
        super(CPCModel, self).__init__()
        if is_nas:
            self.encoder = NetworkCIFAR(in_channels, code_size, layers, genotype)
        else:
            self.encoder = ResNet(in_channels, code_size)
        self.lstm = network_regressive(input_dim=anchor_num, hidden_dim=code_size, num_layers=1) # 这里5是因为选择5个连续的切片标签相同的切片作为锚定样本
    
    def forward(self, anchor_input, pos_input, neg_input):
        # 将锚定样本数据中5个样本分别编码，并将编码拼接输入LSTM网络去提取序列关系，其中，anchor_input维度[batchsize, anchor_num, 6, slice_len]，6为曲线条数
        anchor_encodeds = []
        for i in range(anchor_input.size(1)):
            anchor_t = anchor_input[:, i, :, :] # 获取一个锚定标签, batchsize * 6 * slice_len
            anchor_encoder = self.encoder(anchor_t) # resnet输出 batchsize * code_size
            anchor_encoder = anchor_encoder.unsqueeze(dim=1)
            anchor_encodeds.append(anchor_encoder) 
        anchor_encodeds = torch.cat(anchor_encodeds, dim=1) # 还是[batchsize, anchor_num, code_size]
        # lstm输入维度为N*L*H，其中H为通道数

        anchor_encodeds = anchor_encodeds.permute(0, 2, 1).contiguous() # [batchsize, code_size, anchor_num]
        preds = self.lstm(anchor_encodeds) # (batchsize, code_size)
        self.anchor_latent = preds

        # 负样本编码
        neg_encoders = []
        for i in range(neg_input.size(1)):
            neg_t = neg_input[:, i, :, :] 
            neg_encoder = self.encoder(neg_t)
            neg_encoders.append(neg_encoder.unsqueeze(dim=1)) # batchsize * code_size
        neg_encoders = torch.cat(neg_encoders, dim=1) # batchsize * 9 * code_size
        self.neg_latent = neg_encoders

        # 正样本编码
        pos_t = pos_input[:, 0, :, :]
        pos_encoders = self.encoder(pos_t) # batchsize * code_size
        self.pos_latent = pos_encoders

        return preds, pos_encoders, neg_encoders

    def get_feature(self):
        return self.anchor_latent, self.pos_latent, self.neg_latent
    
class ContrastiveModel(nn.Module):
    """
    resnet+LSTM的对比预测编码器
    """
    def __init__(self, in_channels, code_size, anchor_num, is_nas, layers, genotype):
        super(ContrastiveModel, self).__init__()
        if is_nas:
            self.encoder = NetworkCIFAR(in_channels, code_size, layers, genotype)
        else:
            self.encoder = ResNet(in_channels, code_size)
    
    def forward(self, anchor_input, pos_input, neg_input):
        # 将锚定样本数据中5个样本分别编码，并将编码拼接输入LSTM网络去提取序列关系，其中，anchor_input维度[batchsize, anchor_num, 6, slice_len]，6为曲线条数
        anchor_encodeds = []
        for i in range(anchor_input.size(1)):
            anchor_t = anchor_input[:, i, :, :] # 获取一个锚定标签, batchsize * 6 * slice_len
            anchor_encoder = self.encoder(anchor_t) # resnet输出 batchsize * code_size
            anchor_encoder = anchor_encoder.unsqueeze(dim=1)
            anchor_encodeds.append(anchor_encoder) 
        anchor_encodeds = torch.cat(anchor_encodeds, dim=1) # 还是[batchsize, anchor_num, code_size]
        # lstm输入维度为N*L*H，其中H为通道数

        anchor_encodeds = anchor_encodeds.permute(0, 2, 1).contiguous() # [batchsize, code_size, anchor_num]
        preds = self.lstm(anchor_encodeds) # (batchsize, code_size)
        self.anchor_latent = preds

        # 负样本编码
        neg_encoders = []
        for i in range(neg_input.size(1)):
            neg_t = neg_input[:, i, :, :] 
            neg_encoder = self.encoder(neg_t)
            neg_encoders.append(neg_encoder.unsqueeze(dim=1)) # batchsize * code_size
        neg_encoders = torch.cat(neg_encoders, dim=1) # batchsize * 9 * code_size
        self.neg_latent = neg_encoders

        # 正样本编码
        pos_t = pos_input[:, 0, :, :]
        pos_encoders = self.encoder(pos_t) # batchsize * code_size
        self.pos_latent = pos_encoders

        return preds, pos_encoders, neg_encoders

    def get_feature(self):
        return self.anchor_latent, self.pos_latent, self.neg_latent
        

class network_regressive(nn.Module):
    """
    双向LSTM网络
    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(network_regressive, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
    
    def forward(self, x):
        x, _ = self.lstm(x, None)
        # x: batchsize * seq_len * hidden_size
        # 取LSTM最后一个hidden state，锚定样本经过LSTM预测后只需要保存最后一个时间步的输出，然后和正样本、负样本相比较
        x = x[:, -1, :]
        return x

class InfoNCELoss(nn.Module):
    """
    计算CPC中InfoNCELoss损失
    参数：
        anchor：锚点样本特征张量
        positive: 正样本特征张量
        negatives: 负样本特征张量
    返回：
        InfoNCE损失平均值
    """
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        # 对比学习中温度参数用于调整正样本分数的范围，temperature较大时，正样本分数分散，差异较大，较小时正样本分布更加集中，差异较小
        self.temperature = temperature
    
    def forward(self, anchor, positive, negatives):
        """
        anchor: [batch_size, hidden_dim] --> [batch_size, code_size]
        positive: [batch_size, code_size]
        negatives: [batch_size, N, code_size]
        公式分析：
            https://blog.csdn.net/weixin_47187147/article/details/136435884
        """
        # 标准化特征向量, p=2表示使用L2范数进行归一化（欧几里得距离）,对于一个形状为 (batch_size, feature_dim) 的张量 anchor，经过 F.normalize(anchor, p=2, dim=1) 之后，每个样本的特征向量都会被归一化为单位长度。
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives,  p=2, dim=1)

        # 正样本分数计算
        positive_score = torch.sum(anchor * positive, dim=-1) / self.temperature # 元素相乘得到[batch_size, code_size]的张量，torch.sum得到batchsize大小的张量，
        positive_score = positive_score.unsqueeze(dim=1) # 为了与负样本对齐，增加一个维度 --> [batch_size, 1]

        # 负样本分数计算
        negatives_score = torch.bmm(negatives, anchor.unsqueeze(2)) # 利用矩阵乘法计算，negatives:[batch_size, N, code_size] anchor: [batchsize, code_size, 1] --> [batchsize, N, 1]
        negatives_score = negatives_score.squeeze(dim=2) / self.temperature # [batchsize, N]

        logits = torch.cat((positive_score, negatives_score), dim=1) # [batchsize, 1+N]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device) # 所有元素初始化为 0,表示正样本

        return F.cross_entropy(logits, labels)
         

class Encoder_classifier(nn.Module):
    """
    ResNet+分类头
    """
    def __init__(self, in_channels, code_size, classes, is_nas, layers, genotype):
        super(Encoder_classifier, self).__init__()
        if is_nas:
            self.encoder = NetworkCIFAR(in_channels, code_size, layers, genotype)
        else:
            self.encoder = ResNet(in_channels=in_channels, classes=code_size)
        
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        
        self.fc = nn.Linear(code_size, classes)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x