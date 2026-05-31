import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class protoNet(nn.Module):
    """
    原型网络，encoder为模型编码器，dot是距离计算方式
    """
    def __init__(self, encoder, dot=False):
        super(protoNet, self).__init__()
        self.encoder = encoder
        self.dot = dot
        self.loss_func = nn.CrossEntropyLoss()
        self.drop = nn.Dropout()
    
    def __dist__(self, x, y, dim):
        """
        距离计算，两种方式：点乘或者欧氏距离
        """
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x-y, 2)).sum(dim)
    
    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3) # 给S和Q增加一个维度，计算完后维度(B, total_Q, N)，表明每个查询样本和每个类别原型的距离

    def forward(self, support, query, N, K, total_Q):
        """
        N way K shot，查询样本total_Q
        """
        support_emb = self.encoder(support) # (B * N * K, D), where D is the hidden size
        query_emb = self.encoder(query) # (B * total_Q, D)
        hidden_size = support_emb.size(-1) # D
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, hidden_size) # (B, N, K, D)
        query = support.view(-1, total_Q, hidden_size) # (B, total_Q, D)
        
        support = torch.mean(support, 2) # 求每个类别的原型
        logits = self.__batch_dist__(support, query)
        minn, _ = logits.min(-1) # 计算logits在最后一个维度，也就是维度N上的最小值，其中minn维度为(B, total_Q)，表明每个查询样本与所有类别原型的最小距离
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # 扩展一个新的类别，之后logits维度为(B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N+1), 1) # 预测每个样本的类别索引
        return logits, pred

    def loss(self, logits, label):
        N = logits.size(-1)
        return self.loss_func(logits.view(-1, N), label.view(-1))
    
    def accuracy(slef, pred, label):
        return torch.mean(pred.view(-1) == label.view(-1)).type(torch.FloatTensor)
