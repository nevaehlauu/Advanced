"""
采用对抗学习的DANN模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    """
    梯度反转层
    """
    @staticmethod
    def forward(ctx, x, constant):
        # ctx: 上下文对象，用于在前向传播和反向传播之间存储信息，x: 输入张量，constant：一个常数，用于控制梯度反转的程度
        ctx.constant = constant
        return x.view_as(x) # view_as保持形状不变
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: 从后续层传递过来的梯度
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None # 返回反转之后的梯度
    
    def grad_reverse(x, constant):
        # 用于调用 GradReverse 的 apply 方法。它接受输入张量 x 和反转常数 constant，并返回经过梯度反转处理的结果。
        return GradReverse.apply(x, constant)


class Classifier(nn.Module):
    """
    DANN的分类器
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(F.dropout(x))
        return F.log_softmax(x, 1)  # 将输出转化为每个类的对数概率


class Domain_classifier(nn.Module):
    """
    域判别器
    """
    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)
    
    def forward(self, x, constant):
        # 梯度反转层
        x = GradReverse.grad_reverse(x, constant)
        x = F.relu(self.fc1(x))
        x = self.fc2(F.dropout(x))
        return F.log_softmax(x, 1)