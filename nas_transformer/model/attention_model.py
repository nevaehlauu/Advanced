import torch
import math
import torch.nn as nn
import copy
import torch.nn.functional as F

"""
self-attention层可搜索参数主要是num_head，也就是多头注意力机制头的个数
MultiHeadedAttention：可以改变多头注意力机制头个数的self-attention模块
MultiHeadedAttention_qkv：可以改变头个数h和q,k以及v维度的self-attention模块，其中q,k维度相等，且q, k, v维度都需要能够整除h

MLP模块，也就是位置编码前馈网络可以改变隐层通道数d_ff和输出通道数——这个模块也可以直接在模型搜索Cell的时候再写，那样看能否同时搜索激活函数

看看能否搜索激活函数（relu和leaky_relu，swish还有none）

搜索结构里面包含左右分支，每个分支需要分别搜索self-attention和MLP，一个分支的操作还可以是直接去掉整个分支，也就是“dead_branch”，或者identity，也就是相当于skip connection结构
"""
def clones(module, n):
    """
    克隆N个完全相同的子模块，使用了copy.deepcopy
    : param module: 模型
    : param n: N个
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention(query, key, value, dropout=None):
    """
    Scaled Dot Product Attention模块
    query: L_q * d_k, key: L_k * d_k, value: L_k * d_v
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # scores：[L_q, L_k]，这里实际应该为batchsize * sequence_length * sequence_length
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制
    h: 头的个数，也即将注意力机制分为多少模块
    d_model: 输入通道维度
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 确保能够完全分成这么多头
        assert d_model % h == 0
        # 将d_v设置为和d_k相等
        self.d_k = d_model // h
        self.h = h
        # 将全连接操作复制4层
        self.linears = clones(nn.Linear(d_model, d_model), 4) 
        self.attn = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): 
        """
        X * W_q=query, X * W_k = key, X * W_v = value, 
        这里q, k, v会在后面使用时的forward函数中给出，如self.attn(y, y, y, mask)
        """
        n_batches = x.size(0) # L_q

        # 先使用线性变换，然后把d_model分配给h个Head，每个head为d_k=d_model/h，变换后query, key和value维度为(batchsize, length, h, d_k) --> (batchsize, h, length, d_k)，前三层FC分别对应query, key, value
        query, key, value = [l(y).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, y in zip(self.linears, (x, x, x))]

        # 使用注意力机制计算每一个头中的attn，x：L_q * d_v，也即L_q * d_k，多个拼接后依旧为L_q*d_model
        x, self.attn = attention(query, key, value, dropout=self.dropout)

        # 将所有头中得到的结果concat，并再次经过一层全连接，输出维度为length * d_model(96 * 512)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
class MultiHeadedAttention_qkv(nn.Module):
    """
    改变qkv通道数
    h: 头的个数，也即将注意力机制分为多少模块
    d_model: 输入通道维度
    q_k_channel：query和key的输出通道数，两者应该相同
    v_channel: value输出通道数
    return: 输出维度为[batchsize, length, v_channel]
    """
    def __init__(self, h, d_model, q_k_channel, v_channal, dropout=0.1):
        super(MultiHeadedAttention_qkv, self).__init__()
        # 确保能够完全分成这么多头
        assert q_k_channel % h == 0
        assert v_channal % h == 0
        # 将d_v设置为和d_k相等
        self.d_k = int(q_k_channel // h)
        self.d_v = int(v_channal // h)
        self.h = h
        # 将全连接操作复制4层
        self.query = nn.Linear(d_model, int(q_k_channel))
        self.value = nn.Linear(d_model, int(v_channal))
        self.linear = nn.Linear(int(v_channal), d_model)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): 
        """
        X * W_q=query, X * W_k = key, X * W_v = value, 
        这里q, k, v会在后面使用时的forward函数中给出，如self.attn(y, y, y, mask)
        """
        n_batches = x.size(0) # L_q

        # 先使用线性变换，然后把d_model分配给h个Head，每个head为d_k=d_model/h，变换后query, key和value维度为(batchsize, length, h, d_k) --> (batchsize, h, length, d_k)，前三层FC分别对应query, key, value
        # q, k, v维度分别为batchsize, length, q_channel/h 和 batchsize, length, v_channel/h
        query = self.query(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.query(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.value(x).view(n_batches, -1, self.h, self.d_v).transpose(1, 2)

        # 使用注意力机制计算每一个头中的attn，x：L_q * d_v，也即L_q * d_k，多个拼接后依旧为L_q*d_model
        x, self.attn = attention(query, key, value, dropout=self.dropout)

        # 将所有头中得到的结果concat，并再次经过一层全连接，输出维度为length * v_channel
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_v)
        return self.linear(x)

class FeedForward(nn.Module):
    """
    隐层d_ff和输出通道数out_model可选的前馈网络
    位置编码前馈网络：构建前向传播神经网络，transformer中feed forward层：两层FC，中间加上激活函数
    位置编码前馈网络旨在对每个位置的特征进行独立的非线性变换（sequence上每个维度进行独立的非线性变换）
    """
    def __init__(self, d_model, mlp_ratio, dropout=0.1): # d_model也即隐层，d_ff=4096
        super(FeedForward, self).__init__()
        d_ff = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FeedForward_out(nn.Module):
    """
    隐层d_ff和输出通道数out_model可选的前馈网络
    位置编码前馈网络：构建前向传播神经网络，transformer中feed forward层：两层FC，中间加上激活函数
    位置编码前馈网络旨在对每个位置的特征进行独立的非线性变换（sequence上每个维度进行独立的非线性变换）
    """
    def __init__(self, d_model, mlp_ratio, out_channel, dropout=0.1): # d_model也即隐层，d_ff=4096
        super(FeedForward_out, self).__init__()
        d_ff = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, out_channel)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Zero(nn.Module):
    """
    直接去掉整条分支
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.mul(0.)

class LayerNorm(nn.Module):
    """
    一种代替batchNorm的归一化方法
    batchnorm是在channel维度上进行归一化，也就是会对(batchsize, sequence)的数据进行归一化，由于sequence存在sequence不一致的情况，测试时如果遇见sequence相差太多的可能预测效果不好
    LayerNorm是在batchsize维度上归一化，针对batch中单一样本的一个token计算在channel上均值和方差，无需全局均值和方差（batchnorm需要），因此更为稳定
    对于一个长度为(b, s, c)的特征，batchnorm后得到一个长度为c的一维tensor，而layernorm得到(b, s, 1)
    """
    def __init__(self, channel, eps = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(channel)) # nn.Parameter()定义可训练的参数，生成形状为feature的向量，用于缩放归一化结果(feature这里为channel), 对channel整体进行一个缩放
        self.b_2 = nn.Parameter(torch.zeros(channel)) # 用于平移归一化结果
        self.eps = eps
    
    def forward(self, x): 
        """
        计算在最后一个维度上的均值和标准差，并使用缩放因子对其进行缩放，平移因子进行平移
        (batchsize, sequence, hidden_dim)在hidden_dmi维度上操作
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean)  / (std + self.eps) + self.b_2

class Generator(nn.Module):
    """
    根据decoder的隐状态输出一个词，或者在分类任务中根据输入的sequence输出概率分布
    d_model为输出大小（对于每个patch都要有一个预测，在transformer中sequence相当于一个测井片段，里面分为多个patch,且patch=sequence_len），vocab为词典大小，分类任务中相当于类别数
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.fc = nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return self.fc(x)

class PositionalEncoding(nn.Module):
    """
    位置编码操作
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model) # 创建一个max_len * d_model大小的矩阵, 全为0
        position = torch.arange(0, max_len).unsqueeze(1) # 创建一个max_len * 1大小的矩阵，包含了0~max_len-1的连续整数
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # pos / 10000 * (2i/d_model)
        # 分别计算位置编码矩阵的奇数列和偶数列，pe[:, 0::2]表示列索引从0开始，每个两个取一个下标，相当于对偶数列进行操作
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 1 * max_len * d_model
        # 将其注册为模型的缓冲区 pe
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        # self.pe[:, : x.size(1)]指对x的sequence长度进行位置编码，position包含0~max_len-1的连续整数
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    