import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
import math, copy
from configs.config import parse_args

class Embeddings(nn.Module):
    """
    对测井数据进行编码，将每一段测井数据作为一个句子，每一个点作为一个标签
    使用transformer进行分类任务需要将标签和测井片段进行拼接（batchsize, channel, 96）--> (batchsize, channel, 97)，这里channel应该是vqgan转换为token之后的codebook维度256
    @param well_len: 测井片段长度
    @param patch_size: 测井片段划分的patch大小
    @param in_channels: 输入通道数，也即测井曲线条数
    @param d_model: hidden_size大小

    输入数据维度：batchsize, in_channel, sequence_len
    """
    def __init__(self, well_len, patch_size, in_channel, d_model, dropout=0.1):
        super(Embeddings, self).__init__()
        self.well_len = well_len
        self.patch_size = patch_size # 96 / 8 = 12，也就是一条测井片段被分为8个patch，每个patch长度为12
        self.in_channel = in_channel
        self.d_model = d_model

        # 对测井片段进行变换，获取测井块，且每一个测井块映射到d_model维度
        self.embedding = nn.Conv1d(in_channels=self.in_channel, out_channels=self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.embedding(x) # (batchsize, d_model, well_len / patch_size)
        x = x.transpose(-1, -2) #(b, patch_len, d_model)，也就是(b, 12, 512)
        return x   

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
    
def clones(module, n):
    """
    克隆N个完全相同的子模块，使用了copy.deepcopy
    : param module: 模型
    : param n: N个
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot Product Attention模块
    query: L_q * d_k, key: L_k * d_k, value: L_k * d_v
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # scores：[L_q, L_k]，这里实际应该为batchsize * sequence_length * sequence_length
    if mask is not None:
        # 将mask值为0   的位置元素替换为-1e9(负无穷大)，后面在softmax中值趋于0，可以有效屏蔽
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制
    h: 头的个数，也即将注意力机制分为多少模块
    d_model: 
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
    
    def forward(self, query, key, value, mask = None): 
        """
        X * W_q=query, X * W_k = key, X * W_v = value, 
        这里q, k, v会在后面使用时的forward函数中给出，如self.attn(y, y, y, mask)
        """
        if mask is not None:
            # 在张量dim=1的维度上加上新维度，所有h个head的mask均相同，原来[batchsize, length] --> [batchsize, 1, length]
            mask = mask.unsqueeze(1)
        n_batches = query.size(0) # L_q

        # 先使用线性变换，然后把d_model分配给h个Head，每个head为d_k=d_model/h，变换后query, key和value维度为(batchsize, length, h, d_k) --> (batchsize, h, length, d_k)，前三层FC分别对应query, key, value
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 使用注意力机制计算每一个头中的attn，x：L_q * d_v，也即L_q * d_k，多个拼接后依旧为L_q*d_model
        x, self.attn = attention(query, key, value, mask = mask, dropout=self.dropout)

        # 将所有头中得到的结果concat，并再次经过一层全连接，输出维度为d_model * d_model(512 * 512)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """
    位置编码前馈网络：构建前向传播神经网络，transformer中feed forward层：两层FC，中间加上激活函数
    位置编码前馈网络旨在对每个位置的特征进行独立的非线性变换（sequence上每个维度进行独立的非线性变换）
    """
    def __init__(self, d_model, d_ff, dropout=0.1): # d_model也即隐层，d_ff=4096
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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

class SublayerConnection(nn.Module):
    """
    LayerNorm(x + Sublayer(x))操作，其中，Sublayer(x)为多头注意力机制或者位置编码前馈网络
    实现：LayerNorm + sublayer(x) + dropout + 残差连接
    这里为了方便直接将LayerNorm放在了残差连接前，先进行归一化，在残差连接
    """
    def __init__(self, size, dropout): # size为d_model
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        """
        sublayer在后面EncoderLayer的forward中使用时给出
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """
    一层的encoder操作：多头注意力+sublayer+前馈网络+sublayer
    """
    def __init__(self, size, self_attn, feed_forward, dropout): # 注意力机制，前馈网络和sublayer都已经写过dropout层，这里不需要再写
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    """
    堆叠N层EncoderLayer层，作为完成的Encoder模块
    """
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size) # layer.size指输出张量的特征维度，EncoderLayer中有size这个属性
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # 最后在进行一个LayerNorm
        return self.norm(x)

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

class Transformer_encoder(nn.Module):
    def __init__(self, encoder, generator, src_embed, position_embed, d_model):
        super(Transformer_encoder, self).__init__()
        self.encoder = encoder
        self.generator = generator
        self.src_embed = src_embed
        self.position_embed = position_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pre_logits = nn.Identity()
    
    # def encode(self, x):
    #     return self.encoder(x)
    
    def forward(self, x):
        x = self.src_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # (b, 1, d_model)
        x = torch.cat((cls_token, x), dim=1) # (b, sequence_len / patch_size + 1, d_model)，且dim=1维度的第0个位置为标签维度
        x = self.position_embed(x)
        x = self.encoder(x)
        x = self.pre_logits(x[:, 0])
        x = self.generator(x)
        return x

def make_model(well_len, patch_size,  in_channel, tgt_vovab, cell_num, h=8, d_model=512, d_ff=2048, dropout=0.1):
    """
    构建可用于分类的transformer模型，也即只包含编码操作
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer_encoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n=cell_num),
        Generator(d_model, tgt_vovab),
        Embeddings(well_len=well_len, patch_size=patch_size, in_channel=in_channel, d_model=d_model),
        c(position),
        d_model
        )
    
    # 随机初始化参数,Xavier 初始化是一种常用的权重初始化方法，旨在使得参数在前向传播和反向传播过程中保持相对一致的方差。
    for p in model.parameters():
        if p.dim() > 1: 
            nn.init.xavier_normal(p)
    
    return model

def Net(args):
    # 参数分别是输入测井曲线数，vqvae里面codebook维度，使用的曲线，分类数，self-attention中q-k-v维度
    # 这里Embedding时需要对每条测井曲线分别预处理，获得token表示
    return make_model(well_len=args.slice_length, patch_size=args.patch_size, in_channel=args.in_channel, tgt_vovab=10, cell_num= args.cell_num) # 六条测井曲线


# net = Net()


if __name__ == "__main__":
    # pass
    args = parse_args()
    model = Net(args)
    print(model)
    well = torch.rand(1, 5, 96)
    embeding = Embeddings(well_len=96, patch_size=8, in_channel=5, d_model=512)
    out_embedding = embeding(well)
    print("-------------------", out_embedding.shape)

    # encoderLayer = EncoderLayer(512, MultiHeadedAttention(8, 512), PositionwiseFeedForward(512, 4096), 0.1)
    well_encoder = torch.rand(1, 12, 512)
    # out_encoder = encoderLayer(well_encoder)
    # print("----------------", out_encoder.shape)

    feed_forward = PositionwiseFeedForward(512, 4096)
    out_feed = feed_forward(well_encoder)
    print("----------------", out_feed.shape)
    
    summary(model, (5, 96), device="cpu")