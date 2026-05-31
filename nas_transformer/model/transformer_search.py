"""
基于NAS搜索Transformer结构时，所搭建的基础结构
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
from model.senet import SENet18
import torch.nn.functional as F

class MixedOp(nn.Module):
  """
  混合操作模块：根据给定的权重，对输入张量应用不同的操作模块，并将它们的输出加权求和，得到最终的输出。
  这种设计可以用于搜索神经网络结构或构建具有多个不同操作的模型，以增加模型的灵活性和表达能力。
  作用：让网络架构参数的离散空间”连续“

  这里只搜索channel维度不变的self-attention结构和两层MLP中d_ff
  """
  def __init__(self, d_model, operation_map):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList() #存储混合操作的子模块
    for primitive in operation_map:
        op = operation_map[primitive](d_model)
        self._ops.append(op)

  def forward(self, x, weights):
    #zip(weights, self._ops)将权重和操作模块进行配对。op in zip()表示迭代器中的每个元素都是一个由权重值和操作模块组成的元组
    #weight即为在Network类中定义的Softmax处理之后的参数
    #将一个个节点里面几种可能性加权求和，相当与softmax方法，所有权重相加为1，并且将较好的结果权重训练的尽量大，接近于1，其他小
    return sum(w * op(x) for w, op in zip(weights, self._ops)) 

class Cell(nn.Module):
    """
    包含两个Cell，一个用于搜索self-attention结构，一个用于搜索MLP结构，并且每一个结构都需要包含左右两个分支
    self-attention结构需要搜索num_head个数
    MLP结构需要搜索MLP_radio，还有使用的激活函数

    需要计算Cell里面待确定的每个边每个操作的权重，并选取每条边最大权重对应操作，4条边（2条self-attention+2条MLP）
    MLP和self-attention搜索空间长度不同，使用长的补足短的，也就是计算权重时，对于短的用0补足
    也可以采用两个权重矩阵分别计算MLP和self-attention的权重矩阵

    一个self-attention + 一个MLP，堆叠
    """
    def __init__(self, d_model):
        super(Cell, self).__init__()
        # self-attention
        self.attention_ops = nn.ModuleList()
        attention = MixedOp(d_model, Attention_MAP)
        self.attention_ops.append(attention)
        # FeedForward
        self.mlp_ops = nn.ModuleList()
        MLP = MixedOp(d_model, FeedForward_MAP)
        self.mlp_ops.append(MLP)
        self.bn = LayerNorm(d_model)
    
    # def forward(self, x, left_attention_weight, right_attention_weight, left_mlp_weight, right_mlp_weight):
    #     """
    #     同时搜索左右分支
    #     """
    #     # left_attention_ops = sum(weights * op(x) for weights, op in zip(left_attention_weight, self.attention_ops))
    #     # right_attention_ops = sum(weights * op(x) for weights, op in zip(right_attention_weight, self.attention_ops))
    #     # left_mlp_ops = sum(weights * op(x) for weights, op in zip(left_mlp_weight, self.mlp_ops))
    #     # right_mlp_ops = sum(weights * op(x) for weights, op in zip(right_mlp_weight, self.mlp_ops))
    #     left_attention_ops = self.attention_ops[0](x, left_attention_weight)
    #     right_attention_ops = self.attention_ops[0](x, right_attention_weight)
    #     x = self.bn(left_attention_ops + right_attention_ops)
    #     left_mlp_ops = self.mlp_ops[0](x, left_mlp_weight)
    #     right_mlp_ops = self.mlp_ops[0](x, right_mlp_weight)
    #     return self.bn(left_mlp_ops + right_mlp_ops)

    def forward(self, x, left_attention_weight, left_mlp_weight):
        """
        根据Transformer结构，一个需要搜索的分支，一个残差结构
        """
        # left_attention_ops = sum(weights * op(x) for weights, op in zip(left_attention_weight, self.attention_ops))
        # right_attention_ops = sum(weights * op(x) for weights, op in zip(right_attention_weight, self.attention_ops))
        # left_mlp_ops = sum(weights * op(x) for weights, op in zip(left_mlp_weight, self.mlp_ops))
        # right_mlp_ops = sum(weights * op(x) for weights, op in zip(right_mlp_weight, self.mlp_ops))
        left_attention_ops = self.attention_ops[0](x, left_attention_weight)
        x = self.bn(left_attention_ops + x)
        left_mlp_ops = self.mlp_ops[0](x, left_mlp_weight)
        return self.bn(left_mlp_ops + x)


class Network(nn.Module):
    """
    Cell堆叠得到Transformer中Encoder
    @param: d_model: token维度
    @param: cell_num: Cell堆叠层数
    @param: num_classes: 分类数
    """
    def __init__(self, d_model, cell_num, num_classes, criterion, device):
        super(Network, self).__init__()
        self.d_model = d_model
        self._criterion = criterion # 损失函数，用于后面超参的更新
        self.device = device
        self.cell_num = cell_num
        self.num_classes = num_classes
        #self.embedding = nn.Conv1d(in_channels=5, out_channels=self.d_model, kernel_size=8, stride=8) #senet每个patch的尺寸为12，因此这里滑动时以8为单位
        self.embedding = SENet18(in_channels=5, classes=8) # inputsize是输入的曲线条数，这里outsize不起作用，SENet将曲线映射到512维度，并且映射之后每个patch的尺寸为

        self.generate = Generator(d_model, vocab=num_classes)
        self.position_embed = PositionalEncoding(d_model, dropout=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pre_logits = nn.Identity()
        self.cells = nn.ModuleList()
        
        for i in range(cell_num):
           cell = Cell(d_model)
           self.cells += [cell]
        self._initialize_alphas() # 初始化参数
    
    def new(self):
        """
        建立一个新的Network，并将当前对象的架构参数复制到新建的对象
        """
        model_new = Network(self.d_model, self.cell_num, self.num_classes, self._criterion, self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        x = self.embedding(x) # 这里x维度为batchsize, 64(一个切片8个patch，8*8), 256(token维度)
        x = x.transpose(-1, -2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # (b, 1, d_model)
        x = torch.cat((cls_token, x), dim=1) # (b, sequence_len / patch_size + 1, d_model)，且dim=1维度的第0个位置为标签维度
        x = self.position_embed(x)

        ####################### encoder部分，也就是Cell堆叠过程    
       
        ################ 左右分支堆叠情况
        # for i, cell in enumerate(self.cells):
        #     left_attention_weights = F.softmax(self.left_attention_alphas, dim=-1)
        #     right_attention_weights = F.softmax(self.right_attention_alphas, dim=-1)
        #     left_mlp_weights = F.softmax(self.left_mlp_alphas, dim=-1)
        #     right_mlp_weights = F.softmax(self.right_mlp_alphas, dim=-1)
        #     x = cell(x, left_attention_weights, right_attention_weights, left_mlp_weights, right_mlp_weights)

        ################# 一个分支搜索，一个分支为残差连接情况
        for i, cell in enumerate(self.cells):
            left_attention_weights = F.softmax(self.left_attention_alphas, dim=-1)
            left_mlp_weights = F.softmax(self.left_mlp_alphas, dim=-1)
            x = cell(x, left_attention_weights, left_mlp_weights)

        
        x = self.pre_logits(x[:, 0])
        x = self.generate(x)
        return x
    
    def _loss(self, input, target):
        logits = self(input) #调用父类__call__,调用forward
        return self._criterion(logits, target) #返回交叉熵损失

    def _initialize_alphas(self): #Cell参数初始化
       # Cell有4条边待选择，两个self-attention的边，两个MLP的边，且两个矩阵长度不同
       attention_num_ops = len(Attention_MAP)
       mlp_num_ops = len(FeedForward_MAP)
    #    self.left_attention_alphas = Variable(1e-3*torch.randn(1, attention_num_ops).cuda(), requires_grad=True)
    #    self.right_attention_alphas = Variable(1e-3*torch.randn(1, attention_num_ops).cuda(), requires_grad=True)
    #    self.left_mlp_alphas = Variable(1e-3*torch.randn(1, mlp_num_ops).cuda(), requires_grad=True)
    #    self.right_mlp_alphas = Variable(1e-3*torch.randn(1, mlp_num_ops).cuda(), requires_grad=True)
       self.left_attention_alphas = 1e-3*torch.randn(attention_num_ops).to(self.device)
    #    self.right_attention_alphas = 1e-3*torch.randn(attention_num_ops).to(self.device)
       self.left_mlp_alphas = 1e-3*torch.randn(mlp_num_ops).to(self.device)
    #    self.right_mlp_alphas = 1e-3*torch.randn(mlp_num_ops).to(self.device)
       self.left_attention_alphas.requires_grad = True
    #    self.right_attention_alphas.requires_grad = True
       self.left_mlp_alphas.requires_grad = True
    #    self.right_mlp_alphas.requires_grad = True
    #    self._arch_parameters = [self.left_attention_alphas, self.right_attention_alphas, self.left_mlp_alphas, self.right_mlp_alphas]
       self._arch_parameters = [self.left_attention_alphas, self.left_mlp_alphas]

    
    def arch_parameters(self):
       return self._arch_parameters

    def genotype(self):
        """
        根据训练结果获取训练后的Cell
        """
        def _parse(weights, operation):
            """
            根据权重选择最优的边
            """
            gene = []
            W = weights.clone()
            max_value, max_index = torch.max(W, dim=0) #选择最大权重对应的操作索引
            gene.append((operation[max_index.item()], max_index.item()))
            return gene
            
        # gene_left_attention = _parse(F.softmax(self.left_attention_alphas, dim=-1).data.cpu().numpy())
        # attention_gene = [_parse(F.softmax(self.left_attention_alphas, dim=-1), Attention_OPERATION), _parse(F.softmax(self.left_attention_alphas, dim=-1), Attention_OPERATION)]
        # mlp_gene = [_parse(F.softmax(self.left_mlp_alphas, dim=-1), FeedForward_OPERATION), _parse(F.softmax(self.right_mlp_alphas, dim=-1), FeedForward_OPERATION)]

        attention_gene = [_parse(F.softmax(self.left_attention_alphas, dim=-1), Attention_OPERATION)]
        mlp_gene = [_parse(F.softmax(self.left_mlp_alphas, dim=-1), FeedForward_OPERATION)]

        genotype = Genotype(
            self_attention = attention_gene,
            mlp = mlp_gene 
        )
        return genotype