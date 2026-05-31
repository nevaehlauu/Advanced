import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.operations import *
from torch.autograd import Variable
from darts.genotypes import PRIMITIVES
from darts.genotypes import Genotype


class MixedOp(nn.Module):
  """
  混合操作模块：根据给定的权重，对输入张量应用不同的操作模块，并将它们的输出加权求和，得到最终的输出。
  这种设计可以用于搜索神经网络结构或构建具有多个不同操作的模型，以增加模型的灵活性和表达能力。
  作用：让网络架构参数的离散空间”连续“
  """
  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList() #存储混合操作的子模块
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      # if 'pool' in primitive:
      #   op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False)) #先池化，在批归一化
      self._ops.append(op)

  def forward(self, x, weights):
    #zip(weights, self._ops)将权重和操作模块进行配对。op in zip()表示迭代器中的每个元素都是一个由权重值和操作模块组成的元组
    #weight即为在Network类中定义的Softmax处理之后的参数
    #将一个个节点里面几种可能性加权求和，相当与softmax方法，所有权重相加为1，并且将较好的结果权重训练的尽量大，接近于1，其他小
    return sum(w * op(x) for w, op in zip(weights, self._ops)) #输出值乘上权重相加，权重即α


class Cell(nn.Module):
  """
  定义一个细胞模块，用于构建神经网络的基本单元，相当于ResNet中两个残差块构建过程
  构建一个有多个细胞模块组成的神经网络，每个细胞模块根据给定的输入节点和权重，生成新的输出节点。
  这种设计可以用于搜索神经网络结构或构建具有多个不同操作的模型，以增加模型的灵活性和表达能力。
  """

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev): 
    #multiplier乘法器，C_prev_prev, C_prev前两个节点通道数，reduction_prev前一个细胞是否为降采样模块
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)
    self.reduction = reduction

    # 对输入节点定义
    if reduction_prev: #前一个节点为降采样模块
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False) #进行因式化降采样
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False) #完成一个1*1的激活+卷积+BN操作
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps #self._steps=4,每个cell中有4个节点的连接状态待确定
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()

    #对于中间节点定义
    for i in range(self._steps):
      for j in range(2+i): #对于每一个节点，它有2+i个前驱节点
        # stride = 2 if reduction and j < 2 else 1 #reduction表示是否为Reduction cell（网络1/3和2/3处）
        stride = 1
        op = MixedOp(C, stride) #构建两个节点之间的混合操作
        self._ops.append(op) #所有操作添加到_ops,len(_ops)=14~2+3+4+5

  def forward(self, s0, s1, weights):
    #输入节点预处理
    s0 = self.preprocess0(s0) #第一个输入
    s1 = self.preprocess1(s1) #第二个输入

    #循环应用混合操作模块，通过连接和加权求和的方式生成新的状态。
    states = [s0, s1] #当前节点的前驱节点
    offset = 0
    # 遍历每个intermediate nodes，得到每个节点的output
    for i in range(self._steps):
      # s为当前节点i的output，在ops找到i对应的操作，然后对i的所有前驱节点做相应的操作（调用了MixedOp的forward），然后把结果相加
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states)) #_ops中存放的是MixedOP类的对象，相当于调用MxedOP的__call__->forward
      offset += len(states) #下一个节点的起始行数
      states.append(s) #把当前节点i的output作为下一个节点的输入[s0,s1,b1,b2,b3,b4]
    #将多个状态拼接作为最后输出，一个输出节点
    return torch.cat(states[-self._multiplier:], dim=1) #对节点的output进行concat作为当前cell的输出


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, device, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion #损失函数
    self.device = device
    self._steps = steps #一个Cell内有4个node需要进行操作
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      # nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.Conv1d(C, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm1d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      # if i in [layers//3, 2*layers//3]: #redection cell
      #   C_curr *= 2
      #   reduction = True
      # else:
      #   reduction = False
      # 通道尺寸变化
      C_curr *= 2
      reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr # #四个node采用concat方式连接，所以C需要承4

    self.global_pooling = nn.AdaptiveAvgPool1d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas() # 初始化参数

  def new(self): #建立一个新的Network，并将当前对象的架构参数复制到新建的对象
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.device)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells): #遍历8层的Cell
      if cell.reduction: # 为每一个Cell赋权重
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights) #第k个Cell有两个输入，分别是第k-1，k-2个Cell
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input) #调用父类__call__,调用forward
    return self._criterion(logits, target) #返回交叉熵损失

  def _initialize_alphas(self): #Cell参数初始化
    k = sum(1 for i in range(self._steps) for n in range(2+i)) #参数一共有14行，即cell有14条边待选择
    num_ops = len(PRIMITIVES) #每行（条边）有8种选择

    self.alphas_normal = 1e-3*torch.randn(k, num_ops).to(self.device)
    self.alphas_reduce = 1e-3*torch.randn(k, num_ops).to(self.device)
    self.alphas_normal.requires_grad = True
    self.alphas_reduce.requires_grad = True
    # self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) #初始化normal cell的alphas
    # self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True) #初始化reduction cell的alphas
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
    """
    根据训练结果获得训练后的Cell
    """
    def _parse(weights): #经过softmax后的weight weight=[14*8]
      gene = []
      n = 2
      start = 0 #确定节点的前置边开始的那条边
      for i in range(self._steps):
        end = start + n #确定节点的前置边结束的那条边{[0,2],[2,5],[5,9],[9,14]}
        W = weights[start:end].copy() #复制第i节点所有入度边到W。W长度分别为2，3，4，5
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        #选出包含最大权重的两条边（每条边上有8个操作）||range(i+2)该点入度的边数,即x的取值范围||sort排序是从小到大，所以取负数
        for j in edges: #遍历需要选取操作的边
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k #遍历每条边上8个操作，选出权重最大的操作
          gene.append((PRIMITIVES[k_best], j)) #将该操作加入gene,gene[14*2],第一位是操作，第二维是该操作对应的前直接点的序号
        start = end
        n += 1 #后一个node的入度边数比前一个节点多1
      return gene

    #_parse函数饭后权重采样之后的权重信息
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()) #对normal cell的参数做softmax，将数据取出放在CPU上并转换为numpy格式
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2) #【2，3，4，5】
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

