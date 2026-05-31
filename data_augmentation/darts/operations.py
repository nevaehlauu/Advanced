import torch
import torch.nn as nn

#OPS为字典，键是操作名称，值是一个 lambda 函数，该函数接受输入参数 C（通道数）、stride（步长）和 affine（是否进行仿射变换），并返回对应的操作模块。
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'conv_1x1' : lambda C, stride, affine: ReLUConvBN(C, C, 1, stride, 'same', affine=affine),
  'conv_3x1' : lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 'same', affine=affine),
  'conv_5x1' : lambda C, stride, affine: ReLUConvBN(C, C, 5, stride, 'same', affine=affine),
  'conv_7x1' : lambda C, stride, affine: ReLUConvBN(C, C, 7, stride, 'same', affine=affine),
  'conv_11x1' : lambda C, stride, affine: ReLUConvBN(C, C, 11, stride, 'same', affine=affine),
  'res_block_3' : lambda C, stride, affine: ResidualBlock(C, C, 3, stride, 'same'),
  # 'res_block_5' : lambda C, stride, affine: ResidualBlock(C, C, 5, stride, 'same'),
  # 'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine), #跳跃连接
  # 'dil_conv_3x1': lambda C, stride, affine: DilConv(C, C, 3, stride, padding='same', dilation=2, affine=affine),
  'dil_conv_5x1': lambda C, stride, affine: DilConv(C, C, 3, stride, padding='same', dilation=2, affine=affine),
  'dil_conv_7x1': lambda C, stride, affine: DilConv(C, C, 3, stride, padding='same', dilation=3, affine=affine),
  # 能够将连接变成一个点，以及如果继续连接两个点，是否需要加上一个切断连接的操作
}

# 'conv_1x1' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (1, 1), stride=(stride, 1), padding='same', bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#   ),
#   'conv_3x1' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (3, 1), stride=(stride, 1), padding='same', bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#   ),
#   'conv_5x1' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (5, 1), stride=(stride, 1), padding='same', bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#   ),
#   'conv_7x1' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding='same', bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#   ),
#   'conv_11x1' : lambda C, stride, affine: nn.Sequential(
#     nn.ReLU(inplace=False),
#     nn.Conv2d(C, C, (11, 1), stride=(stride, 1), padding='same', bias=False),
#     nn.BatchNorm2d(C, affine=affine)
#   ),

class ReLUConvBN(nn.Module):
  """
  一个激活+卷积+BN操作
  """
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm1d(C_out, affine=affine) #表示是否应用仿射变化
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    """
    空洞卷积
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
      super(DilConv, self).__init__()
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
        nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
      return self.op(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.res_conv = nn.Sequential(
          nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
          nn.BatchNorm1d(out_channel),
          nn.ReLU(),

          nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
          nn.BatchNorm1d(out_channel),
          nn.ReLU(),
        )
        if in_channel != out_channel:
          self.res_layer = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channel)
          )
        else:
          self.res_layer = None
    def forward(self, x):
        if self.res_layer is not None:
          return self.res_conv(x) + self.res_layer(x)
        else:
          return self.res_conv(x)

class SepConv(nn.Module):
    """
    深度可分离卷积
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
      super(SepConv, self).__init__()
      self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
        nn.Conv1d(C_in, C_in, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm1d(C_in, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
        nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm1d(C_out, affine=affine),
        )

    def forward(self, x):
      return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
  """
  实现因式化降采样层：将输入特征图通道数减半，并进行降采样操作
  """
  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0 #保证c_out//2可以除尽
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) #1×1卷积，将输入通道数减半，并进行降采样
    self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm1d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1) #conv_2(x[:,:,1:,1:])去除左上角一个元素，dim=1在通道维度拼接，输出张量通道数c_out
    out = self.bn(out)
    return out

