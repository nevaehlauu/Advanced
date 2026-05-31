"""
基于梯度的NAS算法步骤3：使用网络及权重进行验证测试
"""
import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

from sklearn.preprocessing import scale

import torch.nn.functional as F

from readtxt import Well_test
from readtxt import read_txt
from readtxt import name
# from readtxt import label_num
# from readtxt import time_step

time_step = 96

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus') #数据集路径
parser.add_argument('--batch_size', type=int, default=96, help='batch size') 
parser.add_argument('--report_freq', type=float, default=50, help='report frequency') #报告频率，也就是训练过程中报告输出频率
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels') #初始通道数
parser.add_argument('--layers', type=int, default=20, help='total number of layers') #模型总层数
parser.add_argument('--model_path', type=str, default='eval-EXP-20240509-220632/train_well_1d.pt', help='path of pretrained model') #预训练模型的路径
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower') #是否在模型中使用辅助网络
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout') #在训练过程中是否使用图像裁剪。
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length') #指定图像裁剪的尺寸
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability') #用于指定路径丢弃的概率。
parser.add_argument('--seed', type=int, default=0, help='random seed') #指定随机数生成的种子
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use') #指定使用的架构类型
args = parser.parse_args()

#用于配置日记记录(logging)
log_format = '%(asctime)s %(message)s' #%(asctime)s 表示日志记录的时间，%(message)s 表示日志的消息内容。
logging.basicConfig(stream=sys.stdout, level=logging.INFO, #logging.basicConfig(...)：配置基本的日志记录设置。
    format=log_format, datefmt='%m/%d %I:%M:%S %p') #datefmt指定日志记录的时间格式。

CIFAR_CLASSES = 10

"""
TODO:
    目前问题
    RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should 
    be a MKLDNN tensor and weight is a dense tensor
"""

acc_total = 0

def main():
  global acc_total

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1) #Python 中用于退出程序的语句。它会终止当前正在执行的程序，并返回一个退出状态码，当数字为0表示正常退出，否则以异常或者错误方式退出

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True #根据当前硬件和输入数据的特性动态选择最佳的卷积算法，从而提高卷积运算的性能。
  torch.manual_seed(args.seed)
  cudnn.enabled=True #是否使用cuDNN加速，为true表示可以利用 cuDNN 提供的优化算法来加速深度神经网络的计算，从而提高性能。
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu) #使用日志记录gpu设备信息
  logging.info("args = %s", args) # 使用日志记录模块输出命令行参数信息

  genotype = eval("genotypes.%s" % args.arch) #
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  model.drop_path_prob = args.drop_path_prob

  with torch.no_grad():
    model.eval()
    model.cpu()
    for j in range(0, len(Well_test)):
      Well = read_txt(Well_test[j])
      Well_x, Well_y = [], []
      X, Y = [], []
      for line in Well:
        Well_x.append(np.array([float(x) for x in line[2:-1]]))  # x是训练数据（第3列到倒数第二列）
        Well_y.append(line[-1])  # y是标签（最后一列）

      Well_x = np.array(Well_x)
      Well_x = scale(Well_x, axis=0, with_mean=True, with_std=True, copy=True)  # 按井标准化

      for i in range(0, len(Well_x) - (time_step - 1), 32):
        #   if Well_y[i] not in name:
        #       name[Well_y[i]] = label_num
        #       label_num += 1
              # continue
          # if ori_label.count(name[Well_y[i]]) < 50000:
        X.append([x for x in Well_x[i:i + time_step]])
        labels = [name[Well_y[index]] for index in range(i, i + time_step)]
        counts = np.bincount(labels)
        Y.append(np.argmax(counts))
      X = np.swapaxes(X, 1, 2)
      X = np.array(X)
      Y = np.array(Y, dtype=np.float32)
      # X = X[:, np.newaxis, :, :]
      # X = X[:, np.newaxis, :]
      testX = torch.from_numpy(X).float()
      outputs, _ = model(testX)
      _, predicted = torch.max(outputs, 1)
      predicted = predicted.numpy()
      accuracy = (predicted == Y).mean()
      # accuracy = torch.max(model(testX), 1)[1].numpy() == Y
      print(Well_test[j].split('/')[-1] + ',' + str(accuracy * 100))
      acc_total += accuracy  
    print(acc_total / len(Well_test))


if __name__ == '__main__':
  main() 

