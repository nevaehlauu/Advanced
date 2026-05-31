import os
import sys
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import logging
import time
import glob

#计算平均值
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

#求top-k精度
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    # correct_k = correct[:k].view(-1).float().sum(0)
    correct_k = correct[:k].contiguous().view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

# 数据增强：Cutout，生成一个边长为length的正方形遮掩（越过边界的话就变成矩形了）
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        # 首先生成两个随机数y和x，范围分别在[0,h]和[0,w]，也就是正方形遮掩的中心坐标
        y = np.random.randint(h)
        x = np.random.randint(w)

        # 防止越界，三个参数分别为输入数组，限定最小值，最大值
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

# 用于CIFAR的数据增强操作
def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

# 统计参数量M
def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

# 保存checkpoint，同时如果是最好模型的话也会copy一下
def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

# 保存模型
def save(model, model_path):
  torch.save(model.state_dict(), model_path)

# 下载模型 
def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

# 随机丢弃路径，来自FractalNet，至少保证有一条路径是连接输入和输出的
def drop_path(x, drop_prob):
  # if drop_prob > 0.:
  #   keep_prob = 1.-drop_prob
  #   # 从伯努利分布中抽取0或1,1的概率为keep_prob，返回的mask是个0/1的tensor，其中1的比例约为keep_prob
  #   mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob))
  #   # 没有丢弃之前的期望为E[x]，加入drop path之后的期望为p*E[x]，需要除以p保持期望不变
  #   x.div_(keep_prob)
  #   x.mul_(mask)
  return x

# 创建文件夹，copy的一些操作
def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def save_log(args):
    """
    保存一下文件信息
    """
    args.save = 'search-{}-{}-{}'.format(args.save, args.classification_name, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return args.save