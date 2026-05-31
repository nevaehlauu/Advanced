import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    # optimizer = torch.optim.Adam(model.module.arch_parameters(), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    # 更新神经网络架构参数，其中betas=(0.9, 0.999): Adam 算法中的两个 beta 参数,用于计算梯度的一阶和二阶矩。
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target) #对model进行一次训练，获取交叉熵损失,获得的的是Ltraib(w,α）
    theta = _concat(self.model.parameters()).data #把参数整理成一行代表一个参数的形式,得到我们要更新的参数theta
    try: ###此处需要先学习带有动量的梯度下降法###
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum) #network_momentum=0.9,momentum*v,用的就是Network进行w更新的momentum
    except:
      moment = torch.zeros_like(theta) #不加momentum
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta #前面的是loss对参数theta求梯度，后面是正则项，即  dwLtrain(w,α)+weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta)) #w'=w − ξ*dwLtrain(w, α)
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad() #清除上一步残留的参数值
    if unrolled: #如unrolled==True,则使用论文中提出的方法
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    # loss = self.model.module._loss(input_valid, target_valid)
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    #w' = w − ξ*dwLtrain(w, α)
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    # Lval(w',α)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    # dαLval(w',α)
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    #dwLval(w',α)
    vector = [v.grad.data for v in unrolled_model.parameters()]
    # (dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)
    # 公式六减公式八 dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta): #theta=w'=w − ξ*dwLtrain(w, α)
    model_new = self.model.new() #model_new有self有共同的架构参数
    model_dict = self.model.state_dict() # Returns a dictionary containing a whole state of the module.

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size()) #获取参数量
      params[k] = theta[offset: offset+v_length].view(v.size()) #将named_parameters中参数复制到params
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params) #更新参数地点
    model_new.load_state_dict(model_dict) #model_new的参数等于更新后的参数
    return model_new.cuda()

  # 计算(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)     其中w+=w + dw'Lval(w',α)*epsilon      w- =w - dw'Lval(w',α)*epsilon
  def _hessian_vector_product(self, vector, input, target, r=1e-2): #vector就是dw'Lval(w',α)
    R = r / _concat(vector).norm() # epsilon
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v) # 将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
    # dαLtrain(w-,α)
    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v) # 将模型中所有的w'更新成w- = w+ - (w-)*2*epsilon = w+dw'Lval(w',α)*epsilon - 2*epsilon*dw'Lval(w',α)=w-dw'Lval(w',α)*epsilon
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())
    # 将模型的参数从w-恢复成w
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    # w=(w-) +dw'Lval(w',α)*epsilon = w-dw'Lval(w',α)*epsilon + dw'Lval(w',α)*epsilon = w
    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

