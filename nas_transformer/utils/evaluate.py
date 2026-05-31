"""
评估代码
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import sample_to_device

def common_forward(model, features, label, criterion):
    """
    前向传播复用代码
    : param model: 网络模型
    : param batch: 批尺寸大小
    : param criterion: 损失函数
    : features_name: 使用的测井曲线列表
    : return: 损失函数以及预测值
    """
    # 获取net模型所在的设备，获取后一般将数据传给当前设备，也就是sample_to_device(batch, cur_device)
    cur_device = next(model.parameters()).device
    features, label = features.to(cur_device), label.to(cur_device)

    output = model(features)
    loss = criterion(output, label.long())
    _, predicted = output.max(1)

    return loss, predicted, label

def evaluate(model, data_loader, criterion):
    """
    评估模型
    :param model: 网络模型
    :param data_loader: dataloader对象
    :param criterion: 损失函数
    :features_name: 使用的测井曲线列表
    :return: 每一轮的准确率，损失，真实标签，预测标签
    """
    model.eval()  # 很重要
    model.training = False
    total_loss = []
    label_nbr = 0
    eq_nbr = 0
    all_label = []
    all_predicted = []

    with torch.no_grad():
        n_batches = len(data_loader)
        pbar = tqdm(enumerate(data_loader, 0),
                    unit=' images', # 进度条中显示单位为“images”
                    unit_scale=data_loader.batch_size, #进度条每移动一个刻度表示处理了data_loader.batch_size个样本
                    total=n_batches,
                    smoothing=0,
                    disable=False,
                    ncols=135)  # 调整长度

        for batch_idx, (feature, label) in pbar:
            
            loss, predicted, label = common_forward(model, feature, label, criterion)
            label_nbr += len(label) # 考虑到整体数量不能被batch_size整除
            eq_nbr += predicted.eq(label).sum().item() # 记录标签准确的数量
            total_loss.append(loss.item()) # 记录损失函数

            all_label.append(label)
            all_predicted.append(predicted)

            pbar.set_description('{:<8d}'.format(batch_idx))

    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss), all_label, all_predicted