"""
前向传播代码及验证集评估代码
"""

import torch

def common_forward(model, features, label, criterion):
    """
    前向传播复用代码
    """
    cur_device = next(model.parameters()).device
    features, label = features.to(cur_device), label.to(cur_device)
    # 对于定边130数据集和其他已经切片好的h5数据集，batch形式为batchsize * 
    output = model(features)
    loss = criterion(output, label.long())
    _, predicted = output.max(1)

    return loss, predicted, label

def evaluate_h5(model, data_loader, criterion):
    """
    验证集和测试集评估函数
    """
    model.eval()
    model.training = False
    total_loss = []
    label_nbr = 0
    eq_nbr = 0
    all_label = []
    all_predicted = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            features = batch["features"]
            label = batch["label"].long()

            loss, predicted, label = common_forward(model, features, label, criterion)
            label_nbr += len(label)
            eq_nbr += predicted.eq(label).sum().item()
            total_loss.append(loss.item())

            all_label.append(label)
            all_predicted.append(predicted)
    
    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss), all_label, all_predicted # 分别是每一轮验证的准确率、损失、标签、预测值

def evaluate_csv(model, data_loader, criterion):
    """
    验证集和测试集评估函数
    """
    model.eval()
    model.training = False
    total_loss = []
    label_nbr = 0
    eq_nbr = 0
    all_label = []
    all_predicted = []

    with torch.no_grad():
        for batch_idx, (features, label) in enumerate(data_loader):
            loss, predicted, label = common_forward(model, features, label, criterion)
            label_nbr += len(label)
            eq_nbr += predicted.eq(label).sum().item()
            total_loss.append(loss.item())

            all_label.append(label)
            all_predicted.append(predicted)
    
    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss), all_label, all_predicted # 分别是每一轮验证的准确率、损失、标签、预测值