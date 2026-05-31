"""
原型网络训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils.utils import set_seeds, save_train_val_fig
from data.dataloader import get_dataloader
from model.senet import SENet18 as Net

def parse_args():
    parser = argparse.ArgumentParser("few_shot_learing")
    parser.add_argument("--epochs", default=200, help="训练轮次")
    parser.add_argument("--classes", default=10, help="分类数")
    parser.add_argument("--learning_rate", default=0.0003, help="学习率")
    parser.add_argument("--decay", default=0.98, help="学习率衰减")
    parser.add_argument("--gpu_id", default="0",  help="gpu的id")
    parser.add_argument("--log_dir_path", default="log/protonet/", help="日志文件存储位置")
    parser.add_argument("--print_period", default=10, help="打印间隔")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=64, help="滑动步长")

    # 下面是可能需要改动的配置
    parser.add_argument("--pretrained", default=True, help="是否加载预训练模型")
    parser.add_argument("--pretrained_filepath", default="pretrain_model/senet_6/best_epoch_model_6.pth", help="预训练模型位置")
    parser.add_argument("--seed", default=42, help="随机数种子")
    parser.add_argument("--in_channel", default=6, help="测井曲线条数")
    parser.add_argument("--train_well_num", default=4, help="进行训练的井数")
    parser.add_argument("--val_well_num", default=13, help="进行训练的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--train_categorize_id", default=1, help="训练集使用的区块")
    parser.add_argument("--val_categorize_id", default=1, help="测试集区块名")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")
    parser.add_argument("--n_batch", default=200, help="每个epoch中生成的批次数量，有点像N way K shot中的episode概念，也就是每轮训练多少个批次")
    parser.add_argument("--n_cls", default=10, help="每个批次中选择的类别数量")
    parser.add_argument("--support", default=10, help="每个类别中选择的support样本数量")
    parser.add_argument("--query", default=2, help="每个类别中选择的query样本数量")

    args = parser.parse_args()
    return args

def euclidean_metric(a, b):
    """
    欧式距离计算公式
    """
    a = a.unsqueeze(1)
    b = b.unsqueeze(0)
    logits = -((a - b)**2).sum(dim=2) 
    return logits 

def cosine_metric(a, b):
    """
    余弦距离计算公式
    """
    # 点积
    a = a.unsqueeze(1)
    b = b.unsqueeze(0)
    dot_product = (a * b).sum(dim=2)

    # 计算模长
    norm_a = a.norm(dim=2)
    norm_b = b.norm(dim=2)

    # 余弦相似度
    cosine_similarity = dot_product / (norm_a * norm_b + 1e-8) # 加上小值防止除0

    return cosine_similarity

def train(model, train_loader, optimizer, criterion, print_period, args):
    model.train()
    total_loss = []
    label_nbr = 0 # 标签总数
    eq_nbr = 0 # 预测正确的标签数
    cur_device = next(model.parameters()).device
    print_period = 10 
    start = time.time()

    for batch_idx, (feature, label) in enumerate(train_loader):
        feature, label = feature.to(cur_device), label.to(cur_device)
        p = args.support * args.n_cls
        data_support, data_query = feature[:p], feature[p:]
        # query_label = label[p:]

        query_label = torch.arange(args.n_cls).repeat(args.query)
        query_label = query_label.to(cur_device)
        # data_support = []
        # data_query = []
        # query_label = []
        # for i in range(args.n_cls):
        #     # data_support.append(feature[i * step: args.support + i * step])
        #     # data_query.append(feature[args.support + i * step : (i+1) * step])
        #     # query_label.append(label[args.support + i * step : (i+1) * step])
        #     cur_nbr = 0
        #     cur_idx = i
        #     while cur_idx < len(feature):
        #         if cur_nbr < args.support:
        #             data_support.append(feature[cur_idx])
        #         else:
        #             data_query.append(feature[cur_idx])
        #             query_label.append(label[cur_idx])
        #         cur_idx += args.n_cls
        #         cur_nbr += 1

        # data_support = torch.stack(data_support)
        # data_query = torch.stack(data_query)
        # query_label = torch.stack(query_label)

        model.zero_grad()
        optimizer.zero_grad()
        proto = model(data_support)
        proto = proto.reshape(args.support, args.n_cls, -1).mean(dim=0) # 求每个类别的样本原型，这里维度为(n_cls, hidden_size)
        
        logits = euclidean_metric(model(data_query), proto) # (query_shot, n_cls)，查询集样本和每个类别原型的距离，距离越小越好，所以logits为负数
        loss = criterion(logits, query_label.long())
        _, pred = torch.max(logits, dim = 1) # 获取每一行最大值和其对应的索引
        acc = (pred == query_label).type(torch.float).mean().item() 
        
        loss.backward()
        optimizer.step()
        label_nbr += len(query_label)
        eq_nbr += pred.eq(query_label).sum().item()
        total_loss.append(loss.item())

        if batch_idx % print_period == 0:
            print('train {}/{}, loss={:.4f} acc={:.4f}%'.format(
                batch_idx, len(train_loader), loss.item(), 100 * acc)) 
        
    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss)
        
def valid(model, valid_loader, criterion, args):
    model.eval()
    model.training = False
    cur_device = next(model.parameters()).device
    total_loss = []
    all_label = []
    all_predict = []
    label_nbr = 0
    eq_nbr = 0

    with torch.no_grad():
        for batch_idx, (feature, label) in enumerate(valid_loader):
            feature, label = feature.to(cur_device), label.to(cur_device)
            p = args.support * args.n_cls
            data_support, data_query = feature[:p], feature[p:]
            query_label = torch.arange(args.n_cls).repeat(args.query)
            query_label = query_label.to(cur_device)

            proto = model(data_support)
            proto = proto.reshape(args.support, args.n_cls, -1).mean(dim = 0)
            logits = euclidean_metric(model(data_query), proto)
            loss = criterion(logits, query_label.long())
            _, pred = torch.max(logits, dim = 1) 
            label_nbr += len(query_label)
            eq_nbr += pred.eq(query_label).sum().item()
            total_loss.append(loss.item())
            all_label.append(query_label)
            all_predict.append(pred)
    
    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss), all_label, all_predict

def main(args):

    set_seeds(args.seed)

    # 数据集
    trainLoader, testLoader = get_dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.train_well_num, args.val_well_num, args.frequency_aug, args.train_categorize_id, args.val_categorize_id, args.noise_ration, args.n_batch, args.n_cls, args.support + args.query)
    device = torch.device("cuda:" + args.gpu_id)
    model = Net(in_channels=args.in_channel, classes=args.classes)
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.decay)

    # 加载预训练参数
    if args.pretrained == True and args.pretrained_filepath is not None and Path(args.pretrained_filepath).exists():
        loaded_model = torch.load(args.pretrained_filepath, map_location=torch.device("cpu"))
        net_dict = model.embedding.state_dict()
        # 判断model尺寸是否相同，仅加载相同的model
        pretrained_dict = {k : v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}
        net_dict.update(pretrained_dict)
        model.embedding.load_state_dict(net_dict, strict=False)

    # 模型日志存储
    date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  # 加上时间
    args.log_dir_path = str((Path(args.log_dir_path) / date_time / str(args.train_well_num)).resolve())
    Path(args.log_dir_path).mkdir(parents=True, exist_ok=True) # 确保父目录存在

    print("----------------训练开始----------------------")

    # 在日志中记录准确率等值
    result_dict = {"trainloss": [], "valloss": [], "trainacc": [], "valacc": []}
    best_acc = -0x3f3f3f3f
    best_epoch = 0

    for epoch in range(args.epochs):
        
        print("---------第{}轮训练开始-------------".format(epoch))

        train_acc, train_loss = train(model, trainLoader, optimizer, loss_func, args.print_period, args)
        val_acc, val_loss, val_label, val_predict = valid(model, testLoader, loss_func, args)
        scheduler.step()

        result_dict['trainloss'].append(train_loss)
        result_dict['valloss'].append(val_loss)
        result_dict['trainacc'].append(train_acc)
        result_dict['valacc'].append(val_acc)

        print("epoch: {}, train_acc: {:.4f}%, val_acc: {:.4f}%, train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, 100 * train_acc, 100 * val_acc, train_loss, val_loss
        ))

        save_train_val_fig(result_dict['trainacc'], result_dict['valacc'], 'train_acc', 'val_acc', "train and val accuracy", "epochs", "accuracy", str(Path(args.log_dir_path) / "train_val_acc.png"))
        save_train_val_fig(result_dict['trainloss'], result_dict['valloss'], 'train_loss', 'val_loss', "train and val loss", "epochs", "loss", str(Path(args.log_dir_path) / "train_val_loss.png"))

        # 保存结果最好的模型
        if best_acc < val_acc:
            best_acc = val_acc
            best_epoch = epoch
            print("best acc: {:4f}%, best epoch: {}".format(100 * best_acc, epoch))
            print("save model")
            savepath = str(Path(args.log_dir_path) / "best_epoch_model.pth")
            torch.save(model.state_dict(), savepath)

        with open(Path(args.log_dir_path) / 'logging.json', "w") as f:
            json.dump(result_dict, f, indent=2)
    
    print("Training best val acc: " + str(best_acc) + ", best epoch: " + str(best_epoch))
    print('Training complete, models saved in {}'.format(args.log_dir_path))

if __name__ == '__main__':

    my_args = parse_args()
    main(my_args)