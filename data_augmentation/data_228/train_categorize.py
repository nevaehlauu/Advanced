import torch
import sys
from pathlib import Path
import os
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.utils import set_seeds, save_train_val_fig
import time
from readtxt_categorize import dataloader, dataloader_split
from network import Net_1d as Net
# from SENet import Net as Net
from datetime import datetime
import json
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser("few_shot_learing")
    parser.add_argument("--epochs", default=100, help="训练轮次")
    parser.add_argument("--classes", default=10, help="分类数")
    parser.add_argument("--learning_rate", default=0.0005, help="学习率")
    parser.add_argument("--decay", default=0.98, help="学习率衰减")
    parser.add_argument("--gpu_id", default="3",  help="gpu的id")
    parser.add_argument("--seed", default=10, help="随机数种子")
    parser.add_argument("--log_dir_path", default="few_shot_228/no_aug/", help="日志文件存储位置")
    parser.add_argument("--print_period", default=10, help="打印间隔")
    parser.add_argument("--pretrained", default=False, help="是否加载预训练模型")
    parser.add_argument("--pretrained_filepath", default="", help="预训练模型位置")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=96, help="滑动步长")
    parser.add_argument("--batchsize", default=1024)

    # 下面是可能需要改动的配置
    parser.add_argument("--in_channel", default=6, help="测井曲线条数")
    parser.add_argument("--train_well_num", default=50, help="进行训练的井数")
    parser.add_argument("--val_well_num", default=13, help="进行训练的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--train_categorize_id", default=1, help="训练集使用的区块")
    parser.add_argument("--val_categorize_id", default=1, help="测试集区块名")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")

    args = parser.parse_args()
    return args

def train(model, train_loader, optimizer, criterion, print_period):
    model.train()
    total_loss = []
    label_nbr = 0 # 标签总数
    eq_nbr = 0 # 预测正确的标签数
    cur_device = next(model.parameters()).device
    print_period = 10 
    start = time.time()

    for batch_idx, (feature, label) in enumerate(train_loader):
        feature, label = feature.to(cur_device), label.to(cur_device)
        model.zero_grad()
        optimizer.zero_grad()
        y0, output = model(feature)
        loss = criterion(output, label.long())
        _, predict = output.max(1)
        loss.backward()
        optimizer.step()
        label_nbr += len(label)
        eq_nbr += predict.eq(label).sum().item()
        total_loss.append(loss.item())

        if batch_idx % print_period == 0:
            use_time = time.time() - start
            start = time.time()
            print("Batch: {} / {}, loss: {:.4f}, accuracy: {:.4f}%, speed: {:.4f} sliced/s, use time: {:.4f} s".format(
                batch_idx, len(train_loader),
                sum(total_loss) / len(total_loss),
                100 * eq_nbr / label_nbr,
                label_nbr / use_time,
                use_time
            ))
    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss)
        
def valid(model, valid_loader, criterion):
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
            y0, output = model(feature)
            loss = criterion(output, label.long())
            _, predict = output.max(1)
            label_nbr += len(label)
            eq_nbr += predict.eq(label).sum().item()
            total_loss.append(loss.item())
            all_label.append(label)
            all_predict.append(predict)
    
    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss), all_label, all_predict

def main(args):

    set_seeds(args.seed)

    # 数据集
    trainLoader, validLoader, label_name = dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.train_well_num, args.val_well_num, args.frequency_aug, args.batchsize, args.train_categorize_id, args.val_categorize_id, args.noise_ration)
    device = torch.device("cuda:" + args.gpu_id)
    model = Net(in_channels=args.in_channel, classes=len(label_name))
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.decay)

    # 加载预训练参数
    if args.pretrained == True and args.pretrained_filepath is not None and Path(args.pretrained_filepath).exists():
        loaded_model = torch.load(args.pretrained_filepath, map_location=torch.device("cpu"))
        net_dict = model.state_dict()
        # 判断model尺寸是否相同，仅加载相同的model
        pretrained_dict = {k : v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}
        net_dict.update(pretrained_dict)
        model.load_state_dict(net_dict, strict=False)

    # 模型日志存储
    date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  # 加上时间
    args.log_dir_path = str((Path(args.log_dir_path) / date_time).resolve())
    Path(args.log_dir_path).mkdir(parents=True, exist_ok=True) # 确保父目录存在

    print("----------------训练开始----------------------")

    # 在日志中记录准确率等值
    result_dict = {"trainloss": [], "valloss": [], "trainacc": [], "valacc": []}
    best_acc = -0x3f3f3f3f
    best_epoch = 0

    for epoch in range(args.epochs):
        
        print("---------第{}轮训练开始-------------".format(epoch))

        train_acc,  train_loss = train(model, trainLoader, optimizer, loss_func, args.print_period)
        val_acc, val_loss, val_label, val_predict = valid(model, validLoader, loss_func)
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
