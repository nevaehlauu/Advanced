"""
对搜索得到的Transformer进行训练，测试其性能
"""
import os
import sys
import time

curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from utils.evaluate import common_forward, evaluate
from utils.utils import set_seeds
from dataloader.dataloader import dataloader

"""
训练代码
在根目录下运行该脚本
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path("./").resolve()))

from configs.get_config import get_train_cfg, cfg2dict
from utils.utils import print_block, set_seeds, save_train_val_fig, printcolor, count_parameters_in_MB
from model.transformer_searched import Network
from model.genotypes import Transformer_Encoder as genotype

def parse_args():
    """
    获取参数
    :return: args
    """
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument("--epochs", default=100, help="训练轮次")
    parser.add_argument("--learning_rate", default=0.025, help="学习率")
    parser.add_argument("--decay", default=0.98, help="学习率衰减")
    
    parser.add_argument("--seed", default=42, help="随机数种子")
    parser.add_argument("--log_dir_path", default="log/transformer_search/", help="日志文件存储位置")
    parser.add_argument("--print_period", default=10, help="打印间隔")

    parser.add_argument("--pretrained_filepath", default="log/transformer_pth/best_epoch_model.pth", help="预训练模型位置")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=64, help="滑动步长")
    parser.add_argument("--d_model", default=512, help="嵌入层维度")
    parser.add_argument("--batchsize", default=1024)
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping') #梯度裁剪阈值，用于控制梯度的最大范围，以避免梯度爆炸的问题。


    # 下面是可能需要改动的配置
    parser.add_argument("--num_classes", default=10, help="分类数量")
    parser.add_argument('--cell_num', default=4, help='Cell堆叠层数')
    parser.add_argument("--train_well_num", default=92, help="进行训练的井数")
    parser.add_argument("--val_well_num", default=13, help="进行测试的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--train_categorize_id", default=1, help="训练集使用的区块")
    parser.add_argument("--val_categorize_id", default=1, help="测试集区块名")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")

    parser.add_argument("--gpu_id", default="0",  help="gpu的id")
    parser.add_argument("--pretrained", default=True, help="是否加载预训练模型")


    args = parser.parse_args()

    # assert args.file.endswith('.yaml'), 'You need to provide a .yaml file'
    return args


def run_epoch(model, train_loader, optimizer: torch.optim.Optimizer, criterion, args):
    """
    进行一轮训练
    :param model:
    :param train_loader:
    :param optimizer:
    :param criterion:
    :return: 准确率和损失
    """
    model.train()  # 切换成训练状态
    total_loss = []  # 记录每一个batch的损失函数
    label_nbr = 0 # 标签总数
    eq_nbr = 0 # 预测正确标签数
    cur_device = next(model.parameters()).device  # 获取当前设备
    print_period = 10  # 打印间隔
    start = time.time()

    for batch_idx, (feature, label) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        loss, predicted, label = common_forward(model, feature, label, criterion)
        label_nbr += len(label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        eq_nbr += predicted.eq(label).sum().item()
        total_loss.append(loss.item())

        if batch_idx % print_period == 0:
            use_time = time.time() - start
            start = time.time()
            print("Batch: {} / {}, Sliced: {} / {}, Loss: {:.4f}, Acc: {:.4f}%, Speed:{:.4f} sliced/s, Use Time: {:.4f} s".format(
                batch_idx, len(train_loader),
                label_nbr, len(train_loader.dataset),
                sum(total_loss) / len(total_loss),
                100 * eq_nbr / label_nbr,
                label_nbr / use_time,
                use_time))

    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss)

def eval_val(model, val_loader, criterion):
    val_acc, val_loss, all_label, all_predicted = evaluate(model, val_loader, criterion)
    return val_acc, val_loss, all_label, all_predicted

def main(args):
    """

    """
    set_seeds(args.seed)

    train_loader, val_loader, label_name = dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.train_well_num, args.val_well_num, args.frequency_aug, args.batchsize, args.train_categorize_id, args.val_categorize_id, args.noise_ration)
    device = torch.device("cuda:" + args.gpu_id)

    # 网络 + 优化器 + 学习率管理 + 损失函数
    # 参数的具体含义参照base_config.py
    model = Network(d_model=args.d_model, cell_num=args.cell_num, num_classes=args.num_classes, device=device, genotype=genotype)
    model.to(device)
    print("param size = %fMB", count_parameters_in_MB(model)) #记录参数大小

    # optimizer = optim.Adam(model.parameters(), lr=cfg.model.optimizer.learning_rate, weight_decay=cfg.model.optimizer.weight_decay)  # 优化器
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.model.scheduler.decay)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.decay)
    loss_func = torch.nn.CrossEntropyLoss()

    # 加载预训练参数
    print(Path(args.pretrained_filepath).exists())
    if args.pretrained == True and args.pretrained_filepath is not None and Path(args.pretrained_filepath).exists():
        print("-----------------加载预训练模型-----------------")
        loaded_model = torch.load(args.pretrained_filepath, map_location=torch.device("cpu"))
        net_dict = model.state_dict()
        # 判断model尺寸是否相同，仅加载相同的model
        # print(net_dict.keys())
        # print(loaded_model.keys())

        pretrained_dict = {k : v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}
        net_dict.update(pretrained_dict)
        model.load_state_dict(net_dict, strict=True)

    # 模型日志存储
    date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  # 加上时间
    args.log_dir_path = str((Path(args.log_dir_path) / date_time).resolve())
    Path(args.log_dir_path).mkdir(parents=True, exist_ok=True) # 确保父目录存在

    printcolor('---------------- 训练开始 ----------------')

    result_dict = {"trainloss": [], "valloss": [], "trainacc": [], "valacc": []}  # 用于保存记录到json里面的字典
    min_loss = 0x3f3f3f3f
    best_acc = -0x3f3f3f3f
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # 记录轮次信息
        tmp_strs = []
        for param_group in optimizer.param_groups:
            tmp_strs.append('Changing learning rate to {:8.10f}'.format(param_group['lr']))
        print_block("\n".join(tmp_strs), title='第' + str(epoch) + '轮')
        # 训练和评估
        train_acc, train_loss = run_epoch(model, train_loader, optimizer, loss_func, args)
        val_acc, val_loss, all_label, all_predicted = eval_val(model, val_loader, loss_func)
        # 更新学习率
        scheduler.step()

        # 保存训练信息到result中
        result_dict['trainloss'].append(train_loss)
        result_dict['valloss'].append(val_loss)
        result_dict['trainacc'].append(train_acc)
        result_dict['valacc'].append(val_acc)
        print_block("训练集loss: {:.4f}, 测试集loss: {:.4f}, 训练集acc: {:.4f}, 测试集acc: {:.4f}".
                    format(train_loss, val_loss, train_acc, val_acc), title="本次训练的结果")
        
        # 保存验证集表现最好的模型
        if best_acc < val_acc:
            best_acc = val_acc
            best_epoch = epoch
            print('best acc: {:.4f}%, best epoch: {}'.format(100 * best_acc, best_epoch))
            print('save model')
            savepath = str(Path(args.log_dir_path) / "best_epoch_model.pth")
            torch.save(model.state_dict(), savepath)

        # 保存json和图片
        with open(Path(args.log_dir_path) / 'logging.json', "w") as f:
            json.dump(result_dict, f, indent=2)
        
        save_train_val_fig(result_dict['trainacc'], result_dict['valacc'], 'train_acc', 'val_acc', "train and val accuracy", "epochs", "accuracy", str(Path(args.log_dir_path) / "train_val_acc.png"))
        save_train_val_fig(result_dict['trainloss'], result_dict['valloss'], 'train_loss', 'val_loss', "train and val loss", "epochs", "loss", str(Path(args.log_dir_path) / "train_val_loss.png"))

        
    print("Training best val acc: " + str(best_acc) + ", best epoch: " + str(best_epoch))
    print('Training complete, models saved in {}'.format(args.log_dir_path))


if __name__ == '__main__':
    """
    主要代码逻辑都写到main里
    注意, if __name__ == '__main__'仍然处于全局域里, 在这里面直接写逻辑会造成全局变量混乱
    """
    my_args = parse_args()
    main(my_args)
