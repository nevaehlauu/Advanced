"""
搜索Transformer模型的训练过程
"""
import os
import sys
import time
import glob
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

"""
根据测井数据集搜索适合的Transformer模型（搜索self-attention的num_head和mlp的mlp_ratio）
"""
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import yaml
from utils.evaluate import evaluate
from dataloader.dataloader import dataloader
from model.transformer_search import Network
from utils.architect import Architect

"""
训练代码
在根目录下运行该脚本
"""
import json
from datetime import datetime
from pathlib import Path

# 和上面rootPath功能都是将当前路径添加到Python的系统路径中
sys.path.append(str(Path("./").resolve()))

from utils.utils import print_block, set_seeds, printcolor, save_fig, sample_to_device, count_parameters_in_MB


def parse_args():
    """
    获取参数
    :return: args
    """
    parser = argparse.ArgumentParser(description='Train a model')
    
    parser = argparse.ArgumentParser("few_shot_learing")
    parser.add_argument("--epochs", default=50, help="训练轮次")
    parser.add_argument("--learning_rate", default=0.025, help="学习率")
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='L2 正则化系数,用于防止模型过拟合')
    parser.add_argument("--report_freq", default=10, help="打印间隔")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")

    parser.add_argument("--pretrained_filepath", default="log/senet_pth/best_epoch_model.pth", help="预训练模型位置")
    parser.add_argument('--d_model', default=512, help='token维度')
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=64, help="滑动步长")
    parser.add_argument('--log_dir_path', type=str, default="log/transformer_search", help='文件保存的地址')    
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping') #梯度裁剪阈值，用于控制梯度的最大范围，以避免梯度爆炸的问题。
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss') #是否使用one-short策略展开验证损失
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding') #学习率
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding') #权重损失

    # 下面是可能需要改动的配置
    parser.add_argument("--seed", default=100, help="随机数种子")
    parser.add_argument('--cell_num', type=int, default=8, help='搜索时模型堆叠层数')
    parser.add_argument("--batchsize", default=512, help="")
    parser.add_argument("--train_well_num", default=2, help="进行训练的井数")
    parser.add_argument("--val_well_num", default=13, help="进行训练的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--train_categorize_id", default=1, help="训练集使用的区块")
    parser.add_argument("--val_categorize_id", default=1, help="测试集区块名")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")
    parser.add_argument("--gpu", default="0",  help="gpu的id")
    parser.add_argument("--pretrained", default=True, help="是否加载预训练模型")

    args = parser.parse_args()
    return args


def run_epoch(train_loader, valid_loader, model, architect, optimizer: torch.optim.Optimizer, criterion, lr, args):
    """
    进行一轮训练
    :param model:
    :param train_loader:
    :param optimizer:
    :param criterion:
    :return: 准确率和损失
    """
    total_loss = []  # 记录每一个batch的损失函数
    label_nbr = 0 # 标签总数
    eq_nbr = 0 # 预测正确标签数
    cur_device = next(model.parameters()).device  # 获取当前设备
    print_period = 10  # 打印间隔
    start = time.time()

    for batch_idx, (feature, label) in enumerate(train_loader):
        model.train()  # 切换成训练状态
        # model.zero_grad()
        #######NAS搜索时前向传播相较于普通的前向传播有变化
        feature, label = sample_to_device(feature, device=cur_device), sample_to_device(label, device=cur_device)
        feature = feature.detach().requires_grad_(False)
        label = label.long().detach().requires_grad_(False)

        feature_search, label_search = next(iter(valid_loader))
        feature_search, label_search = sample_to_device(feature_search, device=cur_device), sample_to_device(label_search, device=cur_device)
        feature_search = feature_search.detach().requires_grad_(False)
        label_search = label_search.long().detach().requires_grad_(False)

        ### 优化部分：DARTS是交替优化的，第一步先优化alpha，第二步再优化w
    
        # 更新架构权重alpha，unrolled为True时就是用论文的公式进行alpha的更新
        architect.step(feature, label, feature_search, label_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        output = model(feature)
        loss = criterion(output, label)

        _, predicted = output.max(1)

        # loss, predicted, label = common_forward(model, batch, cur_device, criterion, features_name)
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
            logging.info('train %03d %03d %f %f', batch_idx, len(train_loader), sum(total_loss) / len(total_loss),  100 * eq_nbr / label_nbr)

    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss)

def eval_val(model, val_loader, criterion):
    val_acc, val_loss, all_label, all_predicted = evaluate(model, val_loader, criterion)
    return val_acc, val_loss, all_label, all_predicted

def main(args):
    """

    """
    set_seeds(args.seed)

    # 获取训练集
    train_loader, val_loader, label_name = dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.train_well_num, args.val_well_num, args.frequency_aug, args.batchsize, args.train_categorize_id, args.val_categorize_id, args.noise_ration)

    # 网络 + 优化器 + 学习率管理 + 损失函数
    # 参数的具体含义参照base_config.py
    device = torch.device("cuda:" + args.gpu)
    loss_func = nn.CrossEntropyLoss()
    loss_func.to(device=device)
    model = Network(d_model=args.d_model, cell_num=args.cell_num, num_classes=len(label_name), criterion=loss_func, device=device)
    model = model.to(device)
    logging.info("param size = %fMB", count_parameters_in_MB(model))
    logging.info('gpu device = %s', device)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.model.optimizer.learning_rate, weight_decay=cfg.model.optimizer.weight_decay)  # 优化器
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.model.scheduler.decay)

    #####优化器
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    # 加载预训练参数
    print("------------" + str(Path(args.pretrained_filepath).exists()))
    if args.pretrained == True and args.pretrained_filepath is not None and Path(args.pretrained_filepath).exists():
        loaded_model = torch.load(args.pretrained_filepath, map_location=torch.device("cpu"))
        net_dict = model.embedding.state_dict()
        pretrained_dict = {k : v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}
        net_dict.update(pretrained_dict)
        model.embedding.load_state_dict(net_dict, strict=True)
        print("------------预训练参数加载成功--------------")

    # 模型日志存储
    date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  # 加上时间
    args.log_dir_path = str((Path(args.log_dir_path) / date_time).resolve())
    Path(args.log_dir_path).mkdir(parents=True, exist_ok=True) # 确保父目录存在

    printcolor('---------------- 训练开始 ----------------')

    result_dict = {"trainloss": [], "valloss": [], "trainacc": [], "valacc": []}  # 用于保存记录到json里面的字典
    min_loss = 0x3f3f3f3f
    best_acc = -0x3f3f3f3f
    best_epoch = 0
    best_genotype = None
    
    for epoch in range(args.epochs):
        # 更新学习率
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # 选出权重最大的前k个前驱节点
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print("left_attention_alphas: ", F.softmax(model.left_attention_alphas, dim=-1))
        print("left_mlp_alphas: ", F.softmax(model.left_mlp_alphas, dim=-1))

        # 记录轮次信息
        tmp_strs = []
        for param_group in optimizer.param_groups:
            tmp_strs.append('Changing learning rate to {:8.10f}'.format(param_group['lr']))
        print_block("\n".join(tmp_strs), title='第' + str(epoch) + '轮')
        # 训练和评估
        train_acc, train_loss = run_epoch(train_loader, val_loader, model, architect, optimizer, loss_func, lr, args)
        logging.info('train_acc %f', train_acc)
        val_acc, val_loss, all_label, all_predicted = eval_val(model, val_loader, loss_func)
        logging.info('valid_acc %f', val_acc)

        # 保存训练信息到result中
        result_dict['trainloss'].append(train_loss)
        result_dict['valloss'].append(val_loss)
        result_dict['trainacc'].append(train_acc)
        result_dict['valacc'].append(val_acc)
        print_block("训练集loss: {:.4f}, 测试集loss: {:.4f}, 训练集acc: {:.4f}, 测试集acc: {:.4f}".
                    format(train_loss, val_loss, train_acc, val_acc), title="本次训练的结果")

        # 保存结果最好的模型
        if best_acc < val_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_genotype = genotype
            print("best acc: {:4f}%, best epoch: {}".format(100 * best_acc, best_epoch))
            print("--------save model-----------")
            savepath = str(Path(args.log_dir_path) / "best_epoch_model.pth")
            torch.save(model.state_dict(), savepath)

        # 保存json和图片
        with open(Path(args.log_dir_path) / 'logging.json', "w") as f:
            json.dump(result_dict, f, indent=2)

        save_fig(result_dict['trainloss'], "train loss", str(Path(args.log_dir_path) / "train_loss.png"))
        save_fig(result_dict['valloss'], "val loss", str(Path(args.log_dir_path) / "val_loss.png"))
        save_fig(result_dict['trainacc'], "train acc", str(Path(args.log_dir_path) / "train_acc.png"))
        save_fig(result_dict['valacc'], "val acc", str(Path(args.log_dir_path) / "val_acc.png"))

    print("Training best val acc: " + str(best_acc) + ", best epoch: " + str(best_epoch))
    print("Training best val genotype: " + str(best_genotype))
    printcolor('Training complete, models saved in {}'.format(args.log_dir_path), "green")


if __name__ == '__main__':
    """
    主要代码逻辑都写到main里
    注意, if __name__ == '__main__'仍然处于全局域里, 在这里面直接写逻辑会造成全局变量混乱
    """
    my_args = parse_args()
    main(my_args)
