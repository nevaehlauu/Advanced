import torch
import torch.nn as nn
import sys
from pathlib import Path
import os

curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(str(Path("./").resolve()))

import time
import json
import logging
import genotypes
import torch.nn.functional as F
from model import NetworkCIFAR as Network
from evaluate import evaluate_csv, common_forward
from config.darts_config import train_parse_args
# from data_228.readtxt_sample import dataloader
from data_228.readtxt_categorize import dataloader
from utils.utils import set_seeds, save_train_val_fig, count_parameters_in_MB, save_log

def train(train_loader, model, optimizer, criterion, args):
    total_loss = []
    label_nbr = 0
    eq_nbr = 0
    start = time.time()
    model.train()

    for batch_idx, (features, label) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()

        loss, predicted, label = common_forward(model, features, label, criterion)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        label_nbr += len(label)
        eq_nbr += predicted.eq(label).sum().item()
        total_loss.append(loss.item())

        if(batch_idx % args.report_freq == 0):
            use_time = time.time() - start
            start = time.time()
            logging.info('Batch: %03d / %03d, loss: %f, accuracy: %f', batch_idx, len(train_loader), sum(total_loss) / len(total_loss), eq_nbr / label_nbr)
        
    return sum(total_loss) / len(total_loss), eq_nbr / label_nbr

def val(model, val_loader, criterion):
    val_acc, val_loss, label, predicted = evaluate_csv(model, val_loader, criterion)
    return val_acc, val_loss, label, predicted

def main(args):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.seed is not None:
        set_seeds(args.seed)

    start = time.time()
    # 保存训练时模型日志参数，主要是准确率、损失图以及损失准确率日志，还有保存的pth文件
    args.save = save_log("eval", args)
    device = torch.device("cuda:" + args.gpu)
    logging.info("gpu device = {}".format(args.gpu))
    logging.info("args = %s", args)

    trainLoader, validLoader, label_name = dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.train_well_num, args.val_well_num, 
                                                      args.frequency_aug, args.batchsize, args.train_categorize_id, args.val_categorize_id, args.noise_ration)
    
    # trainLoader, validLoader, label_name = dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.train_well_num, 
    #                                                   args.val_well_num, args.frequency_aug, args.depth_aug, args.classification_name, args.batchsize, args)
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, len(label_name), args.layers, genotype)
    model.to(device)
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    # 优化权重w时，学习率调整用的是余弦退火（SGDR），但只训练50个epoch，其实就相当于cos学习率衰减，没有周期变化
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    print("-----------------训练开始---------------")
    
    # 在日志中记录准确率等值
    result_dict = {"trainloss": [], "valloss": [], "trainacc": [], "valacc": []}
    best_acc = -0x3f3f3f3f
    best_epoch = 0

    for epoch in range(args.epochs):

        print("---------第{}轮训练开始-------------".format(epoch))

        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info("epoch: {}, lr: {}".format(epoch, lr))
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_loss, train_acc = train(trainLoader, model, optimizer, criterion, args)
        val_acc, val_loss, val_label, val_predicted = val(model, validLoader, criterion)

        result_dict['trainloss'].append(train_loss)
        result_dict['valloss'].append(val_loss)
        result_dict['trainacc'].append(train_acc)
        result_dict['valacc'].append(val_acc)

        print("epoch: {}, train_acc: {:.4f}%, val_acc: {:.4f}%, train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, 100 * train_acc, 100 * val_acc, train_loss, val_loss
        ))

        save_train_val_fig(result_dict['trainacc'], result_dict['valacc'], 'train_acc', 'val_acc', "train and val accuracy", 
                           "epochs", "accuracy", str(Path(args.save) / "train_val_acc.png"))
        save_train_val_fig(result_dict['trainloss'], result_dict['valloss'], 'train_loss', 'val_loss', "train and val loss", 
                           "epochs", "loss", str(Path(args.save) / "train_val_loss.png"))

        # 保存结果最好的模型
        if best_acc < val_acc:
            best_acc = val_acc
            best_epoch = epoch
            print("best acc: {:4f}%, best epoch: {}".format(100 * best_acc, epoch))
            print("save model")
            savepath = str(Path(args.save) / "best_epoch_model.pth")
            torch.save(model.state_dict(), savepath)

        with open(Path(args.save) / 'logging.json', "w") as f:
            json.dump(result_dict, f, indent=2)
    
    print("Training best val acc: " + str(best_acc) + ", best epoch: " + str(best_epoch))
    print('Training complete, models saved in {}'.format(args.save))
    print('Training epoch: {} spend time: {}'.format(args.epochs, time.time() - start))



if __name__ == '__main__':
    args = train_parse_args()
    main(args)