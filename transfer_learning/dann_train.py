"""
DANN模型的训练：编码器，分类器，域判别器
"""

import torch
from torch.autograd import Variable
import torch.optim as optim
import argparse
from model.senet import SENet18
from model.dann import Classifier, Domain_classifier
from utils.utils import set_seeds, save_train_val_fig
from pathlib import Path
from data.dataloader import get_dataloader
from datetime import datetime
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser("few_shot_learing")
    parser.add_argument("--epochs", default=100, help="训练轮次")
    parser.add_argument("--classes", default=10, help="分类数")
    parser.add_argument("--learning_rate", default=0.0003, help="学习率")
    parser.add_argument("--decay", default=0.98, help="学习率衰减")
    parser.add_argument("--gpu_id", default="2",  help="gpu的id")
    parser.add_argument("--log_dir_path", default="log/dann_log/", help="日志文件存储位置")
    parser.add_argument("--print_period", default=10, help="打印间隔")
    parser.add_argument("--pretrained", default=True, help="是否加载预训练模型")
    parser.add_argument("--pretrained_filepath", default="pretrain_model/cluster_2/best_epoch_model_10.pth", help="预训练模型位置")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=64, help="滑动步长")
    parser.add_argument("--batchsize", default=1024)

    # 下面是可能需要改动的配置
    parser.add_argument("--seed", default=42, help="随机数种子")
    parser.add_argument("--in_channel", default=5, help="测井曲线条数")
    parser.add_argument("--src_train_well_num", default=2, help="源域进行训练的井数")
    parser.add_argument("--src_val_well_num", default=9, help="源域进行测试的井数")
    parser.add_argument("--tgt_train_well_num", default=2, help="目标域进行训练的井数")
    parser.add_argument("--tgt_val_well_num", default=13, help="目标域进行测试的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--src_categorize_id", default=2, help="源域使用的区块")
    parser.add_argument("--tgt_categorize_id", default=1, help="目标域区块名")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")

    args = parser.parse_args()
    return args

def train(feature_model, class_classifier, domain_classifier, source_dataloader, target_dataloader, optimizer, class_criterion, domain_criterion, print_period, epoch, total_epoch):
    feature_model.train()
    class_classifier.train()
    domain_classifier.train()
    cur_device = next(feature_model.parameters()).device

    # 记录三个损失：总损失，源域分类损失以及域判别器损失
    total_loss = []
    source_class_loss = []
    domian_class_loss = []

    # steps，用于计算constant和域损失函数的系数
    start_steps = epoch * len(source_dataloader)
    total_epoch = total_epoch * len(source_dataloader)

    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
        p = float(batch_idx + start_steps) / total_epoch
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        source_data, source_label = sdata
        target_data, target_label = tdata

        # 取最小批次大小，其实这里不需要，因为源域和目标域设置的batchsize相同
        size = min(source_data.shape[0], target_data.shape[0])
        source_data, source_label = source_data[0:size, :, :], source_label[0:size]
        target_data, target_label = target_data[0:size, :, :], target_label[0:size]

        source_data, source_label = source_data.to(cur_device), source_label.to(cur_device)
        target_data, target_label = target_data.to(cur_device), target_label.to(cur_device)

        optimizer.zero_grad()

        # 属于属于的领域标签
        source_domain_label = torch.zeros(source_data.size()[0], dtype=torch.long).to(cur_device)
        target_domain_label = torch.ones(target_data.size()[0], dtype=torch.long).to(cur_device)

        # 源域和目标域特征
        src_feature = feature_model(source_data)
        tgt_feature = feature_model(target_data)

        # 源域分类损失
        class_preds = class_classifier(src_feature)
        class_loss = class_criterion(class_preds, source_label.long())

        # 域判别器损失
        src_preds = domain_classifier(src_feature, constant)
        tgt_preds = domain_classifier(tgt_feature, constant)
        tgt_loss = domain_criterion(tgt_preds, target_domain_label)
        src_loss = domain_criterion(src_preds, source_domain_label)
        domain_loss = tgt_loss + src_loss

        # 总的损失函数，这里constant和损失计算的正则化系数相同，其实也可以替换为1或者跟着轮数变化的系数
        loss = class_loss + (epoch / total_epoch) * domain_loss
        total_loss.append(loss.item())
        source_class_loss.append(class_loss.item())
        domian_class_loss.append(domain_loss.item())

        loss.backward()
        optimizer.step()

        if batch_idx % print_period == 0:
            print("Batch: {} / {}, Loss: {:.4f}, Class Loss: {:.4f}, Domain Loss: {:.4f} s".format(
                batch_idx, len(source_dataloader),
                loss.item(),
                class_loss.item(),
                domain_loss.item()
            ))

    # 总损失，源域分类损失，域判别器损失
    return sum(total_loss) / len(total_loss), sum(source_class_loss) / len(source_class_loss), sum(domian_class_loss) / len(domian_class_loss)


def valid(feature_model, class_classifier, domain_classifier, source_dataloader, target_dataloader):
    feature_model.eval()
    class_classifier.eval()
    domain_classifier.eval()
    cur_device = next(feature_model.parameters()).device

    source_num = 0.0
    target_num = 0.0
    domain_num = 0.0 # 域判别器分类的个数
    src_num = 0.0 # 域判别器在源域数据的个数
    tgt_num = 0.0 # 域判别器在目标域个数
    total_source = 0
    total_target = 0

    for batch_idx, (source_data, source_label) in enumerate(source_dataloader):
        source_data, source_label = source_data.to(cur_device), source_label.to(cur_device)
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1.

        # 属于属于的领域标签
        source_domain_label = torch.zeros(source_data.size()[0], dtype=torch.long).to(cur_device)

        source_feature = feature_model(source_data)
        source_class_preds = class_classifier(source_feature)
        source_label_pred = source_class_preds.data.max(1, keepdim=True)[1] # .data用于访问张量的原始数据，.data 返回的张量进行操作不会影响原始张量的梯度计算。
        source_num += source_label_pred.eq(source_label.data.view_as(source_label_pred)).cpu().sum().item()
        
        src_preds = domain_classifier(feature_model(source_data), constant)
        src_preds = src_preds.data.max(1, keepdim=True)[1]
        src_num += src_preds.eq(source_domain_label.data.view_as(src_preds)).cpu().sum().item()

        total_source += len(source_label)
    
    for batch_idx, (target_data, target_label) in enumerate(target_dataloader):
        target_data, target_label = target_data.to(cur_device), target_label.to(cur_device)
        p = float(batch_idx) / len(target_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1.

        # 属于属于的领域标签
        target_domain_label = torch.zeros(target_data.size()[0], dtype=torch.long).to(cur_device)

        target_feature = feature_model(target_data)
        target_class_preds = class_classifier(target_feature)
        target_label_pred = target_class_preds.data.max(1, keepdim=True)[1] # .data用于访问张量的原始数据，.data 返回的张量进行操作不会影响原始张量的梯度计算。
        target_num += target_label_pred.eq(target_label.data.view_as(target_label_pred)).cpu().sum().item()
        
        tgt_preds = domain_classifier(feature_model(target_data), constant)
        tgt_preds = tgt_preds.data.max(1, keepdim=True)[1]
        tgt_num += tgt_preds.eq(target_domain_label.data.view_as(tgt_preds)).cpu().sum().item()

        total_target += len(target_label)
    
    domain_num = src_num + tgt_num

    # 源域分类准确率，目标域分类准确率，域判别器准确率，域判别器在源域准确率，域判别器在目标域准确率
    return  source_num / total_source, target_num / total_target, domain_num / (total_target + total_source), src_num / total_source, tgt_num / total_target


def main(args):
    set_seeds(args.seed)
    
    device = torch.device("cuda:" + args.gpu_id)
    feature_model = SENet18(in_channels=args.in_channel, classes=args.classes)
    class_classifier = Classifier()
    domain_classifier = Domain_classifier()
    feature_model.to(device)
    class_classifier.to(device)
    domain_classifier.to(device)

    src_train_dataloader, src_test_dataloader = get_dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.src_train_well_num, args.src_val_well_num, 
                                                               args.frequency_aug, args.src_categorize_id, args.src_categorize_id, args.noise_ration, args.batchsize)
    tgt_train_dataloader, tgt_test_dataloader = get_dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.tgt_train_well_num, args.tgt_val_well_num, 
                                                               args.frequency_aug, args.tgt_categorize_id, args.tgt_categorize_id, args.noise_ration, args.batchsize)
    
    # 分类和域判别的损失函数
    class_loss_func = torch.nn.CrossEntropyLoss()
    domain_loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(list(feature_model.parameters()) + list(class_classifier.parameters()) + list(domain_classifier.parameters()), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.decay)

    # 加载特征提取器的预训练参数
    if args.pretrained == True and args.pretrained_filepath is not None and Path(args.pretrained_filepath).exists():
        loaded_model = torch.load(args.pretrained_filepath, map_location=torch.device("cpu"))
        net_dict = feature_model.state_dict()
        # 判断model尺寸是否相同，仅加载相同的model
        pretrained_dict = {k : v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}
        net_dict.update(pretrained_dict)
        feature_model.load_state_dict(net_dict, strict=False)
    
    # 模型日志存储
    date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  # 加上时间
    args.log_dir_path = str((Path(args.log_dir_path) / date_time).resolve())
    Path(args.log_dir_path).mkdir(parents=True, exist_ok=True) # 确保父目录存在

    print("----------------训练开始----------------------")

    # 在日志中记录准确率等值
    result_dict = {"source_cls_acc": [], "target_cls_acc": [], "domain_acc": [], "source_domain_acc": [], "target_domain_acc":[]}
    best_acc = -0x3f3f3f3f
    best_epoch = 0

    for epoch in range(args.epochs):
        print("---------第{}轮训练开始-------------".format(epoch))

        # 总损失，源域分类损失，域判别器损失
        epoch_loss, source_cls_loss, domain_loss = train(feature_model, class_classifier, domain_classifier, src_train_dataloader, tgt_train_dataloader, optimizer, class_loss_func, 
                                                         domain_loss_func, args.print_period, epoch, args.epochs)
        
        # 源域分类准确率，目标域分类准确率，域判别器准确率，域判别器在源域准确率，域判别器在目标域准确率
        source_cls_acc, target_cls_acc, domain_acc, source_domain_acc, target_domain_acc = valid(feature_model, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader)
        scheduler.step()

        result_dict["source_cls_acc"].append(source_cls_acc)
        result_dict["target_cls_acc"].append(target_cls_acc)
        result_dict["domain_acc"].append(domain_acc)
        result_dict["source_domain_acc"].append(source_domain_acc)
        result_dict["target_domain_acc"].append(target_domain_acc)

        print("epoch: {}, source_cls_acc: {:.4f}%, target_cls_acc: {:.4f}%, domain_acc: {:.4f}%, source_domain_acc: {:.4f}%, target_domain_acc: {:.4f}%".format(
            epoch, 100 * source_cls_acc, 100 * target_cls_acc, 100 * domain_acc, 100 * source_domain_acc, 100 * target_domain_acc
        ))

        if target_cls_acc > best_acc:
            best_acc = target_cls_acc
            best_epoch = epoch
            print("best acc: {:4f}%, best epoch: {}".format(100 * best_acc, epoch))
            print("save model")
            savepath = str(Path(args.log_dir_path) / "best_epoch_model.pth")
            torch.save(feature_model.state_dict(), savepath)

        with open(Path(args.log_dir_path) / 'logging.json', "w") as f:
            json.dump(result_dict, f, indent=2)
    
    print("Training best val acc: " + str(best_acc) + ", best epoch: " + str(best_epoch))
    print('Training complete, models saved in {}'.format(args.log_dir_path))

if __name__ == '__main__':

    my_args = parse_args()
    main(my_args)