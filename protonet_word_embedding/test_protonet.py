# 获取每一口测试井的准确率、召回率、F1分数

import numpy as np
import torch
import os
import sys
import random

from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# from model.transformer_searched import Network
from model.senet import SENet18
# from model.genotypes import Transformer_Encoder as genotype
from sklearn.metrics import recall_score, f1_score, accuracy_score
from data.dataloader import get_dataloader
import argparse

def parse_args():
    parser = argparse.ArgumentParser("few_shot_learing")
    parser.add_argument("--epochs", default=200, help="训练轮次")
    parser.add_argument("--classes", default=10, help="分类数")
    parser.add_argument("--learning_rate", default=0.0003, help="学习率")
    parser.add_argument("--decay", default=0.98, help="学习率衰减")
    parser.add_argument("--gpu_id", default="0",  help="gpu的id")
    parser.add_argument("--log_dir_path", default="log/resnet_log/", help="日志文件存储位置")
    parser.add_argument("--print_period", default=10, help="打印间隔")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=64, help="滑动步长")

    # 下面是可能需要改动的配置
    parser.add_argument("--pretrained", default=True, help="是否加载预训练模型")
    parser.add_argument("--pretrained_filepath", default="pretrain_model/senet_5/best_epoch_model.pth", help="预训练模型位置")
    parser.add_argument("--seed", default=42, help="随机数种子")
    parser.add_argument("--in_channel", default=5, help="测井曲线条数")
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

def compute_prototypes(trainLoader, model, device, n_cls):
    """
    根据测试集计算每个原型的距离
    """
    model.eval()
    prototypes = np.zeros((n_cls, 512))
    class_counts = np.zeros(n_cls)

    with torch.no_grad():
        for features, labels in trainLoader:
            features, labels = features.to(device), labels.to(device)
            proto = model(features)
            proto = proto.cpu().numpy()
            labels = labels.cpu().numpy()

            for cls in range(n_cls):
                class_features = proto[labels == cls]
                if class_features.size > 0:
                    prototypes[cls] += class_features.mean(axis=0)
                    class_counts[cls] += 1
    # 归一化原型
    for cls in range(n_cls):
        if class_counts[cls] > 0:
            prototypes[cls] /= class_counts[cls]
    
    return prototypes


def classify(testLoader, prototypes, model, device, n_cls):
    """
    计算验证集样本和每个类别原型的距离，给出标签信息
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feature, label in testLoader:
            feature = feature.to(device)
            all_labels.extend(label.cpu().numpy())

            query_features = model(feature)
            distances = np.linalg.norm(query_features.cpu().numpy()[:, np.newaxis] - prototypes, axis=2)
            preds = np.argmin(distances, axis=1)
            all_preds.extend(preds)

    return np.array(all_preds), np.array(all_labels)

def main(args):
    
    
    # model = Network(d_model=512, cell_num=4, num_classes=10, device=2, genotype=genotype)
    # loaded_model = torch.load("log/transformer_pth/best_epoch_model.pth", map_location=torch.device("cpu"))
    # model.load_state_dict(loaded_model)

    model = SENet18(in_channels=5, classes=10)
    trainLoader, testLoader = get_dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.train_well_num, args.val_well_num, args.frequency_aug, args.train_categorize_id, args.val_categorize_id, args.noise_ration, args.n_batch, args.n_cls, args.support + args.query)
    device = torch.device("cuda:" + args.gpu_id)
    model.to(device)
    model.load_state_dict(torch.load("log/protonet/12_22_2024__17_36_16/50/best_epoch_model.pth"))
    model.eval()
    model.training = False
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (feature, label) in enumerate(testLoader):
            feature, label = feature.to(device), label.to(device)
            p = args.support * args.n_cls
            data_support, data_query = feature[:p], feature[p:]
            query_label = torch.arange(args.n_cls).repeat(args.query)
            query_label = query_label.to(device)

            proto = model(data_support)
            proto = proto.reshape(args.support, args.n_cls, -1).mean(dim = 0)
            logits = euclidean_metric(model(data_query), proto)
            _, pred = torch.max(logits, dim = 1) 
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(query_label.cpu().numpy())            

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # prototypes = compute_prototypes(trainLoader, model, device, args.n_cls)
    # predicted_labels, true_labels = classify(testLoader, prototypes, model, device, args.n_cls)
    # accuracy = accuracy_score(true_labels, predicted_labels)
    # recall = recall_score(true_labels, predicted_labels, average='weighted')
    # f1 = f1_score(true_labels, predicted_labels, average='weighted')


    print(f"Average Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Recall: {recall * 100:.2f}%")
    print(f"Average F1 Score: {f1 * 100:.2f}%")



if __name__ == '__main__':
    args = parse_args()
    main(args)