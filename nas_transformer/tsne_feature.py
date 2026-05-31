"""
数据分布的可视化，按照标签
"""
from sklearn.manifold import TSNE
import torch
from model.senet import SENet18, Net
from dataloader.dataloader import dataloader
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser("few_shot_learing")
    parser.add_argument("--epochs", default=100, help="训练轮次")
    parser.add_argument("--classes", default=10, help="分类数")
    parser.add_argument("--learning_rate", default=0.0003, help="学习率")
    parser.add_argument("--decay", default=0.98, help="学习率衰减")
    parser.add_argument("--gpu_id", default="2",  help="gpu的id")
    parser.add_argument("--log_dir_path", default="log/resnet_log/", help="日志文件存储位置")
    parser.add_argument("--print_period", default=10, help="打印间隔")
    parser.add_argument("--pretrained", default=False, help="是否加载预训练模型")
    parser.add_argument("--pretrained_filepath", default="log/senet_log/12_27_2024__04_14_16/best_epoch_model.pth", help="预训练模型位置")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=64, help="滑动步长")
    parser.add_argument("--batchsize", default=2048)

    # 下面是可能需要改动的配置
    parser.add_argument("--seed", default=42, help="随机数种子")
    parser.add_argument("--in_channel", default=5, help="测井曲线条数")
    parser.add_argument("--train_well_num", default=10, help="进行训练的井数")
    parser.add_argument("--val_well_num", default=9, help="进行训练的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--train_categorize_id", default=2, help="训练集使用的区块")
    parser.add_argument("--val_categorize_id", default=2, help="测试集区块名")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")

    args = parser.parse_args()
    return args

def main(args):

    trainLoader, validLoader, label_name = dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.train_well_num, args.val_well_num, args.frequency_aug, args.batchsize, args.train_categorize_id, args.val_categorize_id, args.noise_ration)
    model = Net(in_channels=args.in_channel, classes=args.classes)

    # 加载预训练模型
    if args.pretrained == True and args.pretrained_filepath is not None and Path(args.pretrained_filepath).exists():
        loaded_model = torch.load(args.pretrained_filepath, map_location=torch.device("cpu"))
        net_dict = model.state_dict()
        # 判断model尺寸是否相同，仅加载相同的model
        pretrained_dict = {k : v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}
        net_dict.update(pretrained_dict)
        model.load_state_dict(net_dict, strict=False)

    features = []
    labels = []
    for batch_ind, (feature, label) in enumerate(validLoader):
        with torch.no_grad():
            feature = model(feature)
        feature = feature.view(feature.size(0), -1).cpu()
        for f, l in zip(feature, label):
            features.append(f.numpy())
            labels.append(l.numpy())
    
    features = np.array(features)
    labels = np.array(labels)
    # tsne = TSNE(n_components=2, random_state=0, init='pca', n_iter=3000)
    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(features)

    # 自定义颜色映射
    # 假设有 3 个类别，定义每个类别的颜色
    color_map = {
        0: '#D42728',   # 红
        1: '#767171',   # 灰
        2: '#248024',   # 绿
        3: '#2F5597',   # 蓝
        4: '#95BEE3',   # 天蓝
        5: '#C55A11',   # 橙色
        6: '#E286C8',   # 粉紫色
        7: '#FFD966',   # 黄色
        8: '#8F67AE',   # 紫色
        9: '#B9B839'    # 
    }


    # 可视化 t-SNE
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], s=90, edgecolor='white', linewidth=1,
                    label=f'Class {label}', color=color_map[label], alpha=1)
        # plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], 
        #             label=f'Class {label}', color=color_map[label], alpha=0.5)
    
    # scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='jet', alpha=0.5)
    # plt.colorbar(scatter)
    # plt.title('t-SNE Visualization of Features')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    plt.savefig("well_scatter_cluster_2_val_dann.png", dpi=800)
    plt.close()

if __name__ == '__main__':

    my_args = parse_args()
    main(my_args)