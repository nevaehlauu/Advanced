"""
通过t-SNE算法对测井数据的可视化
"""
from sklearn.manifold import TSNE
import torch
import os
from model.senet import SENet18 as Net
from model.dann import Classifier, Domain_classifier
from data.dataloader import get_dataloader
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils.utils import set_seeds

def parse_args():
    parser = argparse.ArgumentParser("few_shot_learing")
    parser.add_argument("--epochs", default=100, help="训练轮次")
    parser.add_argument("--classes", default=10, help="分类数")
    parser.add_argument("--learning_rate", default=0.0003, help="学习率")
    parser.add_argument("--decay", default=0.98, help="学习率衰减")
    parser.add_argument("--gpu_id", default="3",  help="gpu的id")
    parser.add_argument("--log_dir_path", default="log/dann_log/", help="日志文件存储位置")
    parser.add_argument("--print_period", default=10, help="打印间隔")
    parser.add_argument("--pretrained", default=True, help="是否加载预训练模型")
    parser.add_argument("--pretrained_filepath", default="pretrain_model/senet_5/best_epoch_model.pth", help="预训练模型位置")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=64, help="滑动步长")
    parser.add_argument("--batchsize", default=1024)

    # 下面是可能需要改动的配置
    parser.add_argument("--seed", default=42, help="随机数种子")
    parser.add_argument("--in_channel", default=5, help="测井曲线条数")
    parser.add_argument("--src_train_well_num", default=10, help="源域进行训练的井数")
    parser.add_argument("--src_val_well_num", default=13, help="源域进行测试的井数")
    parser.add_argument("--tgt_train_well_num", default=10, help="目标域进行训练的井数")
    parser.add_argument("--tgt_val_well_num", default=9, help="目标域进行测试的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--src_categorize_id", default=1, help="源域使用的区块")
    parser.add_argument("--tgt_categorize_id", default=2, help="目标域区块名")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")

    args = parser.parse_args()
    return args

def plot_embedding(X, y, d, title=None, imgName=None):
    """
    Plot an embedding X with the class label y colored by the domain d.

    :param X: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param imgName: the name of saving image

    :return:
    """
    # # normalization
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,8))
    ax = plt.subplot(111)

    shapes = ['o', '^'] # 圆形和三角形

    # 自定义颜色映射
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


    for i in range(X.shape[0]):
        # plt.text(X[i, 0], X[i, 1], str(y[i]),
        #          color=plt.cm.bwr(d[i]/1.),
        #          fontdict={'weight': 'bold', 'size': 9})
        color = color_map[y[i]] # 不同标签不同颜色
        market = shapes[d[i]] # 不同领域不同颜色
        ax.scatter(X[i, 0], X[i, 1], color = color, marker=market, s = 60, edgecolors='white', linewidths=1) # s是点的大小


    # plt.xticks([]), plt.yticks([])

    print('Saving ' + imgName + ' ...')
    plt.savefig(imgName, dpi=800)
    plt.close()

def main(args):
    
    set_seeds(args.seed)
    
    device = torch.device("cuda:" + args.gpu_id)
    feature_model = Net(in_channels=args.in_channel, classes=args.classes)
    class_classifier = Classifier()
    domain_classifier = Domain_classifier()
    feature_model.to(device)
    class_classifier.to(device)
    domain_classifier.to(device)
    feature_model.eval()
    class_classifier.eval()
    domain_classifier.eval()

    src_train_dataloader, src_test_dataloader = get_dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.src_train_well_num, args.src_val_well_num, 
                                                               args.frequency_aug, args.src_categorize_id, args.src_categorize_id, args.noise_ration, args.batchsize)
    tgt_train_dataloader, tgt_test_dataloader = get_dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.tgt_train_well_num, args.tgt_val_well_num, 
                                                               args.frequency_aug, args.tgt_categorize_id, args.tgt_categorize_id, args.noise_ration, args.batchsize)
    

    # 加载预训练模型
    if args.pretrained == True and args.pretrained_filepath is not None and Path(args.pretrained_filepath).exists():
        loaded_model = torch.load(args.pretrained_filepath, map_location=torch.device("cpu"))
        net_dict = feature_model.state_dict()
        # 判断model尺寸是否相同，仅加载相同的model
        pretrained_dict = {k : v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}
        net_dict.update(pretrained_dict)
        feature_model.load_state_dict(net_dict, strict=False)

    s_data, s_labels, s_tags = [], [], []
    for batch in src_train_dataloader:
        data, label = batch
        data = data.to(device)
        s_data.append(data)
        s_labels.append(label)
        s_tags.append(torch.zeros(label.size()[0], dtype=torch.long))
    s_data, s_labels, s_tags = torch.cat(s_data)[:args.batchsize], torch.cat(s_labels)[:args.batchsize], torch.cat(s_tags)[:args.batchsize]

    t_data, t_labels, t_tags = [], [], []
    for batch in tgt_train_dataloader:
        data, label = batch
        data = data.to(device)
        t_data.append(data)
        t_labels.append(label)
        t_tags.append(torch.ones(label.size()[0], dtype=torch.long))
    t_data, t_labels, t_tags = torch.cat(t_data)[:args.batchsize], torch.cat(t_labels)[:args.batchsize], torch.cat(t_tags)[:args.batchsize]

    embedding_s = feature_model(s_data)
    embedding_t = feature_model(t_data)
    # print(embedding_s.shape)

    # init='pca': 使用主成分分析（PCA）来初始化 t-SNE 的计算，这有助于加速收敛。n_iter=3000: 指定优化迭代的次数，3000 次迭代通常足以使 t-SNE 收敛到合适的嵌入表示。
    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    tsne = TSNE(n_components=2, random_state=0)
    dann_tsne = tsne.fit_transform(np.concatenate((embedding_s.cpu().detach().numpy(), embedding_t.cpu().detach().numpy())))

    plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)), np.concatenate((s_tags, t_tags)), 'Domain Adaptation', "domain_2.png")

    # # 自定义颜色映射
    # # 假设有 3 个类别，定义每个类别的颜色
    # color_map = {
    #     0: '#D42728',   # 红
    #     1: '#767171',   # 灰
    #     2: '#248024',   # 绿
    #     3: '#2F5597',   # 蓝
    #     4: '#95BEE3',   # 天蓝
    #     5: '#C55A11',   # 橙色
    #     6: '#E286C8',   # 粉紫色
    #     7: '#FFD966',   # 黄色
    #     8: '#8F67AE',   # 紫色
    #     9: '#B9B839'    # 
    # }


    # # 可视化 t-SNE
    # plt.figure(figsize=(10, 8))
    # for label in np.unique(labels):
    #     indices = np.where(labels == label)
    #     plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], s=90, edgecolor='white', linewidth=1,
    #                 label=f'Class {label}', color=color_map[label], alpha=1)
    
    # plt.savefig("well_scatter_val_pre.png", dpi=800)
    # plt.close()

if __name__ == '__main__':

    my_args = parse_args()
    main(my_args)