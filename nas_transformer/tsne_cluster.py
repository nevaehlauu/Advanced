"""
对不同区块数据进行可视化：
也就是通过模型将每个区块测井数据映射到特征空间，将不同区块数据标为不同颜色
"""

import torch
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader, TensorDataset 
from model.senet import SENet18 as Net
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def parse_args():
    parser = argparse.ArgumentParser("few_shot_learing")
    parser.add_argument("--epochs", default=100, help="训练轮次")
    parser.add_argument("--classes", default=10, help="分类数")
    parser.add_argument("--learning_rate", default=0.0003, help="学习率")
    parser.add_argument("--decay", default=0.98, help="学习率衰减")
    parser.add_argument("--gpu_id", default="2",  help="gpu的id")
    parser.add_argument("--log_dir_path", default="log/resnet_log/", help="日志文件存储位置")
    parser.add_argument("--print_period", default=10, help="打印间隔")
    parser.add_argument("--pretrained", default=True, help="是否加载预训练模型")
    parser.add_argument("--pretrained_filepath", default="pretrain_pth/dann_model.pth", help="预训练模型位置")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_data/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=64, help="滑动步长")
    parser.add_argument("--batchsize", default=2048)

    # 下面是可能需要改动的配置
    parser.add_argument("--seed", default=42, help="随机数种子")
    parser.add_argument("--in_channel", default=5, help="测井曲线条数")
    parser.add_argument("--train_well_num", default=57, help="进行训练的井数")
    parser.add_argument("--val_well_num", default=9, help="进行训练的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--train_categorize_id", default=1, help="训练集使用的区块")
    parser.add_argument("--val_categorize_id", default=2, help="测试集区块名")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")

    args = parser.parse_args()
    return args

def _read_dir(path, categorize_id):
    """
    根据txt文件中包含的区块信息，随机从某个指定区块中挑选指定数量的井
    """
    data_name = {#1: ['W1424', 'W1584', 'W1585', 'W1586', 'W1587', 'W1588', 'W1589', 'W1590', 'W1591', 'W1594', 'W1595', 'W1596', 'W1597', 'W1610', 'W1612', 'W1613', 'W1614', 'W1629', 'W1630', 'W1633', 
                #         'W1639', 'W1640', 'W1642', 'W1645', 'W1646', 'W1648', 'W1652', 'W1654', 'W1663', 'W1667', 'W1668', 'W1670', 'W1672', 'W1676', 'W1679', 'W1685', 'W1690', 'W1691', 'W1693', 'W1694', 
                #         'W1695', 'W1696', 'W1697', 'W1698', 'W1702', 'W1704', 'W1709', 'W651', 'W652', 'W654', 'W655', 'W656', 'W657', 'W658', 'W660', 'W661', 'W662', 'W664', 'W665', 'W666', 'W667', 'W669', 
                #         'W670', 'W671', 'W675', 'W676', 'W677', 'W679', 'W681', 'W683', 'W686', 'W689', 'W691', 'W692', 'W695', 'W696', 'W697', 'W698', 'W699', 'W700', 'W701', 'W702', 'W703', 'W704', 'W705', 
                #         'W706', 'W708', 'W709', 'W711', 'W722', 'W766', 'W782', 'W1615', 'W1628', 'W1669', 'W1675', 'W1686', 'W1707', 'W653', 'W663', 'W668', 'W678', 'W694', 'W707', 'W710'], 
                # 2: ['W792', 'W793', 'W794', 'W795', 'W796', 'W801', 'W802', 'W803', 'W804', 'W805', 'W806', 'W807', 'W809', 'W811', 'W812', 'W813', 'W814', 'W815', 'W816', 'W817', 'W818', 'W820', 'W821', 'W823', 
                #         'W824', 'W825', 'W826', 'W828', 'W831', 'W832', 'W834', 'W835', 'W837', 'W840', 'W842', 'W844', 'W845', 'W846', 'W847', 'W848', 'W849', 'W850', 'W851', 'W852', 'W853', 'W856', 'W857', 'W858', 
                #         'W859', 'W861', 'W863', 'W864', 'W866', 'W868', 'W869', 'W870', 'W871', 'W810', 'W822', 'W827', 'W830', 'W833', 'W843', 'W854', 'W855', 'W860'],
                # 3: ['W101', 'W113', 'W115', 'W118', 'W119', 'W125', 'W137', 'W138', 'W145', 'W147', 'W149', 'W150', 'W152', 'W160', 'W161', 'W162', 'W170', 'W172', 'W173', 'W174', 'W176', 'W177', 'W35'], 
                # 4: ['W381', 'W385', 'W394', 'W405', 'W407', 'W408'], 
                # 5: ['W188', 'W574', 'W59', 'W60', 'W64', 'W69', 'W74', 'W189', 'W62', 'W63'], 
                # 6: ['W568', 'W605', 'W613', 'W614', 'W618', 'W619', 'W620', 'W621', 'W625', 'W626', 'W627', 'W631', 'W615', 'W634'], 
                # -1: ['W421', 'W427', 'W584']
                # 2: ['W1145', 'W1146', 'W1147', 'W1148', 'W1149', 'W1150', 'W1154', 'W1155', 'W1156', 'W1157', 'W1160', 'W1162', 'W1165', 'W1169', 'W1174', 'W1181', 'W1186', 'W1191', 'W1192', 'W1193', 
                #     'W1194', 'W1195', 'W1196', 'W1197', 'W1198', 'W1199', 'W1200', 'W1202', 'W1205', 'W1209', 'W1215', 'W1564', 'W1565', 'W1566', 'W1567', 'W1568', 'W1569', 'W1570', 'W1573', 'W1574', 
                #     'W1575', 'W1577', 'W1578', 'W1579', 'W1580', 'W1581', 'W1739', 'W1767', 'W1784', 'W1785', 'W1802', 'W1823', 'W1835', 'W1844', 'W2351', 'W2362', 'W2367', 'W2369', 'W2373', 'W2374', 
                #     'W2376', 'W2438', 'W2440', 'W2444', 'W2445', 'W2449', 'W2451', 'W2456', 'W2460', 'W2579', 'W2618', 'W2622', 'W2661', 'W2664', 'W2675', 'W2681', 'W2722', 'W2729', 'W2738', 'W2741', 
                #     'W2748', 'W2751', 'W2752', 'W2753', 'W2754']

                1: ['W651', 'W652', 'W654', 'W655', 'W656', 'W657', 'W658', 'W660', 'W661', 'W662'], 
                2: ['W871', 'W810', 'W822', 'W827', 'W830', 'W833', 'W843', 'W854', 'W855', 'W860'],
                }
    
    # 随机挑选符合条件的井，为了随机挑井，这里需要随机得到一个随机数种子
    
    use_well_name = data_name[categorize_id]
    print("--------------", len(use_well_name))
    print("使用的区块名：" + str(categorize_id))

    dir = []
    dir_list = os.listdir(path) # 列出了指定路径 self.path 下的所有文件和子目录
    for name in dir_list:
        file_name = name.split('.')[0]
        if file_name in use_well_name:
            dir_path = os.path.join(path, name)
            dir.append(dir_path)
    return dir

def read_data_from_csv(dir, slice_length, slice_step, categorize_id):
    """
    dir: 井的位置
    dir_label: 与dir相对的训练集或验证集信息，用来的得到训练集和验证集完整的标签信息
    label_name: 标签信息
    slice_length: 切片长
    slice_step: 滑动步长
    train_well_num: 需要处理的井数
    frequency_aug: 是否进行频域增广，以及进行频域增广的类型，包含傅里叶变换和小波变换
    depth_aug: 是否进行时域增广，以及进行时域增广的类型，包含直接添加高斯噪声，
    classification_name: 任务类型，如果为地质分层，应该选择第8列数据，如果为储层划分，第九列数据，如果为油气水划分，应该选择最后一列数据，并且去除切片中标签为0的切片

    注意：深度曲线不应该进行任何数据增广操作
    """
    Well_path = _read_dir(dir, categorize_id)
    print("-------------使用的井数----------------: ", len(Well_path))

    label_name = {'K1z2+1':   0, 
                  'J2a':      1, 
                  'J2z':      2, 
                  'J1y':      3, 
                  'J1f':      4, 
                  'chang1':   5, 
                  'chang2':   6, 
                  'chang3':   7, 
                  'chang4+5': 8,
                  'chang6':   9
                 }

    # 读取每一口井的数据，对其进行操作
    sliced_data = []
    sliced_label = []
    for well in Well_path:
        data_all = pd.read_csv(well, header=None)
        data_value = data_all.iloc[:, 3:8].values # .values将数据转换为numpy数组，第 0 维通常表示行。所以 [:, 2:8] 中的 : 表示选择了所有行,得到的数据形式Numpy数组,且shape为(n, 6)
        # 首先对所有数据进行归一化操作
        data_value = scale(data_value, axis=0, with_mean=True, with_std=True, copy=True)
        data_label = data_all.iloc[:, 8].values
        aug_value = []

        for i in range(data_value.shape[1]):
            single_value = data_value[:, i]
            aug_value.append(single_value)
        aug_value = np.array(aug_value)
        data_value = aug_value

        slice_num = (data_label.shape[0] - slice_length) // slice_step + 1 # 切片个数
        for i in range(slice_num):
            start = i * slice_step
            end = start + slice_length
            # 取中间点标签作为一段切片标签
            mid_index = (start + end) // 2
            a_slice = data_value[:, start:end]

            sliced_data.append(a_slice)
            labels = 1
            sliced_label.append(labels)
        
    sliced_data = torch.from_numpy(np.array(sliced_data)).float()
    sliced_label = torch.from_numpy(np.array(sliced_label, dtype=np.float32))
    return sliced_data, sliced_label

def dataloader(dir_train, dir_val, slice_length, slice_step, batchsize, train_categorize_id, val_categorize_id):
    train_data, train_label = read_data_from_csv(dir_train, slice_length, slice_step, train_categorize_id)
    valid_data, valid_label = read_data_from_csv(dir_val, slice_length, slice_step, val_categorize_id)
    print("train_data.shape: ", train_data.shape)
    print("train_label.shape: ", train_label.shape)
    print("val_data.shape: ", valid_data.shape)
    print("val_label.shape: ", valid_label.shape)

    train_set = TensorDataset(train_data, train_label)
    valid_set = TensorDataset(valid_data, valid_label)

    trainLoader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True, num_workers=0)
    validLoader = DataLoader(dataset=valid_set, batch_size=batchsize, shuffle=True, num_workers=0)

    return trainLoader, validLoader

def main(args):
    trainLoader, validLoader = dataloader(args.train_dir, args.train_dir, args.slice_length, args.slice_step, args.batchsize, args.train_categorize_id, args.val_categorize_id)
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
    for batch_ind, (feature, label) in enumerate(trainLoader):
        with torch.no_grad():
            feature = model(feature)
        feature = feature.view(feature.size(0), -1).cpu()
        for f in feature:
            features.append(f.numpy())
            labels.append(0)
    
    for batch_ind, (feature, label) in enumerate(validLoader):
        with torch.no_grad():
            feature = model(feature)
        feature = feature.view(feature.size(0), -1).cpu()
        for f in feature:
            features.append(f.numpy())
            labels.append(1)
    
    features = np.array(features)
    labels = np.array(labels)
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
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], s=50, edgecolor='white', linewidth=1,
                    label=f'Class {label}', color=color_map[label], alpha=1)

    plt.savefig("well_scatter_cluster_2.png", dpi=800)
    plt.close()

if __name__ == '__main__':

    my_args = parse_args()
    main(my_args)