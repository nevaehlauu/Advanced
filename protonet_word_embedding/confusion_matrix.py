"""
混淆矩阵
"""
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from sklearn.metrics import confusion_matrix
from model.senet import SENet18 as Net
from model.word_embedding import wordEmbTransformer
from utils.utils import euclidean_metric, set_seeds
from data.dataloader import get_dataloader


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
    parser.add_argument("--pretrained_filepath", default="log/protonet/12_23_2024__17_26_01/4/best_epoch_model.pth", help="预训练模型位置")
    parser.add_argument("--seed", default=42, help="随机数种子")
    parser.add_argument("--in_channel", default=6, help="测井曲线条数")
    parser.add_argument("--train_well_num", default=50, help="进行训练的井数")
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

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=600):
    """
    @param label_true: 真实标签
    @param label_pred: 预测标签
    @param label_name: 标签名（实际的地层标签，不是标签索引）
    @param pdf_save_path: 是否保存
    @dpi: 保存分辨率
    """
    zhfont =mpl.font_manager.FontProperties(fname='../assets/fonts/NotoSansCJK-Regular.ttc')

    cm = confusion_matrix(y_true=label_true, y_pred=label_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.yticks(range(label_name.__len__()), label_name, fontproperties=zhfont, fontsize=10)
    plt.xticks(range(label_name.__len__()), label_name, rotation=90, fontproperties=zhfont, fontsize=10)

    plt.tight_layout()
    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)
    
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

def protonet_label_valid(model, model_label, valid_loader, args):
    model.eval()
    model.training = False
    cur_device = next(model.parameters()).device
    all_label = []
    all_predict = []

    with torch.no_grad():
        for batch_idx, (feature, label) in enumerate(valid_loader):
            feature, label = feature.to(cur_device), label.to(cur_device)
            p = args.support * args.n_cls
            data_support, data_query = feature[:p], feature[p:]
            query_label = torch.arange(args.n_cls).repeat(args.query)
            query_label = query_label.to(cur_device)

            proto = model(data_support)
            proto = proto.reshape(args.support, args.n_cls, -1).mean(dim = 0)

            # 生成one-hot编码
            unique_labels = label[:p].unique()  # 获取当前 batch 中的唯一类别标签
            one_hot_labels = torch.eye(args.n_cls)[unique_labels].to(cur_device)  # 生成 One-Hot 编码
            one_hot_labels = one_hot_labels.unsqueeze(-1)  # 变为 (batch_size, n_cls, 1)

            label_feature, lambda_k = model_label(one_hot_labels) 

            # 计算修正后的原型
            label_feature = label_feature.squeeze(-1)  # 变为 (batch_size, hidden_size)

            # 选择与当前 batch 中的标签相对应的 proto
            selected_proto = proto[unique_labels]  # 选择对应的 proto 行，形状为 (num_unique_labels, hidden_size)

            # 修正后的原型
            proto_new = lambda_k * selected_proto + (1 - lambda_k) * label_feature
                
            logits = euclidean_metric(model(data_query), proto_new)
            _, pred = torch.max(logits, dim = 1) 
            all_label.append(query_label)
            all_predict.append(pred)
    
    return all_label, all_predict

def protonet_valid(model, valid_loader, args):
    model.eval()
    model.training = False
    cur_device = next(model.parameters()).device
    all_label = []
    all_predict = []

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
            _, pred = torch.max(logits, dim = 1) 
            all_label.append(query_label)
            all_predict.append(pred)
    
    all_label = torch.cat(all_label).cpu().numpy() # 转为numpy数组
    all_predict = torch.cat(all_predict).cpu().numpy()
    
    return all_label, all_predict


def main(args):
    set_seeds(seed=args.seed)
    device = torch.device("cuda:" + args.gpu_id)
    model = Net(in_channels=args.in_channel, classes=10)
    model.to(device)
    trainLoader, testLoader = get_dataloader(args.train_dir, args.val_dir, args.slice_length, args.slice_step, args.train_well_num, args.val_well_num, args.frequency_aug, args.train_categorize_id, args.val_categorize_id, args.noise_ration, args.n_batch, args.n_cls, args.support + args.query)
    # 加载预训练参数
    if args.pretrained == True and args.pretrained_filepath is not None and Path(args.pretrained_filepath).exists():
        loaded_model = torch.load(args.pretrained_filepath, map_location=torch.device("cpu"))
        net_dict = model.state_dict()
        # 判断model尺寸是否相同，仅加载相同的model
        pretrained_dict = {k : v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}
        net_dict.update(pretrained_dict)
        model.load_state_dict(net_dict, strict=False)
    
    true_label, label_predict = protonet_valid(model, testLoader, args)
    print(true_label)
    


    label_name = ['K1z2+1', 'J2a', 'J2z', 'J1y', 'J1f', 'chang1', 'chang2', 'chang3', 'chang4+5', 'chang6']

    draw_confusion_matrix(true_label, label_predict, label_name, pdf_save_path="proto.png")

if __name__ == '__main__':

    my_args = parse_args()
    main(my_args)