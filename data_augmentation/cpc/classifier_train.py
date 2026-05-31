import os
import sys
import torch
import time
import json
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from pathlib import Path


curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(str(Path("./").resolve()))

import darts.genotypes as genotypes
from config.cpc_config import classifier_parse_args
from cpc.cpc_model import CPCModel, Encoder_classifier, InfoNCELoss
from cpc.cpc_dataset import SortedNumberDataset, Dataset_Normal
from utils.utils import set_seeds, save_train_val_fig

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
        output = model(feature)
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
            output = model(feature)
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

    # if args.train_dir.endswith("/train/"):
    #     label_name = {'K1z2+1': 0, 'J2a': 1, 'J2z': 2, 'J1y':3, 'J1f': 4, 'chang1': 5, 'chang2': 6, 'chang3': 7, 'chang4+5': 8,  'chang6': 9}
    # else:
    #     label_name = {'J2a': 0, 'J2z': 1, 'J1y': 2, 'chang2': 3, 'chang3': 4, 'chang4+5': 5, 'chang6': 6, 'chang7': 7, 'J1f': 8, 'chang1': 9, 
    #                   'y2': 10, 'y3': 11, 'y4+5': 12, 'y6': 13, 'y7': 14, 'y8': 15, 'y9': 16, 'y10': 17, 'chang8': 18, 'chang9': 19, 'y4': 20, 
    #                   'y5': 21, 'K1z2+1': 22, 'y1': 23, 'chang10': 24, 'T2z': 25, 'K1z3': 26}


    # 数据集
    train_dataset = Dataset_Normal(args.train_dir, args.slice_length, args.slice_step, well_num=args.train_well_num, 
                                   classification_name=args.classification_name, args=args, frequency_aug=args.frequency_aug)
    valid_dataset = Dataset_Normal(args.val_dir, args.slice_length, args.slice_step, well_num=args.val_well_num, 
                                   classification_name=args.classification_name, args=args, frequency_aug=args.frequency_aug)
    trainLoader = DataLoader(dataset=train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=6)
    validLoader = DataLoader(dataset=valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=6)

    label_name = train_dataset.name

    device = torch.device("cuda:" + args.gpu_id)
    genotype = eval("genotypes.%s" % args.arch)
    model = Encoder_classifier(in_channels=args.in_channel, code_size=128, classes=len(label_name), is_nas=args.is_nas, layers=args.layers, genotype=genotype)
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

        # 加分类头后，保证encoder部分参数不发生变化
        # for param in model.parameters():
        #     param.requires_grad = False

        print("预训练模型加载完毕")

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
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

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
            # torch.save(model, savepath)

        with open(Path(args.log_dir_path) / 'logging.json', "w") as f:
            json.dump(result_dict, f, indent=2)
    
    print("Training best val acc: " + str(best_acc) + ", best epoch: " + str(best_epoch))
    print('Training complete, models saved in {}'.format(args.log_dir_path))

if __name__ == '__main__':

    my_args = classifier_parse_args()
    main(my_args)
