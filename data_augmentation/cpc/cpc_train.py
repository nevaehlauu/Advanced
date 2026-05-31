import os
import sys
import torch
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
from config.cpc_config import parse_args
from cpc.cpc_model import CPCModel, Encoder_classifier, InfoNCELoss
from cpc.cpc_dataset import SortedNumberDataset, Dataset_Normal
from utils.utils import set_seeds, save_fig

def train(model, train_loader, optimizer, criterion, print_period):
    """
    对比学习：三元组
    """
    model.train()
    total_loss = []
    cur_device = next(model.parameters()).device
    print_period = 10 

    for batch_idx, this_batch in enumerate(train_loader):
        anchor_input, positive_input, negative_input = this_batch[0], this_batch[1], this_batch[2]
        anchor_input, positive_input, negative_input = anchor_input.to(cur_device), positive_input.to(cur_device), negative_input.to(cur_device)
        
        optimizer.zero_grad()
        anchor_encoded, pos_encoded, neg_encoded = model(anchor_input, positive_input,  negative_input) # batchsize * code_size, batchsize * neg_num * code_size

        loss = criterion(anchor_encoded, pos_encoded, neg_encoded)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        if batch_idx % print_period == 0:
            print("batch: {} / {}, loss: {:.4f}".format(batch_idx, len(train_loader), sum(total_loss) / len(total_loss)))
        
    return sum(total_loss) / len(total_loss)
def train_dnsl(model, train_loader, optimizer, print_period):
    """
    困难样本学习
    """
    model.train()
    total_loss = []
    cur_device = next(model.parameters()).device
    print_period = 10 

    for batch_idx, this_batch in enumerate(train_loader):
        anchor_input, positive_input, negative_input = this_batch[0], this_batch[1], this_batch[2]
        anchor_input, positive_input, negative_input = anchor_input.to(cur_device), positive_input.to(cur_device), negative_input.to(cur_device)
        
        optimizer.zero_grad()
        anchor_encoded, pos_encoded, neg_encoded = model(anchor_input, positive_input,  negative_input) # batchsize * code_size, batchsize * neg_num * code_size

        positive_distances = (anchor_encoded - pos_encoded).pow(2).sum(dim=1)
        negative_distances = (anchor_encoded.unsqueeze(1) - neg_encoded).pow(2).sum(dim=2)

        # 如何选取困难样本，还需要商议
        hardest_negativa_distaices, _ = negative_distances.min(1)
        
        triplet_loss = F.softplus(positive_distances - hardest_negativa_distaices)
        loss = triplet_loss.mean()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        if batch_idx % print_period == 0:
            print("batch: {} / {}, loss: {}".format(batch_idx, len(train_loader), sum(total_loss) / len(total_loss)))
        
    return sum(total_loss) / len(total_loss)

def main(args):

    set_seeds(args.seed)

    # # 数据集
    # if args.train_dir.endswith("/train/"):
    #     label_name = {'K1z2+1': 0, 'J2a': 1, 'J2z': 2, 'J1y':3, 'J1f': 4, 'chang1': 5, 'chang2': 6, 'chang3': 7, 'chang4+5': 8,  'chang6': 9}
    # else:
    #     label_name = {'J2a': 0, 'J2z': 1, 'J1y': 2, 'chang2': 3, 'chang3': 4, 'chang4+5': 5, 'chang6': 6, 'chang7': 7, 'J1f': 8, 'chang1': 9, 
    #                   'y2': 10, 'y3': 11, 'y4+5': 12, 'y6': 13, 'y7': 14, 'y8': 15, 'y9': 16, 'y10': 17, 'chang8': 18, 'chang9': 19, 'y4': 20, 
    #                   'y5': 21, 'K1z2+1': 22, 'y1': 23, 'chang10': 24, 'T2z': 25, 'K1z3': 26}

    dataset = SortedNumberDataset(data_path=args.train_dir, batch_size=128, anchor_num=5, negative_num=9, pos_num=1, 
                                  slice_len=args.slice_length, slice_step=args.slice_step, well_num=args.train_well_num, 
                                  classification_name=args.classification_name, args=args, frequency_aug=args.frequency_aug)
    train_dataloader = DataLoader(dataset=dataset, batch_size=args.batchsize, shuffle=True, num_workers=6)

    print("label name: ", dataset.label_name)

    device = torch.device("cuda:" + args.gpu_id)
    genotype = eval("genotypes.%s" % args.arch)
    model = CPCModel(in_channels=args.in_channel, code_size=128, anchor_num=5, is_nas=args.is_nas, layers=args.layers, genotype=genotype)
    # model.to(device)

    loss_func = InfoNCELoss(temperature=args.temperature)
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
    
    model.to(device)

    # 模型日志存储
    date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")  # 加上时间
    date_time = args.classification_name + "_cpc_" + date_time
    args.log_dir_path = str((Path(args.log_dir_path) / date_time).resolve())
    Path(args.log_dir_path).mkdir(parents=True, exist_ok=True) # 确保父目录存在

    print("----------------训练开始----------------------")

    # 在日志中记录准确率等值
    result_dict = {"trainloss": []}
    best_loss = 0x3f3f3f3f
    best_epoch = 0

    for epoch in range(args.epochs):
        
        print("---------第{}轮训练开始-------------".format(epoch + 1))

        train_loss = train(model, train_dataloader, optimizer, loss_func, args.print_period)
        scheduler.step()

        result_dict['trainloss'].append(train_loss)
        save_fig(result_dict['trainloss'], "train loss", str(Path(args.log_dir_path) / "train_loss.png"))
        print("epoch: {}, train_loss: {:.4f}".format(epoch, train_loss))

        # 保存结果最好的模型
        if best_loss > train_loss:
            best_loss = train_loss
            best_epoch = epoch
            print("best loss: {:4f}, best epoch: {}".format(best_loss, epoch))
            print("save model")
            savepath = str(Path(args.log_dir_path) / "best_epoch_model.pth")
            torch.save(model.state_dict(), savepath)
            # torch.save(model, savepath)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= args.patience:
                print("Traning stopped dut to no improvement in loss")
                break

        with open(Path(args.log_dir_path) / 'logging.json', "w") as f:
            json.dump(result_dict, f, indent=2)
    
    print("Training best val acc: " + str(best_loss) + ", best epoch: " + str(best_epoch))
    print('Training complete, models saved in {}'.format(args.log_dir_path))

if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    my_args = parse_args()
    main(my_args)
