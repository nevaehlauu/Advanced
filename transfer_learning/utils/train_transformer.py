"""
功能多了，代码就会复杂起来，没办法的
慢慢就会失控了。。。
"""
import os
import sys
import time

curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
import json
import torch
import yaml
from torch import optim
from pathlib import Path
from utils.evaluate import common_forward, evaluate
from utils.utils import set_seeds, save_fig
from utils.analysis import plt_confusion_matrix
from utils.dataloader import setup_dataloaders


"""
训练代码
在根目录下运行该脚本
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path("./").resolve()))

from configs.get_config import get_train_cfg, cfg2dict
from utils.utils import print_block, set_seeds, load_model_params, printcolor, save_fig, get_classes_map
from darts_common.transformer_token import make_model

def parse_args():
    """
    获取参数
    :return: args
    """
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('--datasets_cfg_file', type=str, default="configs/train_data.yaml", help='存放预训练模型等文件的路径')
    parser.add_argument('--model_config_file', type=str, default="configs/training_data_config.yaml", help='存放数据以及模型配置文件等的路径')
    args = parser.parse_args()

    # assert args.file.endswith('.yaml'), 'You need to provide a .yaml file'
    return args


def run_epoch(model, train_loader, optimizer: torch.optim.Optimizer, criterion, features_name: list):
    """
    进行一轮训练
    :param model:
    :param train_loader:
    :param optimizer:
    :param criterion:
    :return: 准确率和损失
    """
    model.train()  # 切换成训练状态
    total_loss = []  # 记录每一个batch的损失函数
    label_nbr = 0 # 标签总数
    eq_nbr = 0 # 预测正确标签数
    cur_device = next(model.parameters()).device  # 获取当前设备
    print_period = 10  # 打印间隔
    start = time.time()

    for batch_idx, batch in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        loss, predicted, label = common_forward(model, batch, cur_device, criterion, features_name)
        label_nbr += len(label)
        loss.backward()
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

    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss)

def eval_val(model, val_loader, criterion, features_name: list):
    val_acc, val_loss, all_label, all_predicted = evaluate(model, val_loader, criterion, features_name)
    return val_acc, val_loss, all_label, all_predicted

def main(datasets_cfg_flie, model_config_file):
    """
    :param datasets_cfg_flie: 存放预训练模型等文件的路径
    :param model_config_file: 存放数据以及模型配置文件等的路径
    """
    # 获取配置文件
    cfg = get_train_cfg(str(Path(model_config_file).resolve()))
    print_block(cfg, "训练配置参数")

    # 初始化随机数种子
    if cfg.arch.seed is not None:
        set_seeds(cfg.arch.seed)

    # 获取标签信息
    train_classes_map = get_classes_map(cfg.datasets.train.desc_filepath)
    val_classes_map = get_classes_map(cfg.datasets.val.desc_filepath)

    label_name = cfg.model.params.generic.label_name
    train_classes = list(train_classes_map[label_name].keys())
    val_classes = list(val_classes_map[label_name].keys())

    # 使用的测井曲线
    features_name=cfg.model.params.generic.features_name
    # 切片长
    slice_length=cfg.model.params.generic.slice_length

    # 获取训练集
    train_dataset, train_loader = setup_dataloaders(
        # 数据集参数
        **dict(cfg.datasets.train),
        features_name=features_name,
        slice_length=slice_length,
        label_name=label_name,
        classes=train_classes,
        is_train_dataset=True,
        shuffle=True)

    val_dataset, val_loader = setup_dataloaders(
        # 数据集参数
        **dict(cfg.datasets.val),
        features_name=features_name,
        slice_length=slice_length,
        label_name=label_name,
        classes=val_classes,
        is_train_dataset=False,
        shuffle=False)

    # 网络 + 优化器 + 学习率管理 + 损失函数
    # 参数的具体含义参照base_config.py
    model = make_model(datasets_cfg_flie, tgt_vovab=len(train_classes))
   
    model = model.to(cfg.arch.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.optimizer.learning_rate, weight_decay=cfg.model.optimizer.weight_decay)  # 优化器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.model.scheduler.decay)
    loss_func = torch.nn.CrossEntropyLoss()

    # 加载预训练参数
    if cfg.model.pretrained_filepath is not None and Path(cfg.model.pretrained_filepath).exists():
        loaded_model = torch.load(cfg.model.pretrained_filepath, map_location=torch.device("cpu"))['state_dict']
        model = load_model_params(model, loaded_model)

    # 指定日志保存路径
    if cfg.model.log_dir_name is None or cfg.model.log_dir_name == "":
        date_time = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  # 加上时间
        date_time = cfg.model.author + '_' + date_time  # 模型名字加时间吧
        cfg.model.log_dir_path = str((Path(cfg.model.log_dir_path) / date_time).resolve())  # 日志文件加个时间戳吧
    else:
        cfg.model.log_dir_path = str((Path(cfg.model.log_dir_path) / cfg.model.log_dir_name).resolve())  # 日志文件加个时间戳吧
    Path(cfg.model.log_dir_path).mkdir(parents=True, exist_ok=True)  # 确保父目录存在

    printcolor('---------------- 训练开始 ----------------')

    result_dict = {"trainloss": [], "valloss": [], "trainacc": [], "valacc": []}  # 用于保存记录到json里面的字典
    min_loss = 0x3f3f3f3f
    best_acc = -0x3f3f3f3f
    best_epoch = 0
    
    for epoch in range(cfg.arch.epochs):
        # 记录轮次信息
        tmp_strs = []
        for param_group in optimizer.param_groups:
            tmp_strs.append('Changing learning rate to {:8.10f}'.format(param_group['lr']))
        print_block("\n".join(tmp_strs), title='第' + str(epoch) + '轮')
        # 训练和评估
        train_acc, train_loss = run_epoch(model, train_loader, optimizer, loss_func, features_name)
        val_acc, val_loss, all_label, all_predicted = eval_val(model, val_loader, loss_func, features_name)
        # 更新学习率
        for param_group in optimizer.param_groups:
            if param_group['lr'] > cfg.model.optimizer.min_lr:
                scheduler.step()  # 学习率衰减，加着玩玩呗
            else:
                param_group['lr'] = cfg.model.optimizer.min_lr

        # 保存训练信息到result中
        result_dict['trainloss'].append(train_loss)
        result_dict['valloss'].append(val_loss)
        result_dict['trainacc'].append(train_acc)
        result_dict['valacc'].append(val_acc)
        print_block("训练集loss: {:.4f}, 测试集loss: {:.4f}, 训练集acc: {:.4f}, 测试集acc: {:.4f}".
                    format(train_loss, val_loss, train_acc, val_acc), title="本次训练的结果")
        
        # 保存验证集表现最好的模型
        if best_acc < val_acc:
            best_acc = val_acc
            best_epoch = epoch
            print('best acc: {:.4f}%, best epoch: {}'.format(100 * best_acc, best_epoch))
            print('save model')
            save_checkpoint(cfg, model, "best_model")  # 转到cpu，再保存参数
            if cfg.model.params.generic.draw_plt == "True":
                plt_confusion_matrix(all_predicted,
                                     all_label,
                                     train_classes,
                                     save_path=str(Path(cfg.model.log_dir_path) / "confusion_matrix.png"))

        # 保存json和图片
        with open(Path(cfg.model.log_dir_path) / 'logging.json', "w") as f:
            json.dump(result_dict, f, indent=2)

        save_fig(result_dict['trainloss'], "train loss", str(Path(cfg.model.log_dir_path) / "train_loss.png"))
        save_fig(result_dict['valloss'], "val loss", str(Path(cfg.model.log_dir_path) / "val_loss.png"))
        save_fig(result_dict['trainacc'], "train acc", str(Path(cfg.model.log_dir_path) / "train_acc.png"))
        save_fig(result_dict['valacc'], "val acc", str(Path(cfg.model.log_dir_path) / "val_acc.png"))

    printcolor('Training complete, models saved in {}'.format(cfg.model.log_dir_path), "green")

def save_checkpoint(cfg, model, name):
    current_model_path = str(Path(cfg.model.log_dir_path) / (name + "_model.ckpt"))
    torch.save(
        {
            'model': model,  # 整个模型
            'state_dict': model.state_dict(),  # 参数
            'cfg': cfg
        }, current_model_path)

    # 转成字典
    cfg = cfg2dict(cfg)

    # 保存配置信息为yaml文件，方便直接查看
    with open(str(Path(cfg["model"]["log_dir_path"]) / (name + "_model.yaml")), "w", encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)


if __name__ == '__main__':
    """
    主要代码逻辑都写到main里
    注意, if __name__ == '__main__'仍然处于全局域里, 在这里面直接写逻辑会造成全局变量混乱
    """
    my_args = parse_args()
    main(my_args.datasets_cfg_file, my_args.model_config_file)
