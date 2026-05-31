import argparse

def search_parse_args():
    parser = argparse.ArgumentParser("few_shot_learing_search")
    parser.add_argument("--batchsize", default=512)
    parser.add_argument("--epochs", default=50, help="训练轮次")
    parser.add_argument("--learning_rate", default=0.025, help="学习率")
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='L2 正则化系数,用于防止模型过拟合')
    parser.add_argument("--report_freq", default=10, help="打印间隔")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=32, help="滑动步长")
    parser.add_argument("--depth_aug", default="False", help="是否进行时域增广，以及进行什么增广")
    parser.add_argument("--classification_name", default="地质分层", help="测井解释任务类型，地质分层、油气水划分、储层划分")
    parser.add_argument('--layers', type=int, default=4, help='搜索时模型堆叠层数')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping') #梯度裁剪阈值，用于控制梯度的最大范围，以避免梯度爆炸的问题。
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss') #是否使用one-short策略展开验证损失
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding') #学习率
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding') #权重损失

    # 下面是可能需要改动的配置
    parser.add_argument("--seed", default=10, help="随机数种子")
    parser.add_argument("--init_channels", default=6, help="测井曲线条数")
    parser.add_argument("--train_well_num", default=10, help="进行训练的井数")
    parser.add_argument("--val_well_num", default=2, help="进行训练的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--train_categorize_id", default=4, help="训练集使用的区块")
    parser.add_argument("--val_categorize_id", default=4, help="测试集区块名")
    # parser.add_argument("--noise_ration", default=0.05, help="高斯噪声幅度，如果进行高斯增广，为0")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")
    parser.add_argument("--gpu", default="0",  help="gpu的id")
    # parser.add_argument("--gpu", default="3",  help="gpu的id")


    args = parser.parse_args()
    return args

def train_parse_args():
    parser = argparse.ArgumentParser("few_shot_learing_train")
    parser.add_argument("--batchsize", default=1024)
    parser.add_argument("--epochs", default=100, help="训练轮次")
    parser.add_argument("--learning_rate", default=0.025, help="学习率")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='L2 正则化系数,用于防止模型过拟合')
    parser.add_argument("--gpu", default="3",  help="gpu的id")
    parser.add_argument("--report_freq", default=10, help="打印间隔")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=32, help="滑动步长")
    parser.add_argument("--depth_aug", default="False", help="是否进行时域增广，以及进行什么增广")
    parser.add_argument("--classification_name", default="地质分层", help="测井解释任务类型，地质分层、油气水划分、储层划分")
    parser.add_argument('--layers', type=int, default=4, help='搜索时模型堆叠层数')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model') #模型保存路径
    parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--arch', type=str, default='DARTS_2', help='which architecture to use') #要使用的网络架构，默认为'DARTS'。指定了使用的神经网络架构。
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping') #梯度裁剪阈值，用于控制梯度的最大范围，以避免梯度爆炸的问题。

    # 下面是可能需要改动的配置
    parser.add_argument("--seed", default=10, help="随机数种子")
    parser.add_argument("--init_channels", default=6, help="测井曲线条数")
    parser.add_argument("--train_well_num", default=10, help="进行训练的井数")
    parser.add_argument("--val_well_num", default=2, help="进行训练的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--train_categorize_id", default=4, help="训练集使用的区块")
    parser.add_argument("--val_categorize_id", default=4, help="测试集区块名")
    # parser.add_argument("--noise_ration", default=0.05, help="高斯噪声幅度，如果进行高斯增广，为0")
    parser.add_argument("--noise_ration", default=0.0, help="高斯噪声幅度，如果进行高斯增广，为0")

    args = parser.parse_args()
    return args