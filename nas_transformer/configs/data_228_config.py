import argparse

def parse_args():
    parser = argparse.ArgumentParser("few_shot_learing")
    parser.add_argument("--epochs", default=100, help="训练轮次")
    parser.add_argument("--in_channel", default=6, help="测井曲线条数")
    parser.add_argument("--classes", default=10, help="分类数")
    parser.add_argument("--learning_rate", default=0.0005, help="学习率")
    parser.add_argument("--decay", default=0.98, help="学习率衰减")
    parser.add_argument("--gpu_id", default="0",  help="gpu的id")
    parser.add_argument("--seed", default=42, help="随机数种子")
    parser.add_argument("--log_dir_path", default="trainsformer_clip/results/senet", help="日志文件存储位置")
    parser.add_argument("--print_period", default=10, help="打印间隔")
    parser.add_argument("--pretrained", default=False, help="是否加载预训练模型")
    parser.add_argument("--pretrained_filepath", default="", help="预训练模型位置")
    parser.add_argument("--dir", default="../data/well_data/", help="数据集位置")
    parser.add_argument("--train_dir", default="../data/well_228_old/train/", help="训练集位置")
    parser.add_argument("--val_dir", default="../data/well_228_old/test/", help="验证集位置")
    parser.add_argument("--slice_length", default=96, help="切片长")
    parser.add_argument("--slice_step", default=96, help="滑动步长")
    parser.add_argument("--train_well_num", default=92, help="进行训练的井数")
    parser.add_argument("--val_well_num", default=13, help="进行训练的井数")
    parser.add_argument("--frequency_aug", default="None", help="是否进行频域增广，以及进行什么频域增广，wave_1, wave_2, False")
    parser.add_argument("--depth_aug", default="False", help="是否进行时域增广，以及进行什么增广")
    parser.add_argument("--classification_name", default="地质分层", help="测井解释任务类型，地质分层、油气水划分、储层划分")
    parser.add_argument("--batchsize", default=1024)

    args = parser.parse_args()
    return args