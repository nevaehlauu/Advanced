# 配置文件
import argparse

def parse_args():
    """
    提供训练相关参数
    """
    parser = argparse.ArgumentParser("Transformer")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--epochs", default=100, type=int, help="训练轮数")
    parser.add_argument("--slice_length", default=96, type=int, help="测井切片长度")
    parser.add_argument("--slice_step", default=64, type=int, help="测井切片步长")
    parser.add_argument("--patch_size", default=8, type=int, help="测井片段切分patch数")
    parser.add_argument("--in_channels", default=5, type=int, help="使用的测井曲线条数")
    parser.add_argument("--d_model", default=512, type=int, help="隐层大小")
    parser.add_argument("--train_data", default="../data/well_data/", type=str, help="训练集路径")
    parser.add_argument("--test_tata", default="../data/well_data/", type=str, help="验证集路径")
    parser.add_argument("--model_save_path", default="./pretrain_result/vgg16_1d.pth", type=str, help="模型保存路径")
    parser.add_argument("--gpu_id", default="0", help="gpu id")
    parser.add_argument("--learning_rate", default=0.001, help="学习率")
    parser.add_argument("--gamma", default=0.98, help="学习率衰减")
    parser.add_argument("--batchsize", default=2048, help="训练batchsize")
    parser.add_argument("--datasets_cfg_filepath", default="./configs/train_data.yaml", help="配置文件存储位置")
    args = parser.parse_args()
    return args
