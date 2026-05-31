import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import pandas as pd
import pywt
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from config.data_228_config import parse_args
from utils.wavelet_change import wavelet_noising, wavelet_aug

####输入为96*6时，即六条测井曲线同时输入时的处理方法
####将增广后的数据作为单独样本进行处理

### 测井数据预处理（这里只处理训练集数据，测试集数据在test中单独处理）
dir = os.getcwd()
print(">>>当前工作目录：", dir) # 返回当前工作目录

#读取txt文件，按逗号划分，返回数组
#处理步骤；1、定义读井的函数，读取所有井。2、按逗号划分，返回数组

#读取train和test的所有井，其中train的一部分train，一部分valid

# 批量读取同一文件夹下的数据并将其存储在dir中
def _read_dir(path, train_well_num):
    dir = []
    dir_list = os.listdir(path) # 列出了指定路径 self.path 下的所有文件和子目录
    dir_list = dir_list[:train_well_num]
    for i in dir_list:
        dir_path = os.path.join(path, i) # 使用 os.path.join() 函数将父目录路径 self.path 和子目录名称 i 拼接起来,得到一个完整的子目录路径dir_path
        dir.append(dir_path)
    return dir

def well_label(dir, well_num, classification_name):
    """
    获取data_label中出现的所有标签信息，如果包含测试井，这里应该还将测试井的label考虑进去
    """

    label_dict = {}
    label_index = 0

    Well_path = _read_dir(dir, well_num)

    for well in Well_path:
        data_all = pd.read_csv(well, header=None)
        if classification_name == "地质分层":
            data_label = data_all.iloc[:, 8].values
        elif classification_name == "储层划分":
            data_label = data_all.iloc[:, 9].values
        else:
            data_label = data_all.iloc[:, 10].values # 后面还需要去除切片中标签为0的数据（应该去除切片中标签为0，还是直接去除原始数据中标签为0的点还需要讨论）
        
        for label in data_label:
            if label not in label_dict:
                label_dict[label] = label_index
                label_index += 1

    return label_dict

def get_well_label(args):
    """
    获取data_label中出现的所有标签信息，如果包含测试井，这里应该还将测试井的label考虑进去
    """
    train_dir = args.train_dir
    val_dir =args.val_dir
    train_well_num = args.train_well_num
    val_well_num = args.val_well_num
    classification_name = args.classification_name

    label_dict = {}
    label_index = 0

    Well_path = _read_dir(train_dir, train_well_num) + _read_dir(val_dir, val_well_num)

    for well in Well_path:
        data_all = pd.read_csv(well, header=None)
        if classification_name == "地质分层":
            data_label = data_all.iloc[:, 8].values
        elif classification_name == "储层划分":
            data_label = data_all.iloc[:, 9].values
        else:
            data_label = data_all.iloc[:, 10].values # 后面还需要去除切片中标签为0的数据（应该去除切片中标签为0，还是直接去除原始数据中标签为0的点还需要讨论）
        
        for label in data_label:
            if label not in label_dict:
                label_dict[label] = label_index
                label_index += 1

    return label_dict

def wave_low(data_value, wavelet, level):
    """
    小波变换，并采用低频数据进行增广
    """
    aug_value = []
    aug_value.append([data_value[:, 0], data_value[:, 0]])
    # aug_value.append([data_value[:, 0]])
    # 对每条曲线分别进行小波变换
    for i in range(1, data_value.shape[1]):
    # for i in range(data_value.shape[1]):
        single_value = data_value[:, i]
        coeffs = pywt.wavedec(single_value, wavelet, level=level)
        # print("------------", len(single_value), len(data_label), len(coeffs[0]))
        # 注意，小波变换之后应该取低频分量，也就是coeffs中第一个分量，其他分量均是高频分量
        encoder_single_value = []
        encoder_single_value.append(single_value)
        interpolate_coeffs = wavelet_aug(coeffs, single_value)
        # coeffs[0] = interpolate_coeffs[0]
        # coeffs_recon = [interpolate_coeffs[0]] + [0] * len(coeffs[1:])
        # single_value_coeffs = pywt.waverec(coeffs, wavelet)

        # 低频分量增广
        # encoder_single_value.append(single_value_coeffs)
        encoder_single_value.append(interpolate_coeffs[0]) # 2 * len

        # 对增广后的数值数据进行标准化操作
        # todo: 数据归一化操作应该以什么为标准做，这里是对每一条曲线和他增广后的曲线一起做，但是如果将所有曲线在通道维度拼接，是对所有曲线进行归一化操作码
        # encoder_single_value = scale(encoder_single_value, axis=1, with_mean=True, with_std=True, copy=True)
        
        aug_value.append(encoder_single_value) # M * 2 * len
    aug_value = np.array(aug_value) # M * 2 * len -- > M * len * 2
    aug_value = np.swapaxes(aug_value, 1, 2)
    return aug_value

def denosied(data_value, wavelet, level):
    """
    小波去噪
    """
    aug_value = []
    aug_value.append(data_value[:, 0])
    # print(np.array(aug_value).shape)
    for i in range(1, data_value.shape[1]):
        single_value = data_value[:, i]
        encoder_single_value = wavelet_noising(single_value, wavelet, level)
        # print(len(encoder_single_value))
        aug_value.append(encoder_single_value) # M * len
        # print(np.array(aug_value).shape)
    aug_value = np.array(aug_value)
    # print(aug_value.shape) # M * len
    return aug_value

def denosied_aug(data_value, wavelet, level):
    """
    将小波去噪后的数据作为增广数据
    """
    aug_value = []
    aug_value.append([data_value[:, 0], data_value[:, 0]])
    for i in range(1, data_value.shape[1]):
        single_value = data_value[:, i]
        encoder_single_value = []
        encoder_single_value.append(single_value)
        denoised_single_value = wavelet_noising(single_value, wavelet, level)
        encoder_single_value.append(denoised_single_value)
        aug_value.append(encoder_single_value)
    aug_value = np.array(aug_value)
    aug_value = np.swapaxes(aug_value, 1, 2)
    return aug_value

def read_data_from_csv(dir, slice_length, slice_step, train_well_num, frequency_aug, depth_aug, classification_name, args):
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
    Well_path = _read_dir(dir, train_well_num)
    # print("well", Well_path)
    print("-------------使用的井数----------------: ", len(Well_path))
    sliced_data = []
    sliced_label = []

    label_name = get_well_label(args)
    print("data_label_name: ", label_name)

    # 读取每一口井的数据，对其进行操作
    for well in Well_path:
        data_all = pd.read_csv(well, header=None)
        data_value = data_all.iloc[:, 2:8].values # .values将数据转换为numpy数组，第 0 维通常表示行。所以 [:, 2:8] 中的 : 表示选择了所有行,得到的数据形式Numpy数组,且shape为(n, 6)
        # 首先对所有数据进行归一化操作
        data_value = scale(data_value, axis=0, with_mean=True, with_std=True, copy=True)
        if classification_name == "地质分层":
            data_label = data_all.iloc[:, 8].values
        elif classification_name == "储层划分":
            data_label = data_all.iloc[:, 9].values
        else:
            data_label = data_all.iloc[:, 10].values # 后面还需要去除切片中标签为0的数据（应该去除切片中标签为0，还是直接去除原始数据中标签为0的点还需要讨论）
                
        # 对每条曲线进行数据增广，这里首先进行频域增广，看看效果
        aug_value = []
        wavelet = 'db10'
        level = 2

        # 对除深度曲线外的每条曲线进行傅里叶变换
        if frequency_aug == "wave_0":
            aug_value.append([data_value[:, 0], data_value[:, 0]])
            for i in range(1, data_value.shape[1]):
                single_value = data_value[:, i]

        
        # 将小波变化增广后的曲线作为新的样本，这里对低频分量进行插值，增广，这里进行增广时不对深度曲线进行增广
        elif frequency_aug == "wave_low":
            aug_value = wave_low(data_value, wavelet, level)
            data_value = aug_value
        
        # 小波变换,这里选取高频分量插值回原来大小后作为新的样本(效果差)
        elif frequency_aug == "wave_high":
            # 对每条曲线分别进行小波变换
            aug_value.append([data_value[:, 0], data_value[:, 0], data_value[:, 0]])
            for i in range(1, data_value.shape[1]):
                single_value = data_value[:, i]
                coeffs = pywt.wavedec(single_value, wavelet, level=level)
                # 插值后的低频及高频分量
                interpolate_coeffs = wavelet_aug(coeffs, single_value)
                encoder_single_value = []
                encoder_single_value.append(single_value)
                # 高频分量增广
                for i in range(1, len(interpolate_coeffs)):
                    encoder_single_value.append(interpolate_coeffs[i]) # level+1 * len
                
                aug_value.append(encoder_single_value) # M * (level+1) * len
            aug_value = np.array(aug_value) # M * (level+1) * len -- > M * len * (level+1)
            aug_value = np.swapaxes(aug_value, 1, 2)

            data_value = aug_value

        # 小波低频+高频增广
        elif frequency_aug == "wave_low_high":
            # 对每条曲线分别进行小波变换
            aug_value.append([data_value[:, 0], data_value[:, 0], data_value[:, 0], data_value[:, 0]])
            for i in range(1, data_value.shape[1]):
                single_value = data_value[:, i]
                coeffs = pywt.wavedec(single_value, wavelet, level=level)
                # 插值后的低频及高频分量
                interpolate_coeffs = wavelet_aug(coeffs, single_value)
                encoder_single_value = []
                encoder_single_value.append(single_value)
                # 高频分量增广
                for i in range(len(interpolate_coeffs)):
                    encoder_single_value.append(interpolate_coeffs[i]) # level+1 * len
                
                aug_value.append(encoder_single_value) # M * (level+1) * len
            aug_value = np.array(aug_value) # M * (level+1) * len -- > M * len * (level+1)
            aug_value = np.swapaxes(aug_value, 1, 2)

            data_value = aug_value

        # 小波变换+小波去噪
        elif frequency_aug == "wave_denoised":
            aug_value = denosied(data_value, wavelet, level)
            aug_value = np.swapaxes(aug_value, 0, 1) # len * M
            aug_value = wave_low(aug_value, wavelet, level)
            data_value = aug_value

        # 小波去噪(直接去噪)，也就是将去噪后的样本取代原始样本
        elif frequency_aug == "denosied":
            aug_value = denosied(data_value, wavelet, level)
            aug_value = aug_value[:, :, np.newaxis]
            # print(aug_value.shape)
            data_value = aug_value
        
        # 小波去噪，将小波去噪后的样本作为增广数据
        elif frequency_aug == "denosied_aug":
            aug_value = denosied_aug(data_value, wavelet, level)
            data_value = aug_value

        # 不进行任何处理
        else:
            # 直接将测井数据在通道拼接,并给其增加一个维度
            data_value = np.swapaxes(data_value, 0, 1)
            data_value = data_value[:, :, np.newaxis]
            # print("----data_value.shape-----", data_value.shape)

        # 将测井数据划分为切片，并取每段切片的中间点标签作为其标签
        # 遍历测井数据的第三个维度，将前两个维度划分为切片
        for j in range(data_value.shape[2]):
            slice_num = (data_label.shape[0] - slice_length) // slice_step + 1 # 切片个数
            for i in range(slice_num):
                start = i * slice_step
                end = start + slice_length
                # 取中间点标签作为一段切片标签
                mid_index = (start + end) // 2
                if classification_name == "油气水划分" and data_label[mid_index] == '0':
                    continue

                a_slice = data_value[:, start:end, j]

                # 添加高斯噪声
                noise_ration = 0.2
                augment_function = 0.5
                if noise_ration > 0 and augment_function > 0:
                    if torch.rand(1).item() < augment_function:
                        for i in range(a_slice.shape[0]):
                            data = a_slice[i, :]
                            noise_std = np.std(data) * noise_ration
                            a_slice[i, :] = data + np.random.normal(0, noise_std, size=data.shape)

                sliced_data.append(a_slice)
                # labels = np.array(label_name[data_label[index]] for index in range(start, end))
                labels = label_name[data_label[mid_index]]
                sliced_label.append(labels)
        
    # 对数据进行变换并return
    sliced_data = torch.from_numpy(np.array(sliced_data)).float()
    sliced_label = torch.from_numpy(np.array(sliced_label)).float()
    return sliced_data, sliced_label, label_name

def dataloader_split(dir, slice_length, slice_step, train_well_num, frequency_aug, depth_aug, classification_name, batchsize, args):
    sliced_data, sliced_label, label_name = read_data_from_csv(dir, slice_length, slice_step, train_well_num, frequency_aug, depth_aug, classification_name, args)
    print("sliced_data.shape: ", sliced_data.shape)
    print("sliced_label.shape: ", sliced_label.shape)

    trainX, validX, trainY, validY = train_test_split(sliced_data, sliced_label, test_size=0.1, random_state=42)

    train_set = TensorDataset(trainX, trainY)
    valid_set = TensorDataset(validX, validY)

    trainLoader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True, num_workers=0)
    validLoader = DataLoader(dataset=valid_set, batch_size=batchsize, shuffle=False, num_workers=0)

    return trainLoader, validLoader, label_name

def dataloader(dir_train, dir_val, slice_length, slice_step, train_well_num, val_well_num, frequency_aug, depth_aug, classification_name, batchsize, args):
    train_data, train_label, train_label_name = read_data_from_csv(dir_train, slice_length, slice_step, train_well_num, frequency_aug, depth_aug, classification_name, args)
    valid_data, valid_label, val_label_name = read_data_from_csv(dir_val, slice_length, slice_step, val_well_num, None, depth_aug, classification_name, args)
    print("train_data.shape: ", train_data.shape)
    print("train_label.shape: ", train_label.shape)
    print("val_data.shape: ", valid_data.shape)
    print("val_label.shape: ", valid_label.shape)

    train_set = TensorDataset(train_data, train_label)
    valid_set = TensorDataset(valid_data, valid_label)

    trainLoader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True, num_workers=0)
    validLoader = DataLoader(dataset=valid_set, batch_size=batchsize, shuffle=False, num_workers=0)

    return trainLoader, validLoader, train_label_name

if __name__ == '__main__':
    args = parse_args()
#     trainLoader, validLoader, label_name = dataloader_split(args.dir, args.slice_length, args.slice_length, args.train_well_num, args.frequency_aug, args.depth_aug, args.classification_name, args.batchsize)
    # label_dict = get_well_label(args)
    # print(label_dict)