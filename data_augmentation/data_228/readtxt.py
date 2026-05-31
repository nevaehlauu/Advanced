import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import pandas as pd
import pywt
import numpy as np
from scipy import interpolate

####输入为96*6时，即六条测井曲线同时输入时的处理方法

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

def get_well_label(Well_path, classification_name):
    """
    获取data_label中出现的所有标签信息，如果包含测试井，这里应该还将测试井的label考虑进去
    """
    label_dict = {}
    label_index = 0

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

def read_data_from_csv(dir, slice_length, slice_step, train_well_num, frequency_aug, depth_aug, classification_name):
    """
    dir: 井的位置
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
    print("well", Well_path)
    print("-------------使用的井数----------------", len(Well_path))
    sliced_data = []
    sliced_label = []

    label_name = get_well_label(Well_path, classification_name)
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
        # 分别对每条曲线进行小波变换,并且将增广后的曲线和原始曲线在宽度维度拼接,而不同曲线间在通道维度拼接
        if frequency_aug == "wave_1":
            # 对每条曲线分别进行小波变换
            for i in range(data_value.shape[1]):
                single_value = data_value[:, i]
                coeffs = pywt.wavedec(single_value, wavelet, level=level)
                print("------------", len(single_value), len(data_label), len(coeffs[0]))
                # 注意，小波变换之后应该取低频分量，也就是coeffs中第一个分量，其他分量均是高频分量
                encoder_single_value = []
                encoder_single_value.append(single_value)
                if level != 1:
                    # 需要先进行插值，再进行拼接（在哪个维度拼接是一个问题）—— 感觉应该是将不同增广数据放在宽度上，而不同曲线放在channel维度上，那样得到的曲线为channel*height*weight(M*len*2)
                    ori_index = np.linspace(0, len(single_value)-1, len(single_value)) # 原始信号坐标，三个参数表示从[0, len(s)-1]需要有len(l)个点
                    coeffs_index  = np.linspace(0, len(single_value)-1, len(coeffs[0])) # 需要进行插值的坐标
                    f_interpolate = interpolate.interp1d(coeffs_index, coeffs[0], kind='linear')
                    padded_coedd = f_interpolate(ori_index)
                    encoder_single_value.append(padded_coedd)
                else:
                    encoder_single_value.append(coeffs[0]) # 这里encoder_single_value维度为2 * len
                # 拼接，这里在通道维度进行拼接
                # encoder_single_value = torch.cat(tuple(encoder_single_value), dim=1) # 这里得到的数据形式为len*2，2为channel
                # 对增广后的数值数据进行标准化操作
                # todo: 数据归一化操作应该以什么为标准做，这里是对每一条曲线和他增广后的曲线一起做，但是如果将所有曲线在通道维度拼接，是对所有曲线进行归一化操作码
                # encoder_single_value = scale(encoder_single_value, axis=1, with_mean=True, with_std=True, copy=True)
                aug_value.append(encoder_single_value) # M * 2 * len
            # aug_value = torch.cat(tuple(aug_value), dim=0) # M * len * 2，M为曲线条数
            aug_value = np.array(aug_value) # M * 2 * len -- > M * len * 2
            # print(aug_value.shape)
            aug_value = np.swapaxes(aug_value, 1, 2)
            # print(aug_value.shape)

            # # 对通道独立归一化，也就是对每个通道各自归一化，即对每条曲线和其增广后的值进行归一化处理
            # aug_value = scale(aug_value, axis=2, with_mean=True, with_std=True, copy=True)
            data_value = aug_value
        
        # 分别对每条曲线进行小波变换,并且将增广后的曲线和原始曲线全都在通道维度拼接
        elif frequency_aug == "wave_2":
            # 对每条曲线分别进行小波变换
            for i in range(data_value.shape[1]):
                single_value = data_value[:, i]
                coeffs = pywt.wavedec(single_value, wavelet, level=level)
                # 注意，小波变换之后应该取低频分量，也就是coeffs中第一个分量，其他分量均是高频分量
                aug_value.append(single_value)
                if level != 1:
                    # 需要先进行插值，再进行拼接（在哪个维度拼接是一个问题）—— 感觉应该是将不同增广数据放在宽度上，而不同曲线放在channel维度上，那样得到的曲线为channel*height*weight(M*len*2)
                    ori_index = np.linspace(0, len(single_value)-1, len(single_value)) # 原始信号坐标，三个参数表示从[0, len(s)-1]需要有len(l)个点
                    coeffs_index  = np.linspace(0, len(single_value)-1, len(coeffs[0])) # 需要进行插值的坐标
                    f_interpolate = interpolate.interp1d(coeffs_index, coeffs[0], kind='linear')
                    padded_coedd = f_interpolate(ori_index)
                    aug_value.append(padded_coedd)
                else:
                    aug_value.append(coeffs[0]) # 这里encoder_single_value维度为2 * len
            aug_value = np.array(aug_value) # M * 2 * len -- > M * len * 2
            aug_value = aug_value[:, :, np.newaxis]
            print(aug_value.shape)
            data_value = aug_value
        
        else:
            # 直接将测井数据在通道拼接,并给其增加一个维度
            data_value = np.swapaxes(data_value, 0, 1)
            data_value = data_value[:, :, np.newaxis]
            # print("----data_value.shape-----", data_value.shape)

        # 将测井数据划分为切片，并取每段切片的中间点标签作为其标签
        slice_num = (data_label.shape[0] - slice_length) // slice_step + 1 # 切片个数
        for i in range(slice_num):
            start = i * slice_step
            end = start + slice_length
            sliced_data.append(data_value[:, start:end, :])
            # 取中间点标签作为一段切片标签
            mid_index = (start + end) // 2
            if classification_name == "油气水划分" and data_label[mid_index] == '0':
                continue
            # labels = np.array(label_name[data_label[index]] for index in range(start, end))
            labels = label_name[data_label[mid_index]]
            sliced_label.append(labels)
        
    # 对数据进行变换并return
    sliced_data = torch.from_numpy(np.array(sliced_data)).float()
    sliced_label = torch.from_numpy(np.array(sliced_label)).float()
    return sliced_data, sliced_label, label_name

def dataloader(dir, slice_length, slice_step, train_well_num, frequency_aug, depth_aug, classification_name):
    sliced_data, sliced_label, label_name = read_data_from_csv(dir, slice_length, slice_step, train_well_num, frequency_aug, depth_aug, classification_name)
    print("sliced_data.shape: ", sliced_data.shape)
    print("sliced_label.shape: ", sliced_label.shape)

    trainX, validX, trainY, validY = train_test_split(sliced_data, sliced_label, test_size=0.1, random_state=42)

    train_set = TensorDataset(trainX, trainY)
    valid_set = TensorDataset(validX, validY)

    trainLoader = DataLoader(dataset=train_set, batch_size=1024, shuffle=True, num_workers=0)
    validLoader = DataLoader(dataset=valid_set, batch_size=1024, shuffle=False, num_workers=0)

    return trainLoader, validLoader, label_name

from config.data_228_config import parse_args
if __name__ == '__main__':
    args = parse_args()
    trainLoader, validLoader, label_name = dataloader(args.dir, args.slice_length, args.slice_length, args.train_well_num, args.frequency_aug, args.depth_aug, args.classification_name)