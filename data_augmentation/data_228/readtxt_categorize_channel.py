import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 
import os
import sys
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import pandas as pd
import pywt
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils.wavelet_change import wavelet_noising, wavelet_aug

####输入为96*6时，即六条测井曲线同时输入时的处理方法
####将增广后的数据作为单独样本进行处理

### 测井数据预处理（这里只处理训练集数据，测试集数据在test中单独处理）
dir = os.getcwd()
print(">>>当前工作目录：", dir) # 返回当前工作目录

#读取txt文件，按逗号划分，返回数组
#处理步骤；1、定义读井的函数，读取所有井。2、按逗号划分，返回数组

#读取train和test的所有井，其中train的一部分train，一部分valid

class InvalidNumException(Exception):
    """
    当挑选的区块的井大于实际井数时，直接抛出异常
    """
    pass

def _read_dir(path, well_num, categorize_id, is_train):
    """
    根据txt文件中包含的区块信息，随机从某个指定区块中挑选指定数量的井
    """
    if is_train:
        data_name = {0: ['W101', 'W113', 'W115', 'W118', 'W119', 'W125', 'W137', 'W138', 'W145', 'W147', 'W149', 'W150', 'W152', 'W160', 'W161', 'W162', 'W170', 'W172', 'W173', 'W174', 'W176', 'W177', 'W35'], 
                     1: ['W1424', 'W1584', 'W1585', 'W1586', 'W1587', 'W1588', 'W1589', 'W1590', 'W1591', 'W1594', 'W1595', 'W1596', 'W1597', 'W1610', 'W1612', 'W1613', 'W1614', 'W1629', 'W1630', 'W1633', 
                         'W1639', 'W1640', 'W1642', 'W1645', 'W1646', 'W1648', 'W1652', 'W1654', 'W1663', 'W1667', 'W1668', 'W1670', 'W1672', 'W1676', 'W1679', 'W1685', 'W1690', 'W1691', 'W1693', 'W1694', 
                         'W1695', 'W1696', 'W1697', 'W1698', 'W1702', 'W1704', 'W1709', 'W651', 'W652', 'W654', 'W655', 'W656', 'W657', 'W658', 'W660', 'W661', 'W662', 'W664', 'W665', 'W666', 'W667', 'W669', 
                         'W670', 'W671', 'W675', 'W676', 'W677', 'W679', 'W681', 'W683', 'W686', 'W689', 'W691', 'W692', 'W695', 'W696', 'W697', 'W698', 'W699', 'W700', 'W701', 'W702', 'W703', 'W704', 'W705', 
                         'W706', 'W708', 'W709', 'W711', 'W722', 'W766', 'W782'], 

                     2: ['W188', 'W574', 'W59', 'W60', 'W64', 'W69', 'W74'], 
                     3: ['W381', 'W385', 'W394', 'W405', 'W407', 'W408'], 
                     -1: ['W421', 'W427', 'W584'], 
                     4: ['W568', 'W605', 'W613', 'W614', 'W618', 'W619', 'W620', 'W621', 'W625', 'W626', 'W627', 'W631'], 
                     5: ['W792', 'W793', 'W794', 'W795', 'W796', 'W801', 'W802', 'W803', 'W804', 'W805', 'W806', 'W807', 'W809', 'W811', 'W812', 'W813', 'W814', 'W815', 'W816', 'W817', 'W818', 'W820', 'W821', 'W823', 
                         'W824', 'W825', 'W826', 'W828', 'W831', 'W832', 'W834', 'W835', 'W837', 'W840', 'W842', 'W844', 'W845', 'W846', 'W847', 'W848', 'W849', 'W850', 'W851', 'W852', 'W853', 'W856', 'W857', 'W858', 
                         'W859', 'W861', 'W863', 'W864', 'W866', 'W868', 'W869', 'W870', 'W871']}
    else:
        data_name = {1: ['W1615', 'W1628', 'W1669', 'W1675', 'W1686', 'W1707', 'W653', 'W663', 'W668', 'W678', 'W694', 'W707', 'W710'], 
                     2: ['W189', 'W62', 'W63'], 
                     -1: ['W425'], 
                     4: ['W615', 'W634'], 
                     5: ['W810', 'W822', 'W827', 'W830', 'W833', 'W843', 'W854', 'W855', 'W860']}
    
    # 随机挑选符合条件的井，为了随机挑井，这里需要随机得到一个随机数种子
    
    if categorize_id in data_name:
        values = data_name[categorize_id]
        if well_num > len(values):
            raise InvalidNumException(f"{categorize_id} has only {len(values)} elements")
        else:
            use_well_name = random.sample(values, well_num)
    else:
        raise InvalidNumException("categorize_id does not exist")

    print("使用的区块名：" + str(categorize_id))
    print(use_well_name)

    dir = []
    dir_list = os.listdir(path) # 列出了指定路径 self.path 下的所有文件和子目录
    for name in dir_list:
        file_name = name.split('.')[0]
        if file_name in use_well_name:
            dir_path = os.path.join(path, name)
            dir.append(dir_path)
    return dir


def wave_low(data_value, wavelet, level):
    """
    小波变换，并采用低频数据进行增广
    """
    aug_value = []
    aug_value.append([data_value[:, 0], data_value[:, 0]])
    # 对每条曲线分别进行小波变换
    for i in range(1, data_value.shape[1]):
        single_value = data_value[:, i]
        coeffs = pywt.wavedec(single_value, wavelet, level=level)
        # 注意，小波变换之后应该取低频分量，也就是coeffs中第一个分量，其他分量均是高频分量
        encoder_single_value = []
        encoder_single_value.append(single_value)
        interpolate_coeffs = wavelet_aug(coeffs, single_value)
        # 低频分量增广
        encoder_single_value.append(interpolate_coeffs[0]) # 2 * len
        aug_value.append(encoder_single_value) # M * 2 * len
    aug_value = np.array(aug_value) # M * 2 * len -- > M * len * 2
    aug_value = np.swapaxes(aug_value, 1, 2)
    return aug_value

def wave_low_fre(data_value, wavelet, level):
    """
    小波变换：将曲线变换到频域，并选取低频数据后，转回时域
    """


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

def read_data_from_csv(dir, slice_length, slice_step, well_num, frequency_aug, categorize_id, is_train, noise_ration):
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
    Well_path = _read_dir(dir, well_num, categorize_id, is_train)
    # print("well", Well_path)
    print("-------------使用的井数----------------: ", len(Well_path))
    sliced_data = []
    sliced_label = []

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
    for well in Well_path:
        data_all = pd.read_csv(well, header=None)
        data_value = data_all.iloc[:, 2:8].values # .values将数据转换为numpy数组，第 0 维通常表示行。所以 [:, 2:8] 中的 : 表示选择了所有行,得到的数据形式Numpy数组,且shape为(n, 6)
        # 首先对所有数据进行归一化操作
        data_value = scale(data_value, axis=0, with_mean=True, with_std=True, copy=True)
        data_label = data_all.iloc[:, 8].values
 
        # 对每条曲线进行数据增广，这里首先进行频域增广，看看效果
        aug_value = []
        wavelet = 'db10'
        level = 2

        # 通道拼接：傅里叶+加噪，对原始曲线加噪 --> (M * 2, len_seq)
        if frequency_aug == "fft_gauss_channel":
            aug_value.append(data_value[:, 0])
            aug_value.append(data_value[:, 0])
            for i in range(1, data_value.shape[1]):
                single_value = data_value[:, i]
                aug_value.append(single_value)

                waveform_data = np.fft.fft(single_value)
                frequency_data = np.abs(waveform_data)
                aug_value.append(frequency_data)

            aug_value = np.array(aug_value)
            print(aug_value.shape)
            data_value = aug_value
        
        # 通道拼接，小波低频+加噪
        elif frequency_aug == "wave_low_channel":
            aug_value.append(data_value[:, 0])
            aug_value.append(data_value[:, 0])
            for i in range(1, data_value.shape[1]):
                single_value = data_value[:, i]
                aug_value.append(single_value)
                # 小波
                coeffs = pywt.wavedec(single_value, wavelet, level=level)
                interpolate_coeffs = wavelet_aug(coeffs, single_value)
                aug_value.append(interpolate_coeffs[0])
            aug_value = np.array(aug_value)
            print(aug_value.shape)
            data_value = aug_value

        # 通道拼接，平滑滤波+傅里叶        
        elif frequency_aug == "rolling_fft_channel":
            aug_value.append(data_value[:, 0])
            aug_value.append(data_value[:, 0])
            for i in range(1, data_value.shape[1]):
                single_value = data[:, i]
                window = np.ones(int(5)) / float(5)
                smoth_data = np.convolve(single_value, window, 'same')
                aug_value.append(smoth_data)

                waveform_data = np.fft.fft(single_value)
                frequency_data = np.abs(waveform_data)
                aug_value.append(frequency_data)
            aug_value = np.array(aug_value)
            data_value = aug_value

        # 通道拼接，平滑滤波+小波低频增广        
        elif frequency_aug == "rolling_wave_low_channel":
            aug_value.append(data_value[:, 0])
            aug_value.append(data_value[:, 0])
            for i in range(1, data_value.shape[1]):
                single_value = data[:, i]
                window = np.ones(int(5)) / float(5)
                smoth_data = np.convolve(single_value, window, 'same')
                aug_value.append(smoth_data)
                
                # 小波
                coeffs = pywt.wavedec(single_value, wavelet, level=level)
                interpolate_coeffs = wavelet_aug(coeffs, single_value)
                aug_value.append(interpolate_coeffs[0])
            aug_value = np.array(aug_value)
            data_value = aug_value
        
        # 通道拼接，平滑滤波+小波去噪
        elif frequency_aug == "rolling_denosied_aug_channel":
            aug_value.append(data_value[:, 0])
            aug_value.append(data_value[:, 0])
            for i in range(1, data_value.shape[1]):
                single_value = data[:, i]
                window = np.ones(int(5)) / float(5)
                smoth_data = np.convolve(single_value, window, 'same')
                aug_value.append(smoth_data)

                # 小波去噪
                denoised_single_value = wavelet_noising(single_value, wavelet, level)
                aug_value.append(denoised_single_value)
            aug_value = np.array(aug_value)
            data_value = aug_value            
        
        else:
            raise InvalidNumException("frequency_aug does not exist!!!!!!")

        # 将测井数据划分为切片，并取每段切片的中间点标签作为其标签
        # 遍历测井数据的第三个维度，将前两个维度划分为切片
        slice_num = (data_label.shape[0] - slice_length) // slice_step + 1 # 切片个数
        for i in range(slice_num):
            start = i * slice_step
            end = start + slice_length
            # 取中间点标签作为一段切片标签
            mid_index = (start + end) // 2
            a_slice = data_value[:, start:end]

            # 添加高斯噪声
            augment_function = 0.5
            if noise_ration > 0 and augment_function > 0:
                if torch.rand(1).item() < augment_function:
                    for i in range(a_slice.shape[0]):
                        data = a_slice[i, :]
                        noise_std = np.std(data) * noise_ration
                        a_slice[i, :] = data + np.random.normal(0, noise_std, size=data.shape)

            sliced_data.append(a_slice)
            labels = label_name[data_label[mid_index]]
            sliced_label.append(labels)
        
    # 对数据进行变换并return
    sliced_data = torch.from_numpy(np.array(sliced_data)).float()
    sliced_label = torch.from_numpy(np.array(sliced_label)).float()
    return sliced_data, sliced_label, label_name

def dataloader_split(dir, slice_length, slice_step, well_num, frequency_aug, categorize_id, is_train, noise_ration, batchsize):
    sliced_data, sliced_label, label_name = read_data_from_csv(dir, slice_length, slice_step, well_num, frequency_aug, categorize_id, is_train, noise_ration)
    print("sliced_data.shape: ", sliced_data.shape)
    print("sliced_label.shape: ", sliced_label.shape)

    trainX, validX, trainY, validY = train_test_split(sliced_data, sliced_label, test_size=0.1, random_state=42)

    train_set = TensorDataset(trainX, trainY)
    valid_set = TensorDataset(validX, validY)

    trainLoader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True, num_workers=0)
    validLoader = DataLoader(dataset=valid_set, batch_size=batchsize, shuffle=False, num_workers=0)

    return trainLoader, validLoader, label_name

def dataloader(dir_train, dir_val, slice_length, slice_step, train_well_num, val_well_num, frequency_aug, batchsize, train_categorize_id, val_categorize_id, noise_ration):
    train_data, train_label, train_label_name = read_data_from_csv(dir_train, slice_length, slice_step, train_well_num, frequency_aug, train_categorize_id, True, noise_ration)
    valid_data, valid_label, _ = read_data_from_csv(dir_val, slice_length, slice_step, val_well_num, "None", val_categorize_id, False, 0)
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
#     trainLoader, validLoader, label_name = dataloader_split(args.dir, args.slice_length, args.slice_length, args.train_well_num, args.frequency_aug, args.depth_aug, args.classification_name, args.batchsize)
    # label_dict = get_well_label(args)
    # print(label_dict)

    _read_dir("", 2, 2, True)