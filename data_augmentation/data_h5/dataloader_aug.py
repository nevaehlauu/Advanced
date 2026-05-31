"""
加载数据集
"""
import copy
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import pywt
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.utils import get_normalization_1d
# from utils.draw import draw_features
from utils.well_logs import RES
from utils.wavelet_change import wavelet_aug, wavelet_noising
from dataloader import transform_label, read_json, get_outliers, normalization_by_desc_file, read_data_from_csv, read_data_from_dat

class MyDataSet(Dataset):
    def __init__(self,
                 filepath: str,
                 desc_filepath: str,
                 which_wells: list = None,
                 features_name: list = None,
                 label_name: str = None,
                 classes: list = None,
                 classification_name: str = None, # 用于指明任务为储层划分还是其他
                 slice_length: int = 96,
                 slice_step: int = 64,
                 is_train_dataset: bool = True,
                 drop_outlier: bool = False,
                 data_format: str = "dat",
                 add_filter: str = "None", 
                 wavelet: str = "db10", 
                 level: int = 2,
                 well_use_num: int = 15
                 ):
        """
        继承于Dataset类，读取.h5数据集并提供__getitem__函数以及一系列的对应处理操作
        :param filepath: 文件路径
        :param desc_filepath: 文件路径
        :param which_wells: 哪些井用于构建数据集
        :param features_name: 特征集合
        :param label_name: 标签的名字，用于从数据集文件中读取标签
        :param classes: 标签的类别
        :param slice_length: 切片长度
        :param slice_step: 切片步长
        :param is_train_dataset: 是否为训练集？bool变量前面没有加is，除非是方法
        :param data_format: 数据集格式
        :param add_filter: 是否进行数据增广，以及数据增广的形式
        :param wavelet: 小波变换名
        :param level: 小波变换层数
        :param well_use_num: 使用的井数
        """
        print("开始加载" + ("训练集" if is_train_dataset else "验证集") + "\n")
        # -----------------------------------------------------------------------------------------
        # 类别信息
        if classes is None:
            classes = []
        self.classes = [int(float(x)) for x in classes]  # 先float再int当然是防止输入的标签带小数点
        self.classes_dict = dict(zip(self.classes, list(range(len(self.classes)))))  # 标签转换字典
        self.classes_reversal_dict = dict(zip(list(range(len(self.classes))), self.classes))  # 标签反过来转换的字典

        # 获取数据
        self.features, self.label, self.wells_name, self.wells_size = self.read_data(
            filepath,  # 文件路径
            features_name,  # 特征曲线名
            label_name,  # 标签名字
            which_wells,  # 井次，若为None，则代表所有井
            classification_name, # 任务名称
            data_format
        )

        # 筛选异常点
        self.outliers, self.strict_outliers = get_outliers(self.features)

        # 归一化
        # normalization(self.features, self.outliers, self.strict_outliers)
        normalization_by_desc_file(desc_filepath, self.features, self.outliers)
        # draw_features(self.features, features_name, "output/features")

        # 数据增广：这里主要进行小波变换和小波去噪
        self.add_filter = add_filter # 是否进行数据增广以及数据增广的形式
        self.wavelet = wavelet
        self.level = level
        self.well_use_num = well_use_num
        self.aug_features = {} # 进行数据增广后的features
        self.augment_function = 0.5
        self.noise_ratio = 0.1
        self.is_train_dataset = is_train_dataset

        print("使用的总井数为：" + str(self.well_use_num) + "\n")
        print("加载完成，使用的数据集总深度点为：" + str(sum(self.wells_size[:self.well_use_num])) + "\n")

        # 小波去噪（也可以把小波去噪的数据作为增广数据进行处理）
        if add_filter == "denosied":
            for i in range(well_use_num):
                well_name = self.wells_name[i]
                for key in features_name:
                    if key == "DEPTH":
                        continue
                    else:
                        data = self.features[well_name][key]
                        shape = data.shape
                        self.features[well_name][key] = wavelet_noising(data.squeeze().tolist(), self.wavelet, self.level).reshape(shape).astype("float32")
        
        # 将小波去噪的数据作为增广数据
        elif add_filter == "denosied_aug":
            for i in range(well_use_num):
                well_name = self.wells_name[i]

                ######### 这里增广这样写有问题，每一次都新建一个aug_features[well_name]
                self.aug_features[well_name] = {}
                for key in features_name:
                    if key == "DEPTH":
                        self.aug_features[well_name][key] = self.features[well_name][key]
                    else:
                        data = self.features[well_name][key]
                        shape = data.shape
                        self.aug_features[well_name][key] = wavelet_noising(data.squeeze().tolist(), self.wavelet, self.level).reshape(shape).astype("float32")
        
        # 小波变换(低频小波变换，不固定验证集，直接用低频小波变换的结果替代原始数据时准确率有提升)
        elif add_filter == "wave_low":
            for i in range(well_use_num):
                well_name = self.wells_name[i]
                self.aug_features[well_name] = {}
                for key in features_name:
                    if key == "DEPTH":
                        self.aug_features[well_name][key] = self.features[well_name][key]
                    else:
                        data = self.features[well_name][key]
                        shape = data.shape
                        data = data.squeeze().tolist()
                        coeffs = pywt.wavedec(data, wavelet, level)
                        encoder_coeffs = wavelet_aug(coeffs, data)[0] # 取低频分量
                        # self.features[well_name][key].append(encoder_coeffs) # 
                        self.aug_features[well_name][key] = encoder_coeffs.reshape(shape).astype("float32")
        
        # 高频增广
        elif add_filter == "wave_high":
            for i in range(well_use_num):
                well_name = self.wells_name[i]
                self.aug_features[well_name] = {}
                for key in features_name:
                    if key == "DEPTH":
                        self.aug_features[well_name][key] = self.features[well_name][key]
                    else:
                        data = self.features[well_name][key]
                        shape = data.shape
                        data = data.squeeze().tolist()
                        coeffs = pywt.wavedec(data, wavelet, level)
                        encoder_coeffs = wavelet_aug(coeffs, data)[0] # 取低频分量
                        # self.features[well_name][key].append(encoder_coeffs) # 
                        self.aug_features[well_name][key] = encoder_coeffs.reshape(shape).astype("float32")
                
        # 小波去噪+小波变换（先去噪，用去噪之后的数据进行小波变换）
        elif add_filter == "wave_denosied":
            for i in range(well_use_num):
                well_name = self.wells_name[i]
                self.aug_features[well_name] = {}
                for key in features_name:
                    if key == "DEPTH":
                        self.aug_features[well_name][key] = self.features[well_name][key]
                    else:
                        data = self.features[well_name][key]
                        shape = data.shape
                        data = data.squeeze().tolist()
                        self.features[well_name][key] = wavelet_noising(data, self.wavelet, self.level).reshape(shape).astype("float32")
                        coeffs = pywt.wavedec(data, wavelet, level)
                        encoder_coeffs = wavelet_aug(coeffs, data)[0] 
                        self.aug_features[well_name][key] = encoder_coeffs.reshape(shape).astype("float32")
        
        # 傅里叶变换增广
        elif add_filter == "fft_aug":
            for i in range(well_use_num):
                well_name = self.wells_name[i]
                self.aug_features[well_name] = {}
                for key in features_name:
                    data = self.features[well_name][key]

                    # self.aug_features[well_name][key] = 
        
        # 时域上增广，比如三次样条插值等
        # elif add_filter == "depth_aug":
            

        # print("加载完成，数据集总深度点为：" + str(sum(self.wells_size)) + "\n")

        # -----------------------------------------------------------------------------------------
        # 接下来是切片，是直接切完放到内存里，还是直接在__getitem__里切呢
        # 考虑到大模型可能数据量较大，直接切容易爆内存，所以我决定在__getitem__里切
        # 而self.data不是很大，撑死几百兆，直接放到内存里就可以了

        self.features_name = features_name  # 默认features name一定存在于数据集中，不然会报错
        self.label_name = label_name  # 标签名字
        self.classification_name = classification_name # 任务名称
        self.slice_length = slice_length  # 切片长度
        self.slice_step = slice_step  # 这东西跟着数据集，因为他和模型的参数没关系，仅仅用于划分数据集

        # 地板除。若切片长度为奇数，则为idx上下各n个，有 2n + 1 = slice_length；若切片长度为偶数，上下都是n个，中间点在上半区
        # 所以，若已知点 idx，则整个切片应该为，令 begin_idx = idx - (slice_length-1) // 2
        # 则切片索引应该为 [begin_idx, begin_idx + self.slice_length) 左闭，右开
        self.valid_sliced_midpoints = []  # 有效的切片中点
        well_begin_idx = (self.slice_length - 1) // 2  # 一口井的起始坐标
        cur_well_datum_point = 0  # 每口井idx的基准点
        drop_num = 0  # 抛弃切片的总个数

        # for i in range(len(self.wells_name)):
        for i in range(self.well_use_num):
            well_name = self.wells_name[i]  # 目前井的名字
            # 获取异常行，只要一行里的特征，有一个点是异常的，那这一行我就认为是异常的
            cur_well_outlier = None
            for key in self.features_name:
                if cur_well_outlier is None:
                    cur_well_outlier = np.copy(self.outliers[well_name][key])
                else:
                    cur_well_outlier |= self.outliers[well_name][key]

            # 逐个判断是否为有效切片中点
            for cur_idx_in_well in range(well_begin_idx, self.wells_size[i], self.slice_step):  # 起点 终点 步长，左开右闭
                begin_idx = cur_idx_in_well - (self.slice_length - 1) // 2  # 左闭
                end_idx = begin_idx + self.slice_length  # 右开
                if end_idx > self.wells_size[i]:
                    break

                # 抛弃无效点：对于地址划分和油气水划分需要去掉未知标签
                if drop_outlier:
                    if cur_well_outlier[begin_idx:end_idx].sum() > 0:
                        # if cur_well_outlier[begin_idx:end_idx].sum() >= 1:
                        # if cur_well_outlier[begin_idx:end_idx].sum() > (self.slice_length // 10):
                        drop_num += 1
                        continue  # 无效点直接抛弃
                if self.classification_name != "储层划分" and self.label[well_name][self.label_name][cur_idx_in_well] == 0:
                    drop_num += 1
                    continue
                
                # 按照不同任务类型去除未知或是重排标签，这里多传入一个参数：具体任务名称classification_task
                # self.valid_sliced_midpoints.append(cur_well_datum_point + cur_idx_in_well)

                # 将可以作为切片中点的点加入到list中
                self.valid_sliced_midpoints.append(cur_well_datum_point + cur_idx_in_well)

                # 双重检测，防止出错
                if not (cur_idx_in_well == self.get_data_well(cur_well_datum_point + cur_idx_in_well)['idx']):
                    print("err")

            cur_well_datum_point += self.wells_size[i]  # 获取下一口井的基准点

        print("加载完成，数据集总切片数为：" + str(self.__len__()) + " 切片长度为：" + str(self.slice_length) + " 抛弃切片个数为：" + str(drop_num) + "\n")

    def __len__(self):
        # 返回数据集长度
        return len(self.valid_sliced_midpoints)

    def __getitem__(self, idx):
        """
        获取指定索引的数据切片：这里需要选择使用原始切片还是增广之后的切片
        dataloader，在返回时，会自动将np转化为tensor，不需要我们转换
        """
        if self.is_train_dataset and self.add_filter is not None and self.add_filter != "denosied":
            # 随机挑选使用原始切片还是增广后的切片，以及随机添加高斯噪声
            feature_slice = self.get_a_feature_slice(idx)
            aug_feature_slice = self.get_a_aug_feature_slice(idx)
            if torch.rand(1).item() < self.augment_function:
                a_slice = feature_slice
            else:
                a_slice = aug_feature_slice
            
            # 添加高斯噪声
            if torch.rand(1).item() < self.augment_function:
                for key in self.features_name:
                    data = a_slice[key]
                    noise_std = np.std(data) * self.noise_ratio
                    a_slice[key] = data + np.random.normal(0, noise_std, size=data.shape)

            # 三次样条插值
        else:
            a_slice = self.get_a_feature_slice(idx)

            # 添加高斯噪声
            if torch.rand(1).item() < self.augment_function:
                for key in self.features_name:
                    data = a_slice[key]
                    noise_std = np.std(data) * self.noise_ratio
                    a_slice[key] += data + np.random.normal(0, noise_std, size=data.shape)

        return a_slice

    def get_a_feature_slice(self, idx):
        """
        获取指定索引切片的数据，返回内容如下所示：
        @param idx:
        @return: output，是一个字典，由以下字段组成，以切片长度64为例
        {
            "features": 按照顺序将特征曲线拼接完的完整切片 64 * 6
            "AC" / "GR" / "DEPTH" / ... : 特征曲线, 64 * 1
            "label": 标签, 1
            "multi_label": 多标签, 64
        }
        """
        cur_slice_features_list = []
        tmp = self.get_data_well(self.valid_sliced_midpoints[idx])
        cur_well_name = tmp["well"]  # 当前数据的井
        cur_idx_in_well = tmp["idx"]  # 当前数据集在当前井中的索引

        # 获取该切片的起点和中点
        begin_idx = cur_idx_in_well - (self.slice_length - 1) // 2  # 左闭
        end_idx = begin_idx + self.slice_length  # 右开

        output = {}
        # ---- 获取一个切片需要的数据 ----
        for key in self.features_name:
            # 低版本python字典是无序的，所以这里还是用list了
            cur_slice_features_list.append(self.features[cur_well_name][key][begin_idx:end_idx])
            output[key] = self.features[cur_well_name][key][begin_idx:end_idx]  # 保存单独一条的特征曲线

        # 开始拼接
        cur_slice_features = np.concatenate(tuple(cur_slice_features_list), axis=1) # slice_length * M
        # cur_slice_features = np.expand_dims(cur_slice_features, axis=0)
        output["features"] = cur_slice_features.astype("float32")

        if len(self.label[cur_well_name]) == 1:
            # 目前只支持，有且只有一个标签
            label = self.label[cur_well_name][self.label_name][cur_idx_in_well]  # 一个切片的总标签
            output["label"] = transform_label(label.astype("int64").squeeze(), self.classes_dict)  # 标签要按照顺序转换一下
            multi_label = self.label[cur_well_name][self.label_name][begin_idx:end_idx]  # 深度一对一的标签
            output["multi_label"] = transform_label(multi_label.astype("int64").squeeze(), self.classes_dict)  # 标签要按照顺序转换一下

        return output

    def get_a_aug_feature_slice(self, idx):
        """
        获取指定索引切片的数据，返回内容如下所示：
        @param idx:
        @return: output，是一个字典，由以下字段组成，以切片长度64为例
        {
            "features": 按照顺序将特征曲线拼接完的完整切片 64 * 6
            "AC" / "GR" / "DEPTH" / ... : 特征曲线, 64 * 1
            "label": 标签, 1
            "multi_label": 多标签, 64
        }
        """
        cur_slice_features_list = []
        tmp = self.get_data_well(self.valid_sliced_midpoints[idx])
        cur_well_name = tmp["well"]  # 当前数据的井
        cur_idx_in_well = tmp["idx"]  # 当前数据集在当前井中的索引

        # 获取该切片的起点和中点
        begin_idx = cur_idx_in_well - (self.slice_length - 1) // 2  # 左闭
        end_idx = begin_idx + self.slice_length  # 右开

        output = {}
        # ---- 获取一个切片需要的数据 ----
        for key in self.features_name:
            # 低版本python字典是无序的，所以这里还是用list了
            cur_slice_features_list.append(self.aug_features[cur_well_name][key][begin_idx:end_idx])
            output[key] = self.aug_features[cur_well_name][key][begin_idx:end_idx]  # 保存单独一条的特征曲线

        # 开始拼接
        cur_slice_features = np.concatenate(tuple(cur_slice_features_list), axis=1) # slice_length * M
        # cur_slice_features = np.expand_dims(cur_slice_features, axis=0)
        output["features"] = cur_slice_features.astype("float32")

        if len(self.label[cur_well_name]) == 1:
            # 目前只支持，有且只有一个标签
            label = self.label[cur_well_name][self.label_name][cur_idx_in_well]  # 一个切片的总标签
            output["label"] = transform_label(label.astype("int64").squeeze(), self.classes_dict)  # 标签要按照顺序转换一下
            multi_label = self.label[cur_well_name][self.label_name][begin_idx:end_idx]  # 深度一对一的标签
            output["multi_label"] = transform_label(multi_label.astype("int64").squeeze(), self.classes_dict)  # 标签要按照顺序转换一下

        return output

    def get_data_well(self, idx):
        """
        解析获得该idx数据对应的井次和井次里的idx
        :@return: {'well': 井的名字, "idx": well_idx数据在该井中的索引}
        """
        output = {}
        idx_in_well = idx
        # for i in range(len(self.wells_name)):
        for i in range(self.well_use_num):
            if idx_in_well - self.wells_size[i] < 0:
                output = {'well': self.wells_name[i], "idx": idx_in_well}
                break
            else:
                idx_in_well -= self.wells_size[i]

        return output

    def read_data(self, file_path: str, feature_names: list, label_name: str, which_wells: list, classification_name, data_format: str = "dat"):
        if data_format == "dat":
            return read_data_from_dat(file_path, feature_names, label_name, which_wells, classification_name)
        elif data_format == "csv":
            return read_data_from_csv(file_path, feature_names, label_name, which_wells, classification_name)


def setup_dataloaders(
        # 数据集相关
        batch_size: int,
        num_workers: int,
        slice_step: int = 64,
        which_wells: list = None,
        filepath: str = None,
        desc_filepath: str = None,
        drop_outlier: bool = False,
        data_format: str = "dat",

        # 训练参数相关
        features_name: list = None,
        slice_length: int = 96,
        label_name: str = None,
        classes: list = None,
        classification_name: str = None,

        # 其他
        is_train_dataset: bool = False,
        shuffle: bool = True,
        add_filter: str = "False", # 是否进行数据增广，以及数据增广的形式
        wavelet: str = "db10", 
        level: int = 2,
        well_use_num: int = 10,
        random_split: bool = False):
    """
    提供数据集，参数有点多，详见配置文件
    :param batch_size: 批次大小
    :param num_workers: dataloader加载数据集花费的进程数量
    :param filepath: 文件路径
    :param desc_filepath: 文件路径
    :param which_wells: 哪些井用于构建数据集
    :param features_name: 特征集合
    :param label_name: 标签的名字，用于从数据集文件中读取标签
    :param classes: 标签类别信息
    :param classification_task: 任务名称，需要根据不同任务确定怎么处理未知标签
    :param slice_length: 切片长度
    :param slice_step: 切片步长
    :param shuffle: 是否打乱数据集
    :param drop_outlier: 抛弃异常值
    :param is_train_dataset: 是否为训练集？bool变量前面没有加is，除非是方法
    :param data_format: 数据集格式, 数据格式可能会随时调整, 此处传入数据格式保证灵活性
    :param random_split: 是否要帮你划分训练集和测试集，默认91开
    :param add_filter: 是否要帮你划分训练集和测试集，默认91开
    :return: loader: DataLoader类，供pytorch训练网络使用
    """

    def _worker_init_fn(worker_id):
        """
        Worker init fn to fix the seed of the workers
        用于固定数据加载过程中的随机数种子的（作用于子进程）
        :@param worker_id: 进程id
        :@return: 无
        """

        seed = torch.initial_seed() % 2 ** 32 + worker_id  # worker_id 可以不加，每个epoch都不一样，**优先级很高的
        np.random.seed(seed)
        random.seed(seed)

    dataset = MyDataSet(filepath, desc_filepath,  # 文件
                        which_wells,  # 井
                        features_name, label_name, classes, classification_name,  # 名字
                        slice_length, slice_step,  # 切片相关
                        is_train_dataset, drop_outlier, # bool
                        data_format, add_filter, wavelet, level, well_use_num)
    sampler = None  # 用于多GPU分布，暂无

    if random_split:
        val_size = int(len(dataset) * 0.7)
        train_size = int(len(dataset)) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=shuffle if is_train_dataset else False,  # 验证集 或 测试集，不允许打乱，训练集直接打乱
            num_workers=num_workers,
            worker_init_fn=_worker_init_fn,
            sampler=sampler,
            drop_last=True if is_train_dataset else False  # 验证集 或 测试集，不允许抛弃，训练集直接抛弃
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,  # 验证集 或 测试集，不允许打乱，训练集直接打乱
            num_workers=num_workers,
            worker_init_fn=_worker_init_fn,
            sampler=sampler,
            drop_last=False  # 验证集 或 测试集，不允许抛弃，训练集直接抛弃
        )

        return train_dataset, train_loader, val_dataset, val_loader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle if is_train_dataset else False,  # 验证集 或 测试集，不允许打乱，训练集直接打乱
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        sampler=sampler,
        drop_last=True if is_train_dataset else False  # 验证集 或 测试集，不允许抛弃，训练集直接抛弃
    )

    return dataset, loader  # 数据集获取成功，外面一般用的都是loader
    
if __name__ == '__main__':
    # main("./configs/training_data_config.yaml")
    pass
