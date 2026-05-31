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
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.dwt import wavelet_noising, wavelet_noising_db10
from utils.utils import get_normalization_1d
# from utils.draw import draw_features
from utils.well_logs import RES


def transform_label(label, transform_dict):
    """
    将原始的label根据transform_dict里的对应关系，一一转化
    @param label: list, 原始标签
    @param transform_dict: 标签转换的对应字典
    @return: 转换完了的标签
    """
    label = copy.deepcopy(label)  # 遇事不决, copy一下
    # 先获取所有种类对应的mask
    mask = {}
    for key in transform_dict.keys():
        mask[key] = (label == int(key))
    # 再根据mask一一替换
    for key in transform_dict.keys():
        label[mask[key]] = transform_dict[key]
    return label


def read_json(json_cfg_filepath):
    with open(json_cfg_filepath) as f:
        content = json.load(f)
    return content


def get_outliers(features: dict, need_strict: bool = False):
    """
    获取异常点，只支持一维数据！！！
    二维的没写
    :param features:
    :param need_strict:
    :return:
    """

    if features is None:
        features = {}
    outliers = {}  # 只考虑 -99999 的异常值
    strict_outliers = {}  # 更加严格的异常值，目前使用孤立森林算法

    print("扫描数据集的异常值中\n")

    for well_name in features.keys():
        outliers[well_name] = {}
        strict_outliers[well_name] = {}
        for key in features[well_name].keys():
            cur_features = features[well_name][key].squeeze()

            # 普通的outliers只筛选-99999的值
            outliers[well_name][key] = np.zeros(cur_features.shape[0]) != 0  # 初始化，每一个都是false
            outliers[well_name][key] |= (abs(cur_features - (-99999)) < 1)  # 先把-99999剔除掉
            outliers[well_name][key] |= (abs(cur_features - (-9999)) < 1)  # 再把-9999剔除掉
            outliers[well_name][key] |= cur_features < -9000  # 干脆把-9000以下的全扔了
            outliers[well_name][key] |= cur_features > 90000  # 只保留-9000到90000之间的数据

            if need_strict:
                # 进一步的，strict_outliers用机器学习算法进一步筛选异常值
                strict_outliers[well_name][key] = np.copy(outliers[well_name][key])

                reserve = cur_features[~strict_outliers[well_name][key]]  # 把异常值都扔了
                if len(reserve) != 0:  # 是存在整条都是异常值的情况的
                    predictions = IsolationForest().fit(reserve.reshape(-1, 1)).predict(reserve.reshape(-1, 1))  # 用孤立森林来
                    strict_outliers[well_name][key][~strict_outliers[well_name][key]] |= (predictions == -1)  # 等于-1的是异常值

                strict_outliers[well_name][key] = strict_outliers[well_name][key].reshape(-1, 1)

    print("异常值扫描完成\n")
    return outliers, strict_outliers


def normalization_by_desc_file(desc_filepath, features, outliers):
    """
    使用描述文件进行归一化，如果没有文件或者文件不存在的话，直接将异常值置为0
    :param desc_filepath:
    :param features:
    :param outliers:
    :return:
    """

    # 不存在就不归一化了
    if desc_filepath is None or not Path(desc_filepath).exists():
        print("没有描述文件，除了电阻率和SP，不做归一化了")
        for well_name in features.keys():
            for cur_feature_name in features[well_name].keys():
                cur_feature = features[well_name][cur_feature_name]  # 没关系的，我后面会覆盖回去的
                cur_outlier = outliers[well_name][cur_feature_name]  # 最好使用inplace操作，不过没关系的，我后面会覆盖回去的

                if cur_feature_name in RES:
                    # 电阻率需要取对数
                    cur_feature[~cur_outlier] = np.log10(cur_feature[~cur_outlier])

                if cur_feature_name == "SP" and len(cur_feature[~cur_outlier]) != 0:
                    # SP井内做归一化
                    # cur_feature[~cur_outlier] = get_normalization_1d(cur_feature[~cur_outlier], -1, 1)
                    cur_max = cur_feature[~cur_outlier].max()
                    cur_min = cur_feature[~cur_outlier].min()
                    cur_mid = np.median(cur_feature[~cur_outlier][:, 0], axis=0)
                    cur_feature[~cur_outlier] = (cur_feature[~cur_outlier] - cur_mid) / (cur_max - cur_min + 1e-5)

                cur_feature[cur_outlier] = 0  # 异常值清零
                features[well_name][cur_feature_name] = cur_feature  # 覆盖回去
        return

    data_desc = read_json(desc_filepath)

    for well_name in features.keys():
        for cur_proc_info in data_desc["procinfo"]:
            cur_feature_name = list(cur_proc_info.keys())[0]  # 格式是这样子的
            if cur_feature_name not in features[well_name].keys():
                continue

            cur_feature = features[well_name][cur_feature_name]  # 没关系的，我后面会覆盖回去的
            cur_outlier = outliers[well_name][cur_feature_name]  # 最好使用inplace操作，不过没关系的，我后面会覆盖回去的

            for i in range(len(cur_proc_info[cur_feature_name])):
                if cur_proc_info[cur_feature_name][i][0] == "mn":
                    if len(cur_proc_info[cur_feature_name][i]) >= 3:
                        # 直接使用他的参数。为了防止，最后面有一个l之类的符号
                        min_value = float(
                            cur_proc_info[cur_feature_name][i][1] if cur_proc_info[cur_feature_name][i][0][-1].isdigit() else
                            cur_proc_info[cur_feature_name][i][1][:-1])
                        max_value = float(
                            cur_proc_info[cur_feature_name][i][2] if cur_proc_info[cur_feature_name][i][2][-1].isdigit() else
                            cur_proc_info[cur_feature_name][i][2][:-1])
                    else:
                        raise ValueError('使用json文件数据处理出错')
                    if cur_outlier.sum() != 0:
                        cur_feature[cur_outlier] = min_value
                    cur_feature -= min_value
                    if abs(max_value - min_value) > 1e-3:
                        # 为了防止一整列都是-99999的情况。。。
                        cur_feature /= (max_value - min_value)
                # -----------------------------------------------------------------------------------------------------------------------------------
                elif cur_proc_info[cur_feature_name][i][0] == "sd":
                    if len(cur_proc_info[cur_feature_name][i]) >= 3:
                        # 直接使用他的参数。为了防止，最后面有一个l之类的符号
                        mean_value = float(
                            cur_proc_info[cur_feature_name][i][1] if cur_proc_info[cur_feature_name][i][0][-1].isdigit() else
                            cur_proc_info[cur_feature_name][i][1][:-1])
                        std_value = float(
                            cur_proc_info[cur_feature_name][i][2] if cur_proc_info[cur_feature_name][i][2][-1].isdigit() else
                            cur_proc_info[cur_feature_name][i][2][:-1])
                    else:
                        raise ValueError('使用json文件数据处理出错')
                    if cur_outlier.sum() != 0:
                        cur_feature[cur_outlier] = mean_value
                    cur_feature -= mean_value
                    if abs(std_value - 0) > 1e-3:
                        # 为了防止一整列都是-99999的情况。。。
                        cur_feature /= std_value
                # -----------------------------------------------------------------------------------------------------------------------------------
                elif cur_proc_info[cur_feature_name][i][0] == "log":
                    # outlier_idx |= merged_dataset[well_name][:, key_idx] <= 0  # 既然要log，小于等于0的也是异常值
                    cur_feature[cur_feature <= 0] = 1  # 要log的甚至有负数
                    cur_outlier |= (cur_feature <= 0).reshape(-1)  # 你也是异常值
                    if cur_outlier.sum() != 0:
                        cur_feature[cur_outlier] = 1
                    cur_feature = np.log10(cur_feature)

                elif cur_proc_info[cur_feature_name][i][0] == "Subtract the median and then divide by the maximum and minimum values":
                    if len(cur_feature[~cur_outlier]) != 0:
                        cur_max = cur_feature[~cur_outlier].max()
                        cur_min = cur_feature[~cur_outlier].min()
                        cur_mid = np.median(cur_feature[~cur_outlier][:, 0], axis=0)
                        cur_feature[~cur_outlier] = (cur_feature[~cur_outlier] - cur_mid) / (cur_max - cur_min + 1e-5)
                        cur_feature[cur_outlier] = 0

                elif cur_proc_info[cur_feature_name][i][0] == "sd in well":
                    # 自己在井内做标准化
                    if len(cur_feature[~cur_outlier]) != 0:
                        mean_value = cur_feature[~cur_outlier].mean()  # 均值
                        std_value = cur_feature[~cur_outlier].std()  # 标准差
                        cur_feature[~cur_outlier] -= mean_value  # 减去均值
                        cur_feature[~cur_outlier] /= (std_value + 1e-3)  # 除以方差
                    cur_feature[cur_outlier] = 0  # 异常值变成0

            features[well_name][cur_feature_name] = cur_feature
            outliers[well_name][cur_feature_name] = cur_outlier  # 务必使用inplace操作


def read_data_from_dat(h5filepath: str, features_name: list, label_name: str, which_wells: list, classification_name):
    """
    读取数据集

    :param h5filepath: 文件路径
    :param features_name: 特征名字
    :param label_name: 标签名字
    :param reg_feature_name: 回归
    :param which_wells: 指定井名
    :param classification_name 任务名称
    :return:
    features: 特征
    wells_name: 保存井次信息 --> 必须list，python里的dict貌似没有顺序，新版本可能有
    wells_size: 保存每个wells对应的长度
    """
    print("开始读取数据集\n")
    features = {}
    label = {}  # 只有一个标签，所以是label，而不是labels
    wells_name = []
    wells_size = []
    all_well_cnt = 0
    drop_well_cnt = 0

    h5file = h5py.File(h5filepath, "r")
    for well_name in h5file.keys():
        # 第一层，是井次
        # 若该井不符合要求，pass
        if (which_wells is not None) and (well_name not in which_wells):
            continue
        all_well_cnt += 1
        # 做差集，如果features_name中存在该井中不存在的曲线 或 该井不存在标签，则跳过该井
        is_miss_feature = len(list(set(features_name) - set(h5file[well_name].keys()))) != 0
        is_miss_label = label_name is not None and (label_name not in h5file[well_name].keys())
        if is_miss_feature or is_miss_label:
            drop_well_cnt += 1
            continue
        # 记录井名字和该井的词条数目
        features[well_name] = {}
        label[well_name] = {}
        min_length = None  # 统一长度的
        for key in h5file[well_name]:
            # 第二层是曲线名字，只读取特征和标签
            if key in features_name:
                features[well_name][key] = h5file[well_name][key][:].astype("float32").reshape(-1, 1)  # 变成二维后面好拼接
                features[well_name][key][np.isnan(features[well_name][key])] = -99999  # 非法值直接变成-99999
                features[well_name][key][np.isinf(features[well_name][key])] = -99999  # 非法值直接变成-99999
                min_length = features[well_name][key].shape[0] if min_length is None else min(min_length, features[well_name][key].shape[0])
            if label_name is not None and key == label_name:
                label[well_name][key] = h5file[well_name][key][:].astype("float32").reshape(-1, 1)  # 变成二维后面好拼接
                min_length = label[well_name][key].shape[0] if min_length is None else min(min_length, label[well_name][key].shape[0])

                # 对于储层划分任务，需要将标签变为储层和非储层
                # if label_name == "储层划分":
                #     label[well_name][key][label[well_name][key] == "未知"] = "储层"
                #     label[well_name][key][label[well_name][key] != "未知"] = "非储层"
                
                if classification_name == "储层划分" and label_name == "解释结论":
                    label[well_name][key][label[well_name][key] == 0] = 0
                    label[well_name][key][label[well_name][key] != 0] = 1

        # if min_length == 0:
        #     del features[well_name]
        #     del label[well_name]
        #     drop_well_cnt += 1
        #     continue

        # 统一长度
        for key in features[well_name].keys():
            features[well_name][key] = features[well_name][key][:min_length]
        for key in label[well_name]:
            label[well_name][key] = label[well_name][key][:min_length]
        wells_name.append(well_name)  # 记录井的名字
        wells_size.append(min_length)  # 这里默认每条特征曲线都是等长的

    h5file.close()
    print("数据集读取完毕，总井数{:d}，由于曲线缺失抛弃井数{:d}\n".format(all_well_cnt, drop_well_cnt))
    return features, label, wells_name, wells_size


def read_data_from_csv(dir: str, features_name: list, label_name: str, which_wells: list, classification_name):
    """
        读取数据集
        :param dir 数据目录
        :param features_name 特征曲线名称列表
        :param which_wells 井名列表, 为空时读取目录下所有井数据
        :param classification_name 任务名称，为储层划分时将未知变为非储层
    """
    features = {}
    wells_name = []
    # TODO 统计每条特征曲线采样点个数
    sample_len = []
    target_dir = Path(dir)
    csv_files = target_dir.glob("*.csv")
    label = {}  # 只有一个标签，所以是label，而不是labels
    min_length = None  # 统一长度的
    for file_path in csv_files:
        # 获取井名
        # file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_name = Path(file_path).stem
        if (which_wells is not None) and (file_name not in which_wells):
            continue
        wells_name.append(file_name)
        features[file_name] = {}
        label[file_name] = {}
        # 获取数据
        print(file_name)
        df = pd.read_csv(file_path)
        # 获取DataFrame行数
        # sample_len.append(df.shape[0])
        # 读取所有列
        for feature_name in features_name:
            if feature_name in df.columns:
                data_col = df[feature_name].values
                data_col = data_col.reshape(-1, 1)
                features[file_name][feature_name] = data_col
                min_length = features[file_name][feature_name].shape[0] \
                    if min_length is None else min(min_length, features[file_name][feature_name].shape[0])
            if label_name in df.columns and label_name == feature_name:
                label[file_name][label_name] = df[feature_name].values.reshape(-1, 1)
                min_length = label[file_name][label_name].shape[0] \
                    if min_length is None else min(min_length, label[file_name][label_name].shape[0])
                if classification_name == "储层划分" and label_name == "解释结论":
                    label[file_name][label_name][label[file_name][label_name] == 0] = 0
                    label[file_name][label_name][label[file_name][label_name] != 0] = 1
                    
        # 统一长度
        for key in features[file_name].keys():
            features[file_name][key] = features[file_name][key][:min_length]
        for key in label[file_name]:
            label[file_name][key] = label[file_name][key][:min_length]
        sample_len.append(min_length)

    for i in range(len(wells_name)):
        well_name = wells_name[i]
        well_size = sample_len[i]
        for feature_name in features_name:
            if feature_name not in features[well_name].keys():
                # 不存在我人工填充异常值进去
                features[well_name][feature_name] = np.ones((well_size, 1), dtype=float) * -99999
    return features, label, wells_name, sample_len

    
if __name__ == '__main__':
    # main("./configs/training_data_config.yaml")
    pass
