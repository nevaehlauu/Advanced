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


def read_data_from_dat(h5filepath: str, features_name: list, label_name: str, reg_feature_name: str, which_wells: list):
    """
    读取数据集

    :param h5filepath: 文件路径
    :param features_name: 特征名字
    :param label_name: 标签名字
    :param reg_feature_name: 回归
    :param which_wells: 指定井名
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

    # 将reg_feature当作features处理
    if reg_feature_name is not None and reg_feature_name != "":
        features_name = features_name + [reg_feature_name]  # 搞个新的，内存地址已经变了，不会影响外面的 features_name

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


def read_data_from_csv(dir: str, features_name: list, label_name: str, which_wells: list):
    """
        读取数据集
        :param dir 数据目录
        :param features_name 特征曲线名称列表
        :param which_wells 井名列表, 为空时读取目录下所有井数据
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


class MyDataSet(Dataset):
    def __init__(self,
                 filepath: str,
                 desc_filepath: str,
                 which_wells: list = None,
                 features_name: list = None,
                 label_name: str = None,
                 classes: list = None,
                 reg_feature_name: str = None,
                 slice_length: int = 96,
                 slice_step: int = 64,
                 is_train_dataset: bool = True,
                 drop_outlier: bool = False,
                 add_padding: bool = False,
                 data_format: str = "dat",
                 add_filter: bool = False):
        """
        继承于Dataset类，读取.h5数据集并提供__getitem__函数以及一系列的对应处理操作
        :param filepath: 文件路径
        :param desc_filepath: 文件路径
        :param which_wells: 哪些井用于构建数据集
        :param features_name: 特征集合
        :param label_name: 标签的名字，用于从数据集文件中读取标签
        :param reg_feature_name: 用于回归
        :param classes: 标签的类别
        :param slice_length: 切片长度
        :param slice_step: 切片步长
        :param is_train_dataset: 是否为训练集？bool变量前面没有加is，除非是方法
        :param add_padding: 是否在数据集前后填充
        :param data_format: 数据集格式
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
            features_name,  # 井次，若为None，则代表所有井
            label_name,  # 标签名字
            reg_feature_name,
            which_wells,  # 特征曲线的名字
            data_format
        )

        # 筛选异常点
        self.outliers, self.strict_outliers = get_outliers(self.features)

        # for well_name in self.features:
        #     for key in features_name:
        #         shape = self.features[well_name][key].shape
        #         self.features[well_name][key] = wavelet_noising(self.features[well_name][key].squeeze().tolist()).reshape(shape).astype("float32")
        # 归一化
        # normalization(self.features, self.outliers, self.strict_outliers)
        normalization_by_desc_file(desc_filepath, self.features, self.outliers)
        # draw_features(self.features, features_name, "output/features")

        if add_filter:
            for well_name in self.features.keys():
                for key in features_name:
                    shape = self.features[well_name][key].shape
                    self.features[well_name][key] = wavelet_noising(self.features[well_name][key].squeeze().tolist()).reshape(shape).astype("float32")

        print("加载完成，数据集总深度点为：" + str(sum(self.wells_size)) + "\n")

        # -----------------------------------------------------------------------------------------
        # 接下来是切片，是直接切完放到内存里，还是直接在__getitem__里切呢
        # 考虑到大模型可能数据量较大，直接切容易爆内存，所以我决定在__getitem__里切
        # 而self.data不是很大，撑死几百兆，直接放到内存里就可以了

        self.features_name = features_name  # 默认features name一定存在于数据集中，不然会报错
        self.label_name = label_name  # 标签名字
        self.reg_feature_name = reg_feature_name
        self.slice_length = slice_length  # 切片长度
        self.slice_step = slice_step  # 这东西跟着数据集，因为他和模型的参数没关系，仅仅用于划分数据集

        # 地板除。若切片长度为奇数，则为idx上下各n个，有 2n + 1 = slice_length；若切片长度为偶数，上下都是n个，中间点在上半区
        # 所以，若已知点 idx，则整个切片应该为，令 begin_idx = idx - (slice_length-1) // 2
        # 则切片索引应该为 [begin_idx, begin_idx + self.slice_length) 左闭，右开
        self.valid_sliced_midpoints = []  # 有效的切片中点
        well_begin_idx = (self.slice_length - 1) // 2  # 一口井的起始坐标
        cur_well_datum_point = 0  # 每口井idx的基准点
        drop_num = 0  # 抛弃切片的总个数
        for i in range(len(self.wells_name)):
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

                # 如果需要抛弃无效点
                if drop_outlier:
                    if cur_well_outlier[begin_idx:end_idx].sum() > 0:
                        # if cur_well_outlier[begin_idx:end_idx].sum() >= 1:
                        # if cur_well_outlier[begin_idx:end_idx].sum() > (self.slice_length // 10):
                        drop_num += 1
                        continue  # 无效点直接抛弃

                # 将可以作为切片中点的点加入到list中
                self.valid_sliced_midpoints.append(cur_well_datum_point + cur_idx_in_well)

                # 双重检测，防止出错
                if not (cur_idx_in_well == self.get_data_well(cur_well_datum_point + cur_idx_in_well)['idx']):
                    print("err")

            cur_well_datum_point += self.wells_size[i]  # 获取下一口井的基准点

        print("加载完成，数据集总切片数为：" + str(self.__len__()) + " 切片长度为：" + str(self.slice_length) + "抛弃切片个数为：" + str(drop_num) + "\n")

    def __len__(self):
        # 返回数据集长度
        return len(self.valid_sliced_midpoints)

    def __getitem__(self, idx):
        """
        获取指定索引的数据切片
        dataloader，在返回时，会自动将np转化为tensor，不需要我们转换
        """
        a_slice = self.get_a_slice(idx)
        return a_slice

    def get_a_slice(self, idx):
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

        if self.reg_feature_name is not None and self.reg_feature_name in self.features[cur_well_name].keys():
            output[self.reg_feature_name] = self.features[cur_well_name][self.reg_feature_name][cur_idx_in_well].squeeze()  # 保存单独一条的特征曲线
            output["reg_feature"] = output[self.reg_feature_name]  # 引用一下，别名

        # 开始拼接
        cur_slice_features = np.concatenate(tuple(cur_slice_features_list), axis=1)
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
        for i in range(len(self.wells_name)):
            if idx_in_well - self.wells_size[i] < 0:
                output = {'well': self.wells_name[i], "idx": idx_in_well}
                break
            else:
                idx_in_well -= self.wells_size[i]

        return output

    def read_data(self, file_path: str, feature_names: list, label_name: str, reg_feature_name: str, which_wells: list, data_format: str = "dat"):
        if data_format == "dat":
            return read_data_from_dat(file_path, feature_names, label_name, reg_feature_name, which_wells)
        elif data_format == "csv":
            return read_data_from_csv(file_path, feature_names, label_name, which_wells)


def setup_dataloaders(
        # 数据集相关
        batch_size: int,
        num_workers: int,
        slice_step: int = 64,
        which_wells: list = None,
        filepath: str = None,
        desc_filepath: str = None,
        add_padding: bool = False,
        drop_outlier: bool = False,
        data_format: str = "dat",

        # 训练参数相关
        features_name: list = None,
        slice_length: int = 96,
        label_name: str = None,
        classes: list = None,
        reg_feature_name: str = None,

        # 其他
        is_train_dataset: bool = False,
        shuffle: bool = True,
        add_filter: bool = False,
        random_split: bool = False, ):
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
    :param reg_feature_name: 用于回归
    :param slice_length: 切片长度
    :param slice_step: 切片步长
    :param shuffle: 是否打乱数据集
    :param add_padding: 是否在前后填充数据
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
                        features_name, label_name, classes, reg_feature_name,  # 名字
                        slice_length, slice_step,  # 切片相关
                        is_train_dataset, drop_outlier, add_padding,  # bool
                        data_format, add_filter)
    sampler = None  # 用于多GPU分布，暂无

    if random_split:
        val_size = int(len(dataset) * 0.1)
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
            shuffle=shuffle if is_train_dataset else False,  # 验证集 或 测试集，不允许打乱，训练集直接打乱
            num_workers=num_workers,
            worker_init_fn=_worker_init_fn,
            sampler=sampler,
            drop_last=True if is_train_dataset else False  # 验证集 或 测试集，不允许抛弃，训练集直接抛弃
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
