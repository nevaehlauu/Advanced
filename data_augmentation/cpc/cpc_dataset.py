import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
from torchvision import transforms
import os
from sklearn.preprocessing import scale,MinMaxScaler
import pandas as pd
import torch
import scipy
import pywt
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve,AddNoise,Pool
from data_228.readtxt_sample import get_well_label, wave_low, denosied, denosied_aug
from utils.wavelet_change import wavelet_noising, wavelet_aug

class DatasetHandler:
    """
    数据处理
    slice_len: 切片长
    slice_step: 滑动步长
    """
    def __init__(self, data_path, slice_len, slice_step, well_num, classification_name, args, frequency_aug = False):
        self.dir = []
        self.path = data_path
        self.slice_len = slice_len
        self.slice_step = slice_step
        self.well_num = well_num
        self.classification_name = classification_name
        self.frequency_aug = frequency_aug

        self.name = get_well_label(args)
        self.ori_x,self.ori_label = self.read_data()

    # 批量读取同一文件夹下的数据并将其储存在dir中
    def _read_dir(self):
        dir_list = os.listdir(self.path)
        dir_list = dir_list[:self.well_num]
        for i in dir_list:
            path_1 = os.path.join(self.path,i)
            self.dir.append(path_1)
        return self.dir

    # 对path文件夹下的所有数据进行切片
    def read_data(self):  
        dir_all = self._read_dir()
        ori_x = []
        ori_label = []

        for Well in dir_all:
            data_all = pd.read_csv(Well, header=None)
            data_value = data_all.iloc[:,3:8].values # .values将数据转换为numpy数组，第 0 维通常表示行。所以 [:, 2:8] 中的 : 表示选择了所有行,得到的数据形式Numpy数组,且shape为(n, 6)

            if self.classification_name == "地质分层":
                data_label = data_all.iloc[:, 8].values
            elif self.classification_name == "储层划分":
                data_label = data_all.iloc[:, 9].values
            else:
                data_label = data_all.iloc[:, 10].values
        
            data_value = scale(data_value,axis=0, with_mean=True, with_std=True, copy=True)
            
            # 数据增广
            aug_value = []
            wavelet = "db10"
            level = 2
            # 对除深度曲线外的每条曲线进行傅里叶变换
            if self.frequency_aug == "wave_0":
                aug_value.append([data_value[:, 0], data_value[:, 0]])
                for i in range(1, data_value.shape[1]):
                    single_value = data_value[:, i]

            
            # 将小波变化增广后的曲线作为新的样本，这里对低频分量进行插值，增广，这里进行增广时不对深度曲线进行增广
            elif self.frequency_aug == "wave_low":
                aug_value = wave_low(data_value, wavelet, level)
                data_value = aug_value
            
            # 小波变换,这里选取高频分量插值回原来大小后作为新的样本(效果差)
            elif self.frequency_aug == "wave_high":
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
            elif self.frequency_aug == "wave_low_high":
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
            elif self.frequency_aug == "wave_denoised":
                aug_value = denosied(data_value, wavelet, level)
                aug_value = np.swapaxes(aug_value, 0, 1) # len * M
                aug_value = wave_low(aug_value, wavelet, level)
                data_value = aug_value

            # 小波去噪(直接去噪)，也就是将去噪后的样本取代原始样本
            elif self.frequency_aug == "denosied":
                aug_value = denosied(data_value, wavelet, level)
                aug_value = aug_value[:, :, np.newaxis]
                # print(aug_value.shape)
                data_value = aug_value
            
            # 小波去噪，将小波去噪后的样本作为增广数据
            elif self.frequency_aug == "denosied_aug":
                aug_value = denosied_aug(data_value, wavelet, level)
                data_value = aug_value

            # 不进行任何处理
            else:
                # 直接将测井数据在通道拼接,并给其增加一个维度
                data_value = np.swapaxes(data_value, 0, 1)
                data_value = data_value[:, :, np.newaxis]
                # print("----data_value.shape-----", data_value.shape)
            
            # 对数据和以slice_len为长度进行切片,并取silce_step个点的label众数为统一的label
            for j in range(data_value.shape[2]):
                slice_num = (data_label.shape[0] - self.slice_len) // self.slice_step + 1 # 切片数
                for i in range(slice_num):
                    start = i * self.slice_step
                    end = start + self.slice_len
                    mid_index = (start + end) // 2
                    if self.classification_name == "油气水划分" and data_label[mid_index] == '0':
                        continue
                    slice_value = data_value[:, start:end, j]

                    # 添加高斯噪声
                    noise_ration = 0
                    augment_function = 0.5
                    if noise_ration > 0 and augment_function > 0:
                        if torch.rand(1).item() < augment_function:
                            for i in range(slice_value.shape[0]):
                                data = slice_value[i, :]
                                noise_std = np.std(data) * noise_ration
                                slice_value[i, :] = data + np.random.normal(0, noise_std, size=data.shape)
                
                    ori_x.append(slice_value)
                    labels = self.name[data_label[mid_index]]
                    ori_label.append(labels)

        # ori_x: [N, 6, slice_len]，6为曲线条数
        # ori_x = np.expand_dims(ori_x, axis=1) # (N, 1, slice_len, M)
        # ori_x = scipy.ndimage.zoom(ori_x,[1,1,1,2],order=1) # [1,1,1,2]为缩放因子，表示沿着第4个维度进行2倍的放大，其他三个维度不变，order=1表示使用双线性插值的方法进行缩放
        ori_x = torch.from_numpy(np.array(ori_x)).float()
        ori_label = torch.from_numpy(np.array(ori_label, dtype=np.float)).float()    
        return ori_x, ori_label

    def get_n_samples(self):
        # 返回对应子集的样本数量
        return len(self.ori_label)

    def get_pair_by_labels(self, labels):
        # Find samples matching labels，labels指定需要取出的样本标签
        idxs = [] # 用于存储满足labels条件的样本索引
        for i,label in enumerate(labels):
            # 这里获得的是后面setences中传过来的setence，也就是长为len(anchor) + len(netivate) + len(positive)的标签，其中前5为anchor，6是positive，后面为负样本标签
            idx = np.where(self.ori_label == label)[0] # self.ori_label 中与当前 label 相同的所有样本的索引,并存储在 idx 中。
            # 前面i < 4和i==4挑选长度为5的anchor
            if i < 4:
                continue
            elif i == 4:
                index = np.random.randint(0, len(idx) - 4) # 随机选择 idx 列表中的一个索引 index,作为起始位置。
                # 后面连续5个样本，处理比较粗糙，先这样了
                # 从 index 开始,连续向前选取 5 个样本的索引,追加到 idxs 列表中
                for i in range(5):
                    idxs.append(idx[index])
                    index -= 1
            else:
                # 这里随机挑选正样本和负样本
                # 对于其他情况,随机选择 idx 列表中的一个索引,追加到 idxs 列表中
                idx_sel = np.random.choice(idx, 1)[0] 
                idxs.append(idx_sel)

        # 根据 idxs 列表中存储的索引,从 self.ori_x 中取出对应的样本,并存储在 pair 变量中
        # pair = self.ori_x[np.array(idx), 0, :] (N, 1, slice_len, M) --> (slice_len, M)

        # 这里获得了正负样本对
        pair_slice = self.ori_x[np.array(idxs), :] # [N, 6, slice_len] --> len(idxs) * 6 * slice_len

        # Process pair
        # pair = pair.unsqueeze(1) # 升维
        # pair = torch.cat([pair, pair, pair], axis=1) # len(idxs) * 3 * slice_len * M

        # Channel last
        #pair = pair.permute(0, 2, 3, 1)

        # 返回的维度：[len(idxs), M, slice_len], [len(idxs)]
        return pair_slice.float(), labels.int() 
    
class SortedNumberDataset(Dataset):
    """
    获取正负样本对的类
    anchor_num: anchor数量
    negative_num: 负样本数量
    pos_num: 正样本数
    """
    def __init__(self, data_path, batch_size, anchor_num, negative_num, pos_num, slice_len, slice_step, well_num, classification_name, args, frequency_aug):
        self.dataset_handler = DatasetHandler(data_path, slice_len, slice_step, well_num, classification_name, args, frequency_aug)
        self.anchor_num = anchor_num
        self.negative_num = negative_num
        self.pos_num = pos_num
        self.label_name = self.dataset_handler.name

        # 切片数量 // 锚定样本长度 --> 锚定样本数量
        self.n_samples = self.dataset_handler.get_n_samples() // self.anchor_num
        self.n_batch = self.n_samples // batch_size

        # 假设添加一个列表来记录每个样本的seed
        self.seeds = np.random.randint(0, 10, size=self.n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        seed = self.seeds[index]

        # 用于创建一个指定大小的张量,并将所有元素初始化为指定的值seed
        sentence = torch.full([self.anchor_num + self.negative_num + self.pos_num], seed) 
        numbers = torch.arange(0, 10) # 创建一个从0-9的pytorch张量
        sentence[-self.negative_num:] = numbers[numbers!=seed] # 选取sentence张量的后negative_num个值，将其修改为不等于seed的值
        # 前面相当于随机创建一个长为len(sentence)的张量，里面包含着多个正负样本标签，这里获取和创建的标签中label相同的切片（通过随机生成标签获取正负样本对标签）
        pair_slice, labels = self.dataset_handler.get_pair_by_labels(sentence) # 切片维度：[N, 6, slice_len]
        
        anchor_slice = pair_slice[:-self.negative_num-self.pos_num, ...] # 前anchor_num个
        positive_slice = pair_slice[self.anchor_num:self.anchor_num+self.pos_num,...] # 正样本
        negative_slice = pair_slice[-self.negative_num:, ...] # 后面是负样本

        #每个样本都由錨、正、负组成
        return anchor_slice.float(), positive_slice.float(), negative_slice.float(), sentence # sentence是正负样本对的标签，为随机生成

class Dataset_Normal(Dataset):
    """
    获取普通dataset的类
    """
    def __init__(self, data_path, slice_len, slice_step, well_num, classification_name, args, frequency_aug):
        self.dataset_handler = DatasetHandler(data_path, slice_len, slice_step, well_num, classification_name, args, frequency_aug)
        self.ori_x = self.dataset_handler.ori_x
        self.ori_label = self.dataset_handler.ori_label
        self.name = self.dataset_handler.name
    
    def __len__(self):
        return len(self.ori_label)
    
    def __getitem__(self, index):
        a_slice, slice_label = self.ori_x[index], self.ori_label[index]
        return a_slice, slice_label