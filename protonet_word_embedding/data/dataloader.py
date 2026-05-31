import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader 
import os
import sys
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from .dataset import read_data_from_csv

class GeneralDataset(Dataset):
    """
    需要获取测井样本和其标签
    """
    def __init__(self, dir, slice_length, slice_step, well_num, frequency_aug, categorize_id, is_train, noise_ration):
        super(GeneralDataset, self).__init__()
        # 每一个测井数据样本，其对应的标签，以及类别字典，这里如果是训练数据is_train设置为true，否则设置为false
        self.well_data, self.well_label, self.well_label_index, self.label_name = read_data_from_csv(dir, slice_length, slice_step, well_num, frequency_aug, categorize_id, is_train, noise_ration)
        self.length = len(self.well_label)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        well_sampler = self.well_data[index]
        well_label_index = self.well_label_index[index]
        return well_sampler, well_label_index
        
class categoriesSampler():
    """
    根据给定标签按类别抽样
    label：样本的标签（这里对应的是每个标签的索引，不是真实标签）
    n_batch：每个epoch中生成的批次数量，有点像N way K shot中的episode概念，也就是每轮训练多少个批次
    n_cls：每个批次中选择的类别数量
    n_per：每个类别中选择的样本数量
    """
    def __init__(self, label,  n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label) # 标签转换为numpy数组
        self.categoried_index = [] # 初始化索引，存储每个类别的索引
        for i in range(max(label) + 1):
            index = np.argwhere(label == i).reshape(-1) # 找到标签label中所有标签为i的样本的位置
            index = torch.from_numpy(index) # 转换为pytorch张量
            self.categoried_index.append(index) # 将每个类别对应的index放入列表中
        
    def __len__(self):
        return self.n_batch # 返回批次数量

    def __iter__(self):
        # 迭代器，使得categoriesSampler类可以被迭代，生成每个批次的数据
        for i_batch in range(self.n_batch): # 遍历每个批次
            batch = [] # 当前批次的样本列表
            classes = torch.randperm(len(self.categoried_index))[:self.n_cls] # 从所有类别中随机挑选n_cls个类别
            for c in classes:
                sample = self.categoried_index[c] # 获取该类别的样本索引
                pos = torch.randperm(len(sample))[:self.n_per] # 从选取的类别挑选n_per个样本
                batch.append(sample[pos])
            batch = torch.stack(batch).t().reshape(-1)  # 将批次样本列表转换为张量并调整形状
            yield batch # 返回当前批次，以便在迭代中使用


def get_dataloader(dir_train, dir_val, slice_length, slice_step, train_well_num, val_well_num, frequency_aug, train_categorize_id, val_categorize_id, noise_ration, n_batch, n_cls, n_per):
    if frequency_aug in ["fft_gauss_channel", "wave_low_gauss_channel", "rolling_fft_channel", "rolling_wave_low_channel", "rolling_denosied_aug_channel"]:
        test_frequency_aug = "test_channel"
    else:
        test_frequency_aug = "None"

    trainset = GeneralDataset(dir_train, slice_length, slice_step,  train_well_num, frequency_aug, train_categorize_id,  True, noise_ration)
    testset = GeneralDataset(dir_val, slice_length, slice_step, val_well_num, test_frequency_aug,  val_categorize_id, False, 0)
    train_sampler = categoriesSampler(trainset.well_label_index, n_batch, n_cls, n_per)
    test_sampler = categoriesSampler(testset.well_label_index, n_batch, n_cls, n_per)

    trainLoader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)
    testLoader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=0, pin_memory=True)

    return trainLoader, testLoader

if __name__ == '__main__':
    pass