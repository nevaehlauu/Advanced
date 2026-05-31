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
        

def get_dataloader(dir_train, dir_val, slice_length, slice_step, train_well_num, val_well_num, frequency_aug, train_categorize_id, val_categorize_id, noise_ration, batchsize):
    if frequency_aug in ["fft_gauss_channel", "wave_low_gauss_channel", "rolling_fft_channel", "rolling_wave_low_channel", "rolling_denosied_aug_channel"]:
        test_frequency_aug = "test_channel"
    else:
        test_frequency_aug = "None"

    trainset = GeneralDataset(dir_train, slice_length, slice_step,  train_well_num, frequency_aug, train_categorize_id,  True, noise_ration)
    testset = GeneralDataset(dir_val, slice_length, slice_step, val_well_num, test_frequency_aug,  val_categorize_id, False, 0)

    trainLoader = DataLoader(dataset=trainset, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)
    testLoader = DataLoader(dataset=testset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)

    return trainLoader, testLoader

if __name__ == '__main__':
    pass