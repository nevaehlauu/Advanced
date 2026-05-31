from secrets import choice
from tkinter import Label
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import scale,MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import rcParams

path= '../data/well_228_old/test'
class dataset:
    '''
    dataset(datapth:存储数据文件(*.txt)的总文件夹) -> train_loader:训练集, valid_loader:验证集
    '''
    def __init__(self,data_path) :
        '''
        data_path: 存储数据文件(*.txt)的总文件夹
        '''
        self.dir = []
        self.ori_x = []
        self.ori_label = []
        self.path = data_path
        self.slice_step = 121
        self.slide_step = 30
        self.batch_size = 12
        self.name =  {'K1z2+1':   0, 
                      'J2a':      1, 
                      'J2z':      2, 
                      'J1y':      3, 
                      'J1f':      4, 
                      'chang1':   5, 
                      'chang2':   6, 
                      'chang3':   7, 
                      'chang4+5': 8,
                      'chang6':   9}

    def _read_dir(self):
        ''''
        批量读取同一文件夹下的数据并将其储存在dir中
        '''
        dir_list = os.listdir(self.path)
        for i in dir_list:
            path_1 = os.path.join(self.path,i)
            self.dir.append(path_1)
        return self.dir

    def Data_pretreat(self):
        '''
        对path文件夹下的所有数据进行切片,转化为图片的形式
        '''
        dir_all = self._read_dir()
        ori_x = []
        ori_label = []

        for Well in dir_all:
            data_all = pd.read_csv(Well, header=None)
            data_value = data_all.iloc[:,3:-1].values
            
            data_label = data_all.iloc[:,-1].values
            scaler = MinMaxScaler(feature_range=(0,255))
            data_value = scaler.fit_transform(data_value).astype(int)
            print(data_value.shape)
            fig = plt.figure(figsize=(2, 10))##更改画布大小
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(data_value,cmap='gray',aspect='auto')
            plt.savefig('gray_well.jpg', dpi=800, bbox_inches='tight')
            plt.show()
            break
            #data_value = scale(data_value,axis=0, with_mean=True, with_std=True, copy=True)
            # 对数据和以slice_step为长度进行切片,并取silce_step个点的label众数为统一的label
            index = data_value.shape[0] - (self.slice_step - 1)
            for i in range(0, index, self.slide_step):
                
                slice_value = data_value[i:i+self.slice_step, :]
                #slice_value = data_value[i:i+self.slice_step]
                
                slice_label = np.array([self.name[data_label[index]] for index in range(i,i+self.slice_step)])
                counts = slice_label[self.slice_step//2+1]
                self.ori_label.append(counts)
                self.ori_x.append(slice_value)
                
        # ori_x: [N,slice_step,6],type=ndarray,N为转换为图片的张数 ->图片形式的数据
        # ori_label: [N,],type=ndarray ->label
        
        ori_x = np.expand_dims(self.ori_x, axis=1)
        ori_x = np.array(ori_x)
        ori_label = np.array(self.ori_label, dtype=np.float)
        # 划分训练集和验证集
        return ori_x, ori_label

if __name__ == "__main__":
    x = dataset(path)
    ori_x, ori_label  = x.Data_pretreat()
    # ori_x = np.squeeze(ori_x)
    # fig = plt.figure(figsize=(1, 2))##更改画布大小
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(ori_x[10000],cmap='gray',aspect='auto')
    # fig.tight_layout()
    # plt.savefig('gray_auto.jpg', dpi=800, bbox_inches='tight')
    # plt.show()
    
