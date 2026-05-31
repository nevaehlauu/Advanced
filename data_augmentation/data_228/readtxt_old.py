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
num_classes = 10
dir = os.getcwd()
print(">>>当前工作目录：", dir) # 返回当前工作目录

#读取txt文件，按逗号划分，返回数组
#处理步骤；1、定义读井的函数，读取所有井。2、按逗号划分，返回数组

#读取文件所有行，并变为数组形式
def read_txt(file):
    f = open(file)
    result = [] #初始化

    for line in f.readlines(): 
        line = line.strip().split(',')
        result.append(line) 

    return np.array(result)

#读取train和test的所有井，其中train的一部分train，一部分valid
Well_train = glob.glob('../data/well_228_old/train/*.txt') #glob.glob()获取指定目录下的所有井
Well_test = glob.glob('../data/well_228_old/test/*.txt')

name = {'K1z2+1': 0, 'J2a': 1, 'J2z': 2, 'J1y':3, 'J1f': 4, 'chang1': 5, 'chang2': 6, 'chang3': 7, 'chang4+5': 8,  'chang6': 9}

# name =  {'K1z2+1': 0, 'J2a': 1, 'J2z': 2, 'J1y': 3, 'J1f': 4, 'chang1': 5, 'chang2': 6, 'chang3': 7, 'chang4+5': 8, 'chang6': 9,
#         'y2': 10, 'y3': 11, 'y4+5': 12, 'y6': 13, 'y7': 14, 'y8': 15, 'y9': 16, 'y10': 17, 'chang7': 18, 'chang8': 19, 'chang9': 20,
#         'chang4': 21, 'chang5': 22, 'y4': 23, 'y5': 24}

well_used = []
ori_x = []
ori_label = []
step = 96


for j in range(0, len(Well_train)):
# for j in range(0, 20):
    # 小样本学习，仅对20口井进行数据处理
    if name == {}: #没有标签就丢弃这行数据
        break
    Well = read_txt(Well_train[j]) 

    Well_x = []
    Well_y = []
    for line in Well:
        # 将井的数据，即三列到倒数第二列传到Well_x中，井的标签传入Well_y中
        Well_x.append(np.array([float(x) for x in line[2:8]])) 
        Well_y.append(line[8])


    well_used.append(Well_train[j][18:-4]) #看看使用了哪些井，其中[18：-4]为Well_228_old/train/*.txt中"/*"的位置

    Well_x = np.array(Well_x)
    Well_x = scale(Well_x, axis=0, with_mean=True, with_std=True, copy=True) 

    for i in range(0, len(Well_x)-(step-1), 96): #以32为步长时效果最好
        #当标签不止label_num不止标签中的十种时如下
        # if Well_y not in name:
        #     name[Well_y[i]] = label_num
        #     label_num += 1

        ori_x.append([x for x in Well_x[i:i+step]]) 
        labels = [name[Well_y[index]] for index in range(i, i+step)]
        counts = np.bincount(labels) 
        ori_label.append(np.argmax(counts)) #np.argmax()表示取出counts最大值所对应的索引, 找出出现最多的类别数（标签取众数）

ori_x = np.swapaxes(ori_x, 1, 2) # (77674, 96, 6)变为(x, 6, 96)，channel为6，sequence_length=96
ori_x = np.array(ori_x)
ori_label = np.array(ori_label, dtype=np.float32)

print("ori_x.shape: ", ori_x.shape)
print("ori_label.shape: ", ori_label.shape)
# ori_x = ori_x[:, np.newaxis, :, :] #np.newaxis在对应位置加为1的维度，把ori_x的shape从(77674, ,96, 6)变为了(77674,1 ,96, 6)，相当于RGB通道数为1
print("ori_x.shape_now：", ori_x.shape)

# np.savez("data_well/train_data_w200_1d.npz", x=ori_x, y=ori_label)

trainX, validX, trainY, validY = train_test_split(ori_x, ori_label, test_size=0.2, random_state=42)
trainX = torch.from_numpy(trainX).float()
validX = torch.from_numpy(validX).float()
trainY = torch.from_numpy(trainY).float()
validY = torch.from_numpy(validY).float()

train_set = TensorDataset(trainX, trainY)
valid_set = TensorDataset(validX, validY)

trainLoader = DataLoader(dataset=train_set, batch_size=1024, shuffle=True, num_workers=0)
validLoader = DataLoader(dataset=valid_set, batch_size=1024, shuffle=False, num_workers=0)