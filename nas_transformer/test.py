# 获取每一口测试井的准确率、召回率、F1分数

import numpy as np
import torch
import os
import sys
import random

from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# from model.transformer_searched import Network
from model.senet import SENet18
from model.genotypes import Transformer_Encoder as genotype
from sklearn.metrics import recall_score, f1_score

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
        data_name = {1: ['W1424', 'W1584', 'W1585', 'W1586', 'W1587', 'W1588', 'W1589', 'W1590', 'W1591', 'W1594', 'W1595', 'W1596', 'W1597', 'W1610', 'W1612', 'W1613', 'W1614', 'W1629', 'W1630', 'W1633', 
                         'W1639', 'W1640', 'W1642', 'W1645', 'W1646', 'W1648', 'W1652', 'W1654', 'W1663', 'W1667', 'W1668', 'W1670', 'W1672', 'W1676', 'W1679', 'W1685', 'W1690', 'W1691', 'W1693', 'W1694', 
                         'W1695', 'W1696', 'W1697', 'W1698', 'W1702', 'W1704', 'W1709', 'W651', 'W652', 'W654', 'W655', 'W656', 'W657', 'W658', 'W660', 'W661', 'W662', 'W664', 'W665', 'W666', 'W667', 'W669', 
                         'W670', 'W671', 'W675', 'W676', 'W677', 'W679', 'W681', 'W683', 'W686', 'W689', 'W691', 'W692', 'W695', 'W696', 'W697', 'W698', 'W699', 'W700', 'W701', 'W702', 'W703', 'W704', 'W705', 
                         'W706', 'W708', 'W709', 'W711', 'W722', 'W766', 'W782'], 
                     2: ['W792', 'W793', 'W794', 'W795', 'W796', 'W801', 'W802', 'W803', 'W804', 'W805', 'W806', 'W807', 'W809', 'W811', 'W812', 'W813', 'W814', 'W815', 'W816', 'W817', 'W818', 'W820', 'W821', 'W823', 
                         'W824', 'W825', 'W826', 'W828', 'W831', 'W832', 'W834', 'W835', 'W837', 'W840', 'W842', 'W844', 'W845', 'W846', 'W847', 'W848', 'W849', 'W850', 'W851', 'W852', 'W853', 'W856', 'W857', 'W858', 
                         'W859', 'W861', 'W863', 'W864', 'W866', 'W868', 'W869', 'W870', 'W871'],
                     3: ['W101', 'W113', 'W115', 'W118', 'W119', 'W125', 'W137', 'W138', 'W145', 'W147', 'W149', 'W150', 'W152', 'W160', 'W161', 'W162', 'W170', 'W172', 'W173', 'W174', 'W176', 'W177', 'W35'], 
                     4: ['W381', 'W385', 'W394', 'W405', 'W407', 'W408'], 
                     5: ['W188', 'W574', 'W59', 'W60', 'W64', 'W69', 'W74'], 
                     6: ['W568', 'W605', 'W613', 'W614', 'W618', 'W619', 'W620', 'W621', 'W625', 'W626', 'W627', 'W631'], 
                     -1: ['W421', 'W427', 'W584']
                     }
    else:
        data_name = {1: ['W1615', 'W1628', 'W1669', 'W1675', 'W1686', 'W1707', 'W653', 'W663', 'W668', 'W678', 'W694', 'W707', 'W710'], 
                     2: ['W810', 'W822', 'W827', 'W830', 'W833', 'W843', 'W854', 'W855', 'W860'],
                     5: ['W189', 'W62', 'W63'],
                     6: ['W615', 'W634'], 
                     -1: ['W425'], 
                     }
    
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

def read_data_from_csv(dir, slice_length, slice_step, well_num, categorize_id, is_train):
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
    
    
    # model = Network(d_model=512, cell_num=4, num_classes=10, device=2, genotype=genotype)
    # loaded_model = torch.load("log/transformer_pth/best_epoch_model.pth", map_location=torch.device("cpu"))
    # model.load_state_dict(loaded_model)

    model = SENet18(in_channels=5, classes=10)


    with torch.no_grad():
        model.load_state_dict(torch.load("log/senet_log/12_26_2024__21_42_33/best_epoch_model.pth"))
        model.eval()

        rec_total = 0
        f1_total = 0
        acc_total = 0

        # 读取每一口井的数据，对其进行操作
        for well in Well_path:
            sliced_data = []
            sliced_label = []
            data_all = pd.read_csv(well, header=None)
            data_value = data_all.iloc[:, 3:8].values # .values将数据转换为numpy数组，第 0 维通常表示行。所以 [:, 2:8] 中的 : 表示选择了所有行,得到的数据形式Numpy数组,且shape为(n, 6)
            # 首先对所有数据进行归一化操作
            data_value = scale(data_value, axis=0, with_mean=True, with_std=True, copy=True)
            data_label = data_all.iloc[:, 8].values
            aug_value = []

            for i in range(data_value.shape[1]):
                single_value = data_value[:, i]
                aug_value.append(single_value)
            aug_value = np.array(aug_value)
            data_value = aug_value

            slice_num = (data_label.shape[0] - slice_length) // slice_step + 1 # 切片个数
            for i in range(slice_num):
                start = i * slice_step
                end = start + slice_length
                # 取中间点标签作为一段切片标签
                mid_index = (start + end) // 2
                a_slice = data_value[:, start:end]

                sliced_data.append(a_slice)
                labels = label_name[data_label[mid_index]]
                sliced_label.append(labels)
            
            sliced_data = torch.from_numpy(np.array(sliced_data)).float()
            sliced_label = np.array(sliced_label, dtype=np.float32)
            # sliced_label = torch.from_numpy(np.array(sliced_label)).float()

            # 使用预训练模型对每一口井进行测试，获取其准确率
            _, output = model(sliced_data)
            _, predicted = torch.max(output, 1)
            predicted = predicted.numpy()
            accuracy = (predicted == sliced_label).mean()
            print(f"{well.split('/')[-1]}, {accuracy}")
            acc_total += accuracy

            recall = recall_score(sliced_label, predicted, average='macro')
            f1 = f1_score(sliced_label, predicted, average='macro')
            rec_total += recall
            f1_total += f1

            print(f"Recall for {well.split('/')[-1]}: {recall}")
            print(f"F1 Score for {well.split('/')[-1]}: {f1}")

    print(f"Average Accuracy: {acc_total/len(Well_path)}")
    print(f"Average Recall: {rec_total/len(Well_path)}")
    print(f"Average F1 Score: {f1_total/len(Well_path)}")



if __name__ == '__main__':
    dir = "../data/well_228_old/test/"
    slice_length = 96
    slice_step = 64
    well_num = 13
    read_data_from_csv(dir, slice_length, slice_step, well_num, 1, False)