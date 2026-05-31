# 小样本数据分析，包含使用的数据集标签分布
import os
import pandas as pd
# from config.data_228_config import parse_args
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import random
import shutil

####输入为96*6时，即六条测井曲线同时输入时的处理方法
####将增广后的数据作为单独样本进行处理

### 测井数据预处理（这里只处理训练集数据，测试集数据在test中单独处理）
dir = os.getcwd()
print(">>>当前工作目录：", dir) # 返回当前工作目录

#读取txt文件，按逗号划分，返回数组
#处理步骤；1、定义读井的函数，读取所有井。2、按逗号划分，返回数组

#读取train和test的所有井，其中train的一部分train，一部分valid

# 批量读取同一文件夹下的数据并将其存储在dir中
def read_all_well(path):
    dir = []
    dir_list = os.listdir(path) # 列出了指定路径 self.path 下的所有文件和子目录
    for i in dir_list:
        dir_path = os.path.join(path, i) # 使用 os.path.join() 函数将父目录路径 self.path 和子目录名称 i 拼接起来,得到一个完整的子目录路径dir_path
        dir.append(dir_path)
    return dir

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
        # data_name = {1: ['W1424', 'W1584', 'W1585', 'W1586', 'W1587', 'W1588', 'W1589', 'W1590', 'W1591', 'W1594', 'W1595', 'W1596', 'W1597', 'W1610', 'W1612', 'W1613', 'W1614', 'W1629', 'W1630', 'W1633', 
        #                  'W1639', 'W1640', 'W1642', 'W1645', 'W1646', 'W1648', 'W1652', 'W1654', 'W1663', 'W1667', 'W1668', 'W1670', 'W1672', 'W1676', 'W1679', 'W1685', 'W1690', 'W1691', 'W1693', 'W1694', 
        #                  'W1695', 'W1696', 'W1697', 'W1698', 'W1702', 'W1704', 'W1709', 'W651', 'W652', 'W654', 'W655', 'W656', 'W657', 'W658', 'W660', 'W661', 'W662', 'W664', 'W665', 'W666', 'W667', 'W669', 
        #                  'W670', 'W671', 'W675', 'W676', 'W677', 'W679', 'W681', 'W683', 'W686', 'W689', 'W691', 'W692', 'W695', 'W696', 'W697', 'W698', 'W699', 'W700', 'W701', 'W702', 'W703', 'W704', 'W705', 
        #                  'W706', 'W708', 'W709', 'W711', 'W722', 'W766', 'W782'], 
        #              2: ['W792', 'W793', 'W794', 'W795', 'W796', 'W801', 'W802', 'W803', 'W804', 'W805', 'W806', 'W807', 'W809', 'W811', 'W812', 'W813', 'W814', 'W815', 'W816', 'W817', 'W818', 'W820', 'W821', 'W823', 
        #                  'W824', 'W825', 'W826', 'W828', 'W831', 'W832', 'W834', 'W835', 'W837', 'W840', 'W842', 'W844', 'W845', 'W846', 'W847', 'W848', 'W849', 'W850', 'W851', 'W852', 'W853', 'W856', 'W857', 'W858', 
        #                  'W859', 'W861', 'W863', 'W864', 'W866', 'W868', 'W869', 'W870', 'W871'],
        #              3: ['W101', 'W113', 'W115', 'W118', 'W119', 'W125', 'W137', 'W138', 'W145', 'W147', 'W149', 'W150', 'W152', 'W160', 'W161', 'W162', 'W170', 'W172', 'W173', 'W174', 'W176', 'W177', 'W35'], 
        #              4: ['W381', 'W385', 'W394', 'W405', 'W407', 'W408'], 
        #              5: ['W188', 'W574', 'W59', 'W60', 'W64', 'W69', 'W74'], 
        #              6: ['W568', 'W605', 'W613', 'W614', 'W618', 'W619', 'W620', 'W621', 'W625', 'W626', 'W627', 'W631'], 
        #              -1: ['W421', 'W427', 'W584'],
        #              }
        data_name = {1: ['W1424', 'W1584', 'W1585', 'W1586', 'W1587', 'W1588', 'W1589', 'W1590', 'W1591', 'W1594', 'W1595', 'W1596', 'W1597', 'W1610', 'W1612', 'W1613', 'W1614', 'W1629', 'W1630', 'W1633', 
                    'W1639', 'W1640', 'W1642', 'W1645', 'W1646', 'W1648', 'W1652', 'W1654', 'W1663', 'W1667', 'W1668', 'W1670', 'W1672', 'W1676', 'W1679', 'W1685', 'W1690', 'W1691', 'W1693', 'W1694', 
                    'W1695', 'W1696', 'W1697', 'W1698', 'W1702', 'W1704', 'W1709', 'W651', 'W652', 'W654', 'W655', 'W656', 'W657', 'W658', 'W660', 'W661', 'W662', 'W664', 'W665', 'W666', 'W667', 'W669', 
                    'W670', 'W671', 'W675', 'W676', 'W677', 'W679', 'W681', 'W683', 'W686', 'W689', 'W691', 'W692', 'W695', 'W696', 'W697', 'W698', 'W699', 'W700', 'W701', 'W702', 'W703', 'W704', 'W705', 
                    'W706', 'W708', 'W709', 'W711', 'W722', 'W766', 'W782', 'W1615', 'W1628', 'W1669', 'W1675', 'W1686', 'W1707', 'W653', 'W663', 'W668', 'W678', 'W694', 'W707', 'W710'], 
                    # 2: ['W792', 'W793', 'W794', 'W795', 'W796', 'W801', 'W802', 'W803', 'W804', 'W805', 'W806', 'W807', 'W809', 'W811', 'W812', 'W813', 'W814', 'W815', 'W816', 'W817', 'W818', 'W820', 'W821', 'W823', 
                    #         'W824', 'W825', 'W826', 'W828', 'W831', 'W832', 'W834', 'W835', 'W837', 'W840', 'W842', 'W844', 'W845', 'W846', 'W847', 'W848', 'W849', 'W850', 'W851', 'W852', 'W853', 'W856', 'W857', 'W858', 
                    #         'W859', 'W861', 'W863', 'W864', 'W866', 'W868', 'W869', 'W870', 'W871', 'W810', 'W822', 'W827', 'W830', 'W833', 'W843', 'W854', 'W855', 'W860'],
                    # 3: ['W101', 'W113', 'W115', 'W118', 'W119', 'W125', 'W137', 'W138', 'W145', 'W147', 'W149', 'W150', 'W152', 'W160', 'W161', 'W162', 'W170', 'W172', 'W173', 'W174', 'W176', 'W177', 'W35'], 
                    # 4: ['W381', 'W385', 'W394', 'W405', 'W407', 'W408'], 
                    # 5: ['W188', 'W574', 'W59', 'W60', 'W64', 'W69', 'W74', 'W189', 'W62', 'W63'], 
                    # 6: ['W568', 'W605', 'W613', 'W614', 'W618', 'W619', 'W620', 'W621', 'W625', 'W626', 'W627', 'W631', 'W615', 'W634'], 
                    # -1: ['W421', 'W427', 'W584']
                    2: ['W1145', 'W1146', 'W1147', 'W1148', 'W1149', 'W1150', 'W1154', 'W1155', 'W1156', 'W1157', 'W1160', 'W1162', 'W1165', 'W1169', 'W1174', 'W1181', 'W1186', 'W1191', 'W1192', 'W1193', 
                        'W1194', 'W1195', 'W1196', 'W1197', 'W1198', 'W1199', 'W1200', 'W1202', 'W1205', 'W1209', 'W1215', 'W1564', 'W1565', 'W1566', 'W1567', 'W1568', 'W1569', 'W1570', 'W1573', 'W1574', 
                        'W1575', 'W1577', 'W1578', 'W1579', 'W1580', 'W1581', 'W1739', 'W1767', 'W1784', 'W1785', 'W1802', 'W1823', 'W1835', 'W1844', 'W2351', 'W2362', 'W2367', 'W2369', 'W2373', 'W2374', 'W2376']
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

def get_data_label(well):
    # 获取每口井中标签分布
    data_all = pd.read_csv(well, header=None)
    data_label = data_all.iloc[:, 8].values
    # if classification_name == "地质分层":
    #     data_label = data_all.iloc[:, 8].values
    # elif classification_name == "储层划分":
    #     data_label = data_all.iloc[:, 9].values
    # else:
    #     data_label = data_all.iloc[:, 10].values # 后面还需要去除切片中标签为0的数据（应该去除切片中标签为0，还是直接去除原始数据中标签为0的点还需要讨论）
    
    label_num = {}
    # 读取每一口的标签，得到其分布
    for label in data_label:
        if label not in label_num:
            label_num[label] = 1
        else:
            label_num[label] += 1
    return label_num

def get_label_num(dir, well_num, categorize_id, is_train):
    """
    获取不同任务下，指定数量的井中标签分布
    dir: 井的位置
    label_name: 标签信息
    train_well_num: 需要处理的井数
    classification_name: 任务类型，如果为地质分层，应该选择第8列数据，如果为储层划分，第九列数据，如果为油气水划分，应该选择最后一列数据，并且去除切片中标签为0的切片
    """
    # Well_path_all = read_all_well(dir)
    # Well_path = _read_dir(dir, train_well_num)
    Well_path = _read_dir(dir, well_num, categorize_id, is_train)
    # print("well", Well_path)
    # print("-------------总井数------------------", len(Well_path_all))
    print("-------------使用的井数----------------", len(Well_path))
    
    all_label = []
    used_label = []

    # all_label_name = get_well_label(Well_path_all, classification_name)
    # label_name = get_well_label(Well_path, classification_name)

    # 读取每一口井的数据，对其进行操作
    used_label_num = {}
    for well in Well_path:
        label_num = get_data_label(well)
        for label, count in label_num.items():
            if label not in used_label_num:
                used_label_num[label] = count
            else:
                used_label_num[label] += count
    

    # print("data_label_name: ", label_name)
    # print("all_data_label_name: ", all_label_name)
    print(used_label_num)      

    return used_label_num

def label_analysis(label_num_dict, fig_path, title):
    """
    绘制标签分布图
    """
    labels = list(label_num_dict.keys())
    counts = list(label_num_dict.values())
    
    plt.figure(figsize=(25, 15))
    plt.bar(labels, counts)
    plt.xlabel('label', fontsize=25)
    plt.ylabel('count', fontsize=25)
    plt.title(title, fontsize=25)
    plt.xticks(rotation=45, fontsize=25) # 旋转 x 轴标签以避免重叠
    plt.yticks(fontsize=25)
    plt.savefig(fig_path)
    plt.close()

def split_files_into_folders(all_dir, dst_folder1, dst_folder2):
    """
    将文件夹下文件数量按照8：2的比例拆分为两部分，并各自生成新的文件夹
    all_dir: 原始文件夹
    dst_folder1/dst_folder2: 存放拆分后文件的文件夹
    """
    # 创建目标文件夹
    os.makedirs(dst_folder1, exist_ok=True)
    os.makedirs(dst_folder2, exist_ok=True)

    # 获取原始文件夹下所有文件
    all_files = os.listdir(all_dir)

    # 计算拆分比例
    split_point = int(len(all_files) * 0.8)

    # 随机打乱文件列表
    random.shuffle(all_files)

    # 将文件拆分并移动到目标文件夹
    for i, file_name in enumerate(all_files):
        all_files = os.path.join(all_dir, file_name)
        if i < split_point:
            dst_file = os.path.join(dst_folder1, file_name)
        else:
            dst_file = os.path.join(dst_folder2, file_name)
        shutil.move(all_files, dst_file)

if __name__ == '__main__':
    train_dir = "../data/well_data/"
    val_dir = "../data/well_228_old/test/"
    used_label_num = get_label_num(train_dir, 60, 2, True)
    # all_label_num_2, used_label_num_2 = get_label_num(val_dir, val_well_num, "地质分层")

    
    # all_label_num_2, used_label_num_2 = get_label_num(args.dir, args.train_well_num, "油气水划分")
    # all_label_num_3, used_label_num_3 = get_label_num(args.dir, args.train_well_num, "储层划分")

    # label_analysis(all_label_num_1, "data_228/label_analysis/地质分层_train", "Label Distribution")
    label_analysis(used_label_num, "data_228/label_analysis/train_区块2", "Label Distribution")
    # label_analysis(all_label_num_2, "data_228/label_analysis/地质分层_val", "Val Label Distribution")
    # label_analysis(used_label_num_2, "data_228/label_analysis/地质分层_val_all", "Val Label Distribution")

    # split_files_into_folders("../data/well_data/", "../data/augment_data/train_data", "../data/augment_data/val_data")
    print("------all--------", len(os.listdir("../data/well_data/")))
    print("------train--------", len(os.listdir("../data/augment_data/train_data")))
    print("------val--------", len(os.listdir("../data/augment_data/val_data")))
