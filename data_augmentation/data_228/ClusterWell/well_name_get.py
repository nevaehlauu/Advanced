import os
import pandas as pd


def read_floder(path):
    """
    读取指定文件夹下文件名和去除后缀的文件名，return 只包含文件名（不带后缀）和完整文件名的列表
    """
    dir = []
    dir_without_txt = []
    # 读取指定路径下文件名
    dir_list = os.listdir(path)
    for i in dir_list:
        # 取出后缀信息的文件名
        file_name = i.split('.')[0] 
        dir_without_txt.append(file_name)
        dir_path = os.path.join(path, i)
        dir.append(dir_path)
    return dir_without_txt, dir

def read_well_flie(train_path, test_path, well_name_path):
    """
    读取包含所有井坐标信息的表，并在里面挑出228的坐标信息生成一个新的井坐标文件
    """
    well_name_data = pd.read_csv(well_name_path, header=None)
    well_name = well_name_data.iloc[:, 0] # 得到第一行包含井名列表
    print(well_name)
    print("read")

    train_file_name, _ = read_floder(train_path)
    test_file_name, dir = read_floder(test_path)
    well_228_data = []
    train_data = []
    test_data = []
    for index, name in enumerate(well_name):
        if name in train_file_name:
            # 如果井为228中的井，获取其中一整行的信息
            train_data.append(well_name_data.iloc[index, :].tolist())
            well_228_data.append(well_name_data.iloc[index, :].tolist())
        
        elif name in test_file_name:
            test_data.append(well_name_data.iloc[index, :].tolist())
            well_228_data.append(well_name_data.iloc[index, :].tolist())

    
    # 用生成的228的坐标信息，生成一个新的txt文件
    new_file_path1 = '../data/well_228_old/new_well_data.txt'
    new_file_path2 = '../data/well_228_old/train_data.txt'
    new_file_path3 = '../data/well_228_old/test_data.txt'

    # 保存所有坐标信息
    with open(new_file_path1, 'w') as f:
        for data in well_228_data:
            # f.write(data + '\n')
            f.write(','.join(map(str, data)) + '\n')
    
    # 保存train坐标信息
    with open(new_file_path2, 'w') as f:
        for data in train_data:
            # f.write(data + '\n')
            f.write(','.join(map(str, data)) + '\n')

    # 保存test坐标信息
    with open(new_file_path3, 'w') as f:
        for data in test_data:
            # f.write(data + '\n')
            f.write(','.join(map(str, data)) + '\n')
    
    print(f"New well data file saved at: {new_file_path1}, {new_file_path2}, {new_file_path3}")
    return train_data, test_data, well_228_data

if __name__ == "__main__":
    train_path="../data/well_228_old/train/"
    test_path="../data/well_228_old/test/"
    well_name_path = "../data/well_coordinate.txt"
    train_data, test_data, well_228_data = read_well_flie(train_path, test_path, well_name_path)
    print(test_data)