import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 用于设置中文字体在 Matplotlib 中的显示
zhfont =mpl.font_manager.FontProperties(fname='../assets/fonts/NotoSansCJK-Regular.ttc')

path= '../data/region_1/train'
name =  {'K1z2+1':   0, 
            'J2a':      1, 
            'J2z':      2, 
            'J1y':      3, 
            'J1f':      4, 
            'chang1':   5, 
            'chang2':   6, 
            'chang3':   7, 
            'chang4+5': 8,
            'chang6':   9,
            }


def ClassNumber():
    dirs = []
    dir_list = os.listdir(path)
    random_num = random.sample(range(0,len(dir_list)),100)
    for num in random_num:
        path_temp = os.path.join(path, dir_list[num])
        dirs.append(path_temp)
    print("=====路径读取成功=====")
    data_labels = [0]*25
    for dir in dirs:
        data_all = pd.read_csv(dir, header=None)
        labels = data_all.iloc[:,8].values
        for label in labels:
            data_labels[name[label]] += 1
    print(data_labels)
    print("=====标签读取成功=====")

    # name_list = ['K1z2+1','J2a','J2z','J1y', 'J1f', 'chang1', 'chang2', 'chang3', 'chang4+5', 'chang6']
    # plt.figure(figsize=(10,6))
    # plt.xlabel("地层标签", fontproperties=zhfont,fontsize=15)
    # plt.ylabel("标签数量", fontproperties=zhfont,fontsize=15)
    # #plt.title("随机十口井各类别标签数量", fontproperties=zhfont,fontsize=15)
    # plt.bar(range(len(data_labels)), data_labels, color=plt.cm.Accent(4),tick_label=name_list)
    # plt.show()



if __name__ == "__main__":
    ClassNumber()