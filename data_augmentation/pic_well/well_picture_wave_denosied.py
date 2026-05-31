"""
测井曲线绘制
"""

import os
import pandas as pd
import numpy as np
# from config.data_228_config import parse_args
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.colors as colors

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils.wavelet_change import wavelet_aug, wavelet_noising
import pywt
import math

data_path = "../data/well_228_old/train/W113.txt"

# sgn函数
def sgn(num):
    if (num > 0.0):
        return 1.0
    elif (num == 0.0):
        return 0.0
    else:
        return -1.0

def dwt_wavelet_well(filename):
    """
    绘制测井曲线变化图,需要为每一列命名,并添加label信息
    """
    logs = pd.read_csv(filename)
    pd.set_option("display.max_rows",10)
    logs.columns = ['Name','Log_Name','Depth','GR','AC','SP','RT1','RT2','Label']

    depth = logs['Depth'].values
    RT1 = logs['RT1'].values

    start_depth = 895
    end_depth = 905

    # 找到对应深度范围内的索引
    mask = (depth >= start_depth) & (depth <= end_depth)

    # 取出对应范围数据
    selected_RT1 = RT1[mask]
    select_depth = depth[mask]

    # print(selected_RT1)

    # 对这一段数据进行小波变换去噪
    # coeffs = pywt.wavedec(selected_RT1, "db4", level=2)
    # denosied_coeffs = []
    # length0 = len()

    # 分别是重构后的数据，去噪之后数据，还有原始几条数据
    recoeffs, denoised_coeffs, coeffs_1 = wavelet_noising(selected_RT1, "db4", 2)

    print(len(recoeffs)) # 82个点
    print(len(denoised_coeffs[0])) # 低频，25
    print(len(denoised_coeffs[1])) # 高频1，25
    print(len(denoised_coeffs[2])) # 高频2，44

    # 画出每条曲线
    fig1, ax = plt.subplots(1, 3, figsize=(4, 12))
    ax[0].plot(denoised_coeffs[1], np.linspace(0, 24, 25), '-', color='red')
    ax[1].plot(coeffs_1[1], np.linspace(0, 24, 25), '-', color='green')
    ax[2].plot(denoised_coeffs[2], np.linspace(0, 43, 44), '-', color='yellow')

    for i in range(len(ax)):
        ax[i].invert_yaxis()
        ax[i].grid()
        # ax[i].locator_param(axis='x', nbins=3)

    plt.tight_layout()
    plt.savefig("denoised_coeff")
    plt.close()

    # # fig, ax = plt.subplots(1, len(denoised_coeffs), figsize=(12, 8))
    # for i in range(len(denoised_coeffs)):
    #     # 取第几条曲线
    #     fig, ax = plt.subplots(1, len(denoised_coeffs), figsize=(3, 12))
    #     y_idx = np.linspace(0, len(denoised_coeffs[i])-1, len(denoised_coeffs[i]))
    #     ax[i].plot(denoised_coeffs[i], y_idx)
    # plt.savefig("denoised_coeff")
    # plt.close()

def wavelet_noising(data, wavelet, level):
    """
    利用小波变换对测井数据进行去噪
    data: 需要去噪的数据
    wavelet: 小波变换名称
    level: 小波变换级数

    需要考虑小波去噪阈值的选择(如果想出新公式,可以作为一个创新点)
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs_1 = coeffs.copy()
    # print(len(coeffs))
    # print(len(coeffs[0]), "---------")
    # 低频分量
    denoised_coeffs = []
    denoised_coeffs.append(coeffs[0])

    # 软阈值
    length0 = len(data)
    cd1 = coeffs[-1]
    abs_cd1 = np.abs(np.array(cd1))
    median_cd1 = np.median(abs_cd1) # 第一个高频分量中位数

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e)) # 计算软阈值

    for i in range(1, len(coeffs)):
        # 对每一条高频细节分量分别去噪
        length1 = len(coeffs[i])
        for k in range(length1):
            if abs(coeffs[i][k]) >= (lamda / np.log2(1+i)):
                coeffs[i][k] = sgn(coeffs[i][k]) * (abs(coeffs[i][k]) - lamda / np.log2(1+i))
            else:
                coeffs[i][k] = 0.0
    
        denoised_coeffs.append(coeffs[i])

    recoeffs = pywt.waverec(denoised_coeffs, wavelet)
    return recoeffs, denoised_coeffs, coeffs_1

def make_facies_log_plot(logs):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    # ztop=895; zbot=905
    # ztop=860; zbot=910
    
    # print(cluster)
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g', linewidth=2) # 使用实线，并且颜色为绿色
    ax[1].plot(logs.AC, logs.Depth, '-') # 使用实线，但没有指定颜色（通常为蓝色）
    ax[2].plot(logs.SP, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.RT1, logs.Depth, '-', color='r')
    ax[4].plot(logs.RT2, logs.Depth, '-', color='black')
    
    for i in range(len(ax)):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)

        ax[i].tick_params(axis='both', which='major', labelsize=15)
    
    ax[0].set_xlabel("GR", fontsize=15)
    # ax[0].set_xlim(0,200)
    # ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("AC", fontsize=15)
    # ax[1].set_xlim(0,100)
    # ax[1].set_xlim(logs.AC.min(),logs.AC.max())
    ax[2].set_xlabel("SP", fontsize=15)
    # ax[2].set_xlim(0,400)
    # ax[2].set_xlim(logs.SP.min(),logs.SP.max())
    ax[3].set_xlabel("RT1", fontsize=15)
    # ax[3].set_xlim(0,400)
    # ax[3].set_xlim(logs.RT1.min(),logs.RT1.max())
    ax[4].set_xlabel("RT2", fontsize=15)
    # ax[4].set_xlim(0,400)
    # ax[4].set_xlim(logs.RT2.min(),logs.RT2.max())

    for i in range(len(ax)):
        # ax[i].autoscale(axis='both', tight=True)
        ax[i].xaxis.set_label_position("top")
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    f.suptitle('%s: gauss aug'%logs.iloc[0]['Name'], fontsize=25,y=0.94)
    plt.savefig("well.jpg")
    plt.close()

if __name__ == "__main__":
    # logs = dwt_wavelet_well(data_path, "None")
    # make_facies_log_plot(logs)

    dwt_wavelet_well(data_path)