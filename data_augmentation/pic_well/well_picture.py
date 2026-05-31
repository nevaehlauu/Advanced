"""
测井曲线绘制
"""

import torch
import torch.nn.functional as F
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
from scipy.signal import butter, filtfilt
from scipy.special import expit

# data_path = "../data/well_228_old/train/W113.txt"
data_path = "../data/well_228_old/test/W663.txt"


def dwt_wavelet_well(filename, frequency_aug):
    """
    绘制测井曲线变化图,需要为每一列命名,并添加label信息
    """
    logs = pd.read_csv(filename)
    pd.set_option("display.max_rows",10)
    logs.columns = ['Name','Log_Name','Depth','GR','AC','SP','RT1','RT2','Label']

    for i in range(3, 8):
        name = logs.columns[i]
        wavelet = "db10"
        level = 2
        single_value = logs[name]
        # print(single_value)
        if frequency_aug == "denosied":
            # 小波去噪好像没起作用
            logs[name] = wavelet_noising(single_value, wavelet, level)
            print("-----", logs[name])
        
        elif frequency_aug == "wave_low_fre":
            coeffs = pywt.wavedec(single_value, wavelet, level=level)
            print(len(coeffs))
            print(len(coeffs[0]))
            logs[name] = wavelet_aug(coeffs, single_value)[0]
        
        elif frequency_aug == "fft_aug":
            waveform_data = np.fft.fft(single_value)
            # frequence_data = np.abs(np.fft.ifft(waveform_data))
            frequence_data = np.abs(waveform_data)
            logs[name] = frequence_data
        
        elif frequency_aug == "rolling":
            window = np.ones(int(5)) / float(5)
            logs[name] = np.convolve(single_value, window, 'same')
        
        elif frequency_aug == "gauss":
            column_data = logs[name]
            max_values = np.max(column_data)
            min_values = np.min(column_data)
            mean_values = np.mean(column_data)

            if max_values - min_values > mean_values:
                column_data[column_data > mean_values] = column_data[column_data > mean_values] * 6/5
                column_data[column_data <= mean_values] = column_data[column_data <= mean_values] * 4/5
            # noise_std = np.std(single_value) * 0.1
            # logs[name] += np.random.normal(0, noise_std, size=single_value.shape)
            logs[name] = column_data
        
        elif frequency_aug == "gauss_wave_low":
            column_data = logs[name]
            max_values = np.max(column_data)
            min_values = np.min(column_data)
            mean_values = np.mean(column_data)

            if max_values - min_values > mean_values:
                column_data[column_data > mean_values] = column_data[column_data > mean_values] * 6/5
                column_data[column_data <= mean_values] = column_data[column_data <= mean_values] * 4/5
            # noise_std = np.std(single_value) * 0.1
            # logs[name] += np.random.normal(0, noise_std, size=single_value.shape)
            logs[name] = column_data

            coeffs = pywt.wavedec(logs[name], wavelet, level=level)
            print(len(coeffs))
            print(len(coeffs[0]))
            logs[name] = wavelet_aug(coeffs, logs[name])[0]
        
        elif frequency_aug == "gaotong":
            column_data = logs[name]
            nyquist = 0.5 / logs['Depth'].diff().mean() # 奈奎斯特频率
            cutoff = 0.1 * nyquist
            # cutoff = F.sigmoid()(cutoff)
            cutoff = expit(cutoff)
            print(cutoff)
            b, a = butter(4, cutoff, btype='high', analog=False)
            filtered_data = filtfilt(b, a, column_data)
            logs[name] = column_data - 0.5 * (filtered_data - column_data)
        
        # elif frequence_data == "cnn":

    return logs

def make_facies_log_plot(logs):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    # ztop=895; zbot=905
    # ztop=860; zbot=910
    # ztop=1000; zbot=1100
    
    # print(cluster)
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g', linewidth=2)
    ax[1].plot(logs.AC, logs.Depth, '-', linewidth=2)
    ax[2].plot(logs.SP, logs.Depth, '-', color='0.5', linewidth=2)
    ax[3].plot(logs.RT1, logs.Depth, '-', color='r', linewidth=2)
    ax[4].plot(logs.RT2, logs.Depth, '-', color='black', linewidth=2)
    
    for i in range(len(ax)):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=2)

        ax[i].tick_params(axis='both', which='major', labelsize=15)
    
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
    
    ax[0].set_xlabel("GR", fontsize=15)
    # ax[0].set_xlim(0,400)
    # ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("AC", fontsize=15)
    # ax[1].set_xlim(0,200)
    # ax[1].set_xlim(logs.AC.min(),logs.AC.max())
    ax[2].set_xlabel("SP", fontsize=15)
    # ax[2].set_xlim(0,1000)
    # ax[2].set_xlim(logs.SP.min(),logs.SP.max())
    ax[3].set_xlabel("RT1", fontsize=15)
    # ax[3].set_xlim(0,600)
    # ax[3].set_xlim(logs.RT1.min(),logs.RT1.max())
    ax[4].set_xlabel("RT2", fontsize=15)
    # ax[4].set_xlim(0,900)
    # ax[4].set_xlim(logs.RT2.min(),logs.RT2.max())

    for i in range(len(ax)):
        # ax[i].autoscale(axis='both', tight=True)
        ax[i].xaxis.set_label_position("top")
    
    # plt.tight_layout()
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    # f.suptitle('%s: gauss aug'%logs.iloc[0]['Name'], fontsize=25,y=0.94)
    # f.suptitle('cD1', fontsize=25,y=0.94)
    plt.savefig("well_original.png", dpi=600)
    # plt.savefig('plot.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    logs = dwt_wavelet_well(data_path, "None")
    make_facies_log_plot(logs)