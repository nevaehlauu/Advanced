# 模块调用
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import median_filter


# sgn函数
def sgn(num):
    if (num > 0.0):
        return 1.0
    elif (num == 0.0):
        return 0.0
    else:
        return -1.0

# 对测井数据进行小波变换后，对高频分量进行去噪，再变回时域数据
def wavelet_noising_db10(data: list):
    w = pywt.Wavelet('dB10')  # 选择dB10小波基
    ca3, cd3, cd2, cd1 = pywt.wavedec(data, w, level=3)  # 3层小波分解
    length1 = len(cd1)
    length0 = len(data)

    abs_cd1 = np.abs(np.array(cd1))
    median_cd1 = np.median(abs_cd1) # 计算数组中位数

    sigma = (1.0 / 0.6745) * median_cd1 # 根据中位数计算噪声标准差 sigma。这里使用了一个常数 1.0 / 0.6745 是为了得到一个无偏的噪声标准差估计。
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e)) # 计算软阈值 lamda。这个公式是基于 Donoho 和 Johnstone 在 1994 年提出的 VisuShrink 方法。
    usecoeffs = []
    usecoeffs.append(ca3)

    # 软阈值方法
    for k in range(length1):
        if (abs(cd1[k]) >= lamda / np.log2(2)):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - lamda / np.log2(2))
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda / np.log2(3)):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - lamda / np.log2(3))
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda / np.log2(4)):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - lamda / np.log2(4))
        else:
            cd3[k] = 0.0

    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)  # 信号重构

    # plt.figure(figsize=(120, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(np.arange(len(data)), np.array(data), c='purple')  # , 'o', markersize='1')
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(np.arange(len(recoeffs)), recoeffs, c='purple')  # , 'o', markersize='1')
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(np.arange(len(data)), median_filter(np.array(data), size=5))  # , 'o', markersize='1')
    # plt.savefig("output/dwt/result.svg", dpi=6000, format='svg')  # 保存成svg

    return recoeffs


def wavelet_noising(data: list):
    w = pywt.Wavelet('sym8')  # 选择sym8小波基
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 5层小波分解

    # ca5_resize = np.resize(ca5, len(data))
    # cd5_resize = np.resize(cd5, len(data))
    # cd4_resize = np.resize(cd4, len(data))
    # cd3_resize = np.resize(cd3, len(data))
    # cd2_resize = np.resize(cd2, len(data))
    # cd1_resize = np.resize(cd1, len(data))
    #
    # plot_wavelet([ca5_resize, cd1_resize, cd2_resize, cd3_resize, cd4_resize, cd5_resize])

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))  # 固定阈值计算
    usecoeffs = []
    usecoeffs.append(ca5)  # 向列表末尾添加对象

    # 软硬阈值折中的方法
    a = 0.5

    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0

    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k]) >= lamda):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)  # 信号重构

    # plt.figure(figsize=(120, 6))
    # plt.subplot(3, 1, 1)
    # plt.plot(np.arange(len(data)), np.array(data), c='blue')  # , 'o', markersize='1')
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(np.arange(len(recoeffs)), recoeffs, c='blue')  # , 'o', markersize='1')
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(np.arange(len(data)), median_filter(np.array(data), size=5), c='blue')  # , 'o', markersize='1')
    # plt.savefig("output/dwt/blue.svg", dpi=6000, format='svg')  # 保存成svg

    return recoeffs


def plot_wavelet(wavelets):
    nbr = len(wavelets)
    plt.figure(figsize=(120, 6))
    for i in range(len(wavelets)):
        plt.subplot(nbr, 1, i + 1)
        plt.plot(np.arange(len(wavelets[i])), wavelets[i])
    # plt.show()
    plt.savefig("output/dwt/wavelets.svg", dpi=6000, format='svg')  # 保存成svg


if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 100, endpoint=True)
    y = np.sin(x)

    plt.subplot(4, 1, 1)
    plt.plot(x, y, 'o', markersize='1')

    # 对输入数据加入gauss噪声
    # 定义gauss噪声的均值和方差
    mu = 0
    sigma = 0.12
    for i in range(y.size):
        y[i] += random.gauss(mu, sigma)
        y[i] += random.gauss(mu, sigma)
    plt.subplot(4, 1, 2)
    plt.plot(x, y, 'o', markersize='1')

    data_denoising = wavelet_noising(y)  # 调用小波阈值方法去噪
    plt.subplot(4, 1, 3)
    plt.plot(x, data_denoising, 'o', markersize='1')

    plt.subplot(4, 1, 4)
    plt.plot(x, median_filter(y, size=10), 'o', markersize='1')

    plt.show()
