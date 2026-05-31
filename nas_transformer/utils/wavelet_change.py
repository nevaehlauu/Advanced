####小波变换部分代码，包含小波去噪和小波低频插值，以及小波细节分量（高频）插值

import pywt
import math
import numpy as np
from scipy import interpolate

# sgn函数
def sgn(num):
    if (num > 0.0):
        return 1.0
    elif (num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(data, wavelet, level):
    """
    利用小波变换对测井数据进行去噪
    data: 需要去噪的数据
    wavelet: 小波变换名称
    level: 小波变换级数

    需要考虑小波去噪阈值的选择(如果想出新公式,可以作为一个创新点)
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
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
    return recoeffs[:len(data)]

def wavelet_aug(coeffs, single_value):
    """
    对小波变换后的测井数据进行增广插值
    这里既包含高频，又包含低频分量
    """
    result = []
    # print(single_value, "---------")
    # print("--------", coeffs)
    for i in range(len(coeffs)):
        ori_index = np.linspace(0, len(single_value)-1, len(single_value)) # 原始信息坐标
        coeff_index = np.linspace(0, len(single_value)-1, len(coeffs[i])) # 需要插值的信号坐标
        f_interpolate = interpolate.interp1d(coeff_index, coeffs[i], kind='linear')
        padded_coeff = f_interpolate(ori_index)
        result.append(padded_coeff)
    return result

def add_gauss_noisy(data, noise_ratio):
    """
    对输入测井数据添加高斯噪声
    测井数据存在振幅和深度两个方向，添加高斯噪声时仅对振幅方向添加噪声
    :param noise_ration：添加噪声的比例
    添加的噪声均值为0，方差为np.std(x) * noise_ratio，其中前者为原始数据标准差
    """
    noise_std = np.std(data) * noise_ratio
    aug_data = data + np.random.normal(0, noise_ratio, size=data.shape)
    return aug_data

def add_salt_pepper_noise(data, noise_ration):
    """
    对测井数据添加椒盐噪声
    """


if __name__ == "__main__":
    pass