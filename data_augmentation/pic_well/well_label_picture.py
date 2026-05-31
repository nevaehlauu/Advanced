import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import set_option
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
#        '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D','#009966']

facies_colors = ['#F4D03E',  # 橙黄
                 '#F6B041',
                 '#DC7733',  # 棕色
                 '#8C4718',
                 '#9F0D09',
                 '#0C2B40',                
                 '#17557B',
                 '#1C6698',
                 '#77BBE5',
                 '#C7E2F5']  

facies_labels = ['K1z2+1', 'J2a', 'J2z', 'J1y', 'J1f',
                 'chang1', 'chang2','chang3', 'chang4+5','chang6']

facies_lables_name = {'K1z2+1':   0, 
                      'J2a':      1, 
                      'J2z':      2, 
                      'J1y':      3, 
                      'J1f':      4, 
                      'chang1':   5, 
                      'chang2':   6, 
                      'chang3':   7, 
                      'chang4+5': 8,
                      'chang6':   9}

data_path = "../data/well_228_old/train/W113.txt"

print(plt.rcParams['font.family'])

def data_pretreat (filename):
    '''
        将数据转换为DataFrame,并将每一列取名
        添加Lable对应数字化标签列并命名为Facies
    '''
    logs = pd.read_csv(filename)
    # 限制显示的行数
    pd.set_option("display.max_rows",10)
    # 设置列名
    logs.columns = ['Name','Log_Name','Depth','GR','AC','SP','RT1','RT2','Label']
    # 将原本的label转化为数字,并添加到logs中
    labels = []
    for label in logs.Label :
        labels.append(facies_lables_name[label])
    df = pd.DataFrame(labels,columns=['Facies']) # 创建新的DataFrame存储Facies标签
    logs = logs.join(df)

    # 标准化
    scaler = StandardScaler()
    logs[['GR', 'AC', 'SP', 'RT1', 'RT2']] = scaler.fit_transform(logs[['GR', 'AC', 'SP', 'RT1', 'RT2']])
    return logs

def make_facies_log_plot(logs, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    # 创建颜色映射
    cmap_facies = colors.ListedColormap(facies_colors[:], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    # ztop=900; zbot=1200

    
    # cluster=np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)

    # 筛选深度范围内的数值
    mask = (logs.Depth >= ztop) & (logs.Depth <= zbot)

    # 复制Facies列，并重复便于绘图
    cluster = np.repeat(np.expand_dims(logs.loc[mask, 'Facies'].values, 1), 100, 1)

    print(cluster)
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(10, 20))

    ax[0].plot(logs.GR, logs.Depth, '-g', linewidth=2)
    ax[1].plot(logs.AC, logs.Depth, '-', linewidth=2)
    ax[2].plot(logs.SP, logs.Depth, '-', color='0.5', linewidth=2)
    ax[3].plot(logs.RT1, logs.Depth, '-', color='r', linewidth=2)
    ax[4].plot(logs.RT2, logs.Depth, '-', color='black', linewidth=2)

    # 绘制Facies的热图
    im=ax[5].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies)
    
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((12 * ' ').join(['K1z2+1', 'J2a', 'J2z', 
                            'J1y', 'J1f','ch1', 'ch2',
                            'ch3', 'ch4+5','ch6']))
    cbar.set_ticks(range(0,1));
    # cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("AC")
    ax[1].set_xlim(logs.AC.min(),logs.AC.max())
    ax[2].set_xlabel("SP")
    ax[2].set_xlim(logs.SP.min(),logs.SP.max())
    ax[3].set_xlabel("RT1")
    ax[3].set_xlim(logs.RT1.min(),logs.RT1.max())
    ax[4].set_xlabel("RT2")
    ax[4].set_xlim(logs.RT2.min(),logs.RT2.max())
    ax[5].set_xlabel('Facies')
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    # f.suptitle('Well: %s'%logs.iloc[0]['Name'], fontsize=14,y=0.94)
    plt.savefig("well_663.png", dpi=800)
    plt.close()

if __name__ == "__main__":
    logs = data_pretreat(data_path)
    make_facies_log_plot(logs,facies_colors)