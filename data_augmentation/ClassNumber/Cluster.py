import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib as mpl

zhfont =mpl.font_manager.FontProperties(fname='../assets/fonts/NotoSansCJK-Regular.ttc')

def write_cate(categorized_names):
    with open('categorized_well_names.txt','w') as file:
        for category in categorized_names:
            file.write(f"Category{category}:\n")

            for name in categorized_names[category]:
                file.write(f"{name}\n")
            file.write("\n")
    print("搞定")

def Cluster():
    # 导入数据
    iris = pd.read_csv("../data/well_coordinate.txt",header=None)
    names = iris.iloc[:,0].values
    iris = iris.iloc[:,3:5].values
    
    # iris = load_iris().data
    # print(iris)
    iris_db = DBSCAN(eps=500,min_samples=8).fit_predict(iris)


    # 统计每一类的数量
    counts = pd.value_counts(iris_db,sort=True)
    categorized_names = {-1:[],0:[], 1:[], 2:[], 3:[],228:[],28:[],100:[],50:[],20:[],10:[]}
    for name, category in zip(names,iris_db):
        categorized_names[category].append(name)
    
    indices_228 = np.random.choice(len(categorized_names[0]),size=228,replace = False)
    categorized_names[228] = categorized_names[0][indices_228]

    indices_28 = np.np.random.choice(indices_228.size,28,replace=False)
    categorized_names[28] = categorized_names[0][indices_28]

    remain_200_ind = np.setdiff1d(indices_228,categorized_names[28])
    categorized_names[0] = categorized_names[0][remain_200_ind]

    categorized_names[1] = np.random.choice(categorized_names[0],size=100)
    categorized_names[2] = np.random.choice(categorized_names[0],size=50)
    categorized_names[3] = np.random.choice(categorized_names[0],size=10)
    write_cate(categorized_names)

    print(counts)
    fig,ax = plt.subplots(2,1,sharex=True,figsize=(24,16))
    
    ax1 = ax[0]
    ax2 = ax[1]
    ax1.tick_params(axis='both',which='both',labelsize=20)
    ax2.tick_params(axis='both',which='both',labelsize=20)
    # 画聚类后的结果
    ax1.set_ylim(509200, 511800)  # outliers only
    ax2.set_ylim(260800, 270200)  # most of the data

    # 隐藏两个坐标轴系列之间的横线
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()

    # 创建轴断刻度线，d用于调节其偏转角度
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax1.scatter(x=iris[:,0],y=iris[:,1], s=40,c=iris_db,cmap='Dark2')

    ax2.scatter(x=iris[:,0],y=iris[:,1],s=40,c=iris_db,cmap='Dark2')

    plt.ylabel("区块纵坐标/km", fontproperties=zhfont,fontsize=30)
    plt.xlabel("区块横坐标/km", fontproperties=zhfont,fontsize=30)

    fig.subplots_adjust(hspace=0.05)  # adjust space between axes
    plt.savefig('1.broken_yaxis.jpg', dpi=800, bbox_inches='tight')
    plt.show()

def Cluster1():
    # 导入数据
    iris = pd.read_csv("../data/well_coordinate.txt",header=None)
    iris = iris.iloc[:,3:5].values
    # iris = load_iris().data
    # print(iris)

    fig,ax = plt.subplots(2,1,sharex=True,figsize=(24,16))
    
    ax1 = ax[0]
    ax2 = ax[1]
    ax1.tick_params(axis='both',which='both',labelsize=20)
    ax2.tick_params(axis='both',which='both',labelsize=20)
    # 画聚类后的结果
    ax1.set_ylim(509200, 511800)  # outliers only
    ax2.set_ylim(260800, 270200)  # most of the data

    # 隐藏两个坐标轴系列之间的横线
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()

    # 创建轴断刻度线，d用于调节其偏转角度
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax1.scatter(x=iris[:,0],y=iris[:,1], s=40,c='darkblue')

    ax2.scatter(x=iris[:,0],y=iris[:,1],s=40,c='darkblue')

    plt.ylabel("区块纵坐标/km", fontproperties=zhfont,fontsize=30)
    plt.xlabel("区块横坐标/km", fontproperties=zhfont,fontsize=30)

    fig.subplots_adjust(hspace=0.05)  # adjust space between axes
    plt.savefig('1.broken_yaxis.jpg', dpi=800, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    Cluster1()