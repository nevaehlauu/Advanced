# """
# 利用228数据集的坐标信息生成区块图
# """
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# import numpy as np
# import matplotlib as mpl
# from well_name_get import read_floder

# # 设置中文显示
# zhfont =mpl.font_manager.FontProperties(fname='../assets/fonts/NotoSansCJK-Regular.ttc')

# # 包含train、test以及228数据集中井坐标信息的文件
# train_well_data = "../data/well_228_old/train_data.txt"
# test_well_data = "../data/well_228_old/test_data.txt"
# # all_well_data = "../data/well_228_old/new_well_data.txt"
# all_well_data = "../data/well_coordinate.txt"

# def write_cluster(cluster_name):
#     with open('', 'w') as file:
#         for cluster in cluster_name:
#             file.write(f"Category{cluster}: \n")

#             for name in cluster_name[cluster]:
#                 file.write(f"{name}\n")
#             file.write("\n")
#     print("!!!!!!!!!!!!!!!")

# def cluster(path_dir, save_path):
#     """
#     画出train和test的聚类图
#     """
#     train_well = pd.read_csv(path_dir, header=None)

#     # 测试井和训练井坐标信息
#     iris = train_well.iloc[:, 3:5].values # 后两列是坐标信息

#     # eps是每个聚类之间的最小距离，min_samples表示至少有多少个点在一起才能被认为为聚类
#     iris_db = DBSCAN(eps=500, min_samples=20).fit_predict(iris)
#     print(iris_db)

#     # 统计每一类的数量
#     counts = pd.value_counts(iris_db, sort=True)
#     print(counts)

#     # 绘制包含两个子图的figure
#     fig,ax = plt.subplots(2,1,sharex=True,figsize=(24,16))
    
#     ax1 = ax[0]
#     ax2 = ax[1]
#     ax1.tick_params(axis='both',which='both',labelsize=20)
#     ax2.tick_params(axis='both',which='both',labelsize=20)
#     # 画聚类后的结果
#     ax1.set_ylim(509200, 511800)  # outliers only
#     ax2.set_ylim(260800, 270200)  # most of the data

#     ax2.set_xlim(56000, 59000)

#     # 隐藏两个坐标轴系列之间的横线
#     ax1.spines['bottom'].set_visible(False)
#     ax2.spines['top'].set_visible(False)
#     ax1.xaxis.tick_top()

#     # 创建轴断刻度线，d用于调节其偏转角度
#     d = 0.5  # proportion of vertical to horizontal extent of the slanted line
#     kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#                 linestyle="none", color='k', mec='k', mew=1, clip_on=False)
#     ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
#     ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

#     ax1.scatter(x=iris[:,0],y=iris[:,1], s=40,c=iris_db,cmap='nipy_spectral_r')

#     ax2.scatter(x=iris[:,0],y=iris[:,1],s=40,c=iris_db,cmap='nipy_spectral_r')

#     plt.ylabel("Y/km", fontproperties=zhfont,fontsize=30)
#     plt.xlabel("X/km", fontproperties=zhfont,fontsize=30)

#     fig.subplots_adjust(hspace=0.05)  # adjust space between axes
#     plt.savefig(save_path, dpi=800, bbox_inches='tight')
#     plt.show()

# def cluster_trian_and_test(train_dir, test_dir, save_path):
#     """
#     画出train和test的聚类图
#     """
#     train_well = pd.read_csv(train_dir, header=None)
#     test_well = pd.read_csv(test_dir, header=None)

#     # 测试井和训练井坐标信息
#     train_names = train_well.iloc[:, 0].values
#     train_iris = train_well.iloc[:, 3:5].values # 后两列是坐标信息
#     test_names = test_well.iloc[:, 0].values
#     test_iris = test_well.iloc[:, 3:5].values

#     # eps是每个聚类之间的最小距离，min_samples表示至少有多少个点在一起才能被认为为聚类
#     train_iris_db = DBSCAN(eps=50, min_samples=5).fit_predict(train_iris)
#     test_iris_db = DBSCAN(eps=50, min_samples=5).fit_predict(test_iris)
#     print(train_iris_db)
#     print(test_iris_db)

#     # 统计每一类的数量
#     counts = pd.value_counts(train_iris_db, sort=True)
#     print(counts)

#     # for name, category in zip(names, iris_db):

#     # 绘制包含两个子图的figure
#     fig, ax1 = plt.subplots(figsize=(24, 16)) # 创建一个figure，包含两行一列的子图，sharex表示共享x轴
#     ax1.tick_params(axis='both',which='both',labelsize=20)

#     # # 创建轴断刻度线，d用于调节其偏转角度
#     d = 0.5  # proportion of vertical to horizontal extent of the slanted line
#     kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#                 linestyle="none", color='k', mec='k', mew=1, clip_on=False)
#     ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)

#     # 绘制散点图, s=40设置点的大小
#     ax1.scatter(x=train_iris[:,0],y=train_iris[:,1], s=100,c="steelblue",cmap='Dark2', marker='s', label="train well")
#     ax1.scatter(x=test_iris[:,0],y=test_iris[:,1], s=100,c="orange",cmap='Dark2', marker='^', label="test well")

#     for i, name in enumerate(test_names):
#         ax1.annotate(name, (test_iris[i,0], test_iris[i,1]), fontsize=20, color="orange")
#     ax1.legend(loc='upper right', fontsize=30, prop=zhfont, labelcolor='black')
    
#     plt.ylabel("Y/km", fontproperties=zhfont,fontsize=30)
#     plt.xlabel("X/km", fontproperties=zhfont,fontsize=30)

#     # fig.subplots_adjust(hspace=0.05)  # adjust space between axes
#     plt.savefig(save_path, dpi=800, bbox_inches='tight')
#     plt.show()

# def cluster_txt(path_dir, train_path, test_path):
#     """
#     根据228坐标信息生成一个不同井所属区块的文件，并按照训练井和测试井，生成一个包含区块信息的txt文件
#     """
#     data_all = pd.read_csv(path_dir, header=None)

#     # 测试井和训练井坐标信息
#     data_name = data_all.iloc[:, 0].values
#     data_iris = data_all.iloc[:, 3:5].values # 后两列是坐标信息

#     # 训练井和测试井名
#     train_file_name, _ = read_floder(train_path) 
#     test_file_name, dir = read_floder(test_path)

#     # eps是每个聚类之间的最小距离，min_samples表示至少有多少个点在一起才能被认为为聚类
#     iris_db = DBSCAN(eps=50, min_samples=5).fit_predict(data_iris)
#     print(iris_db)

#     train_category_name = {}
#     test_category_name = {}
#     for name, category in zip(data_name, iris_db):
#         if name in train_file_name:
#             if category not in train_category_name:
#                 train_category_name[category] = []
#             train_category_name[category].append(name)
#         elif name in test_file_name:
#             if category not in test_category_name:
#                 test_category_name[category] = []
#             test_category_name[category].append(name)
#         else:
#             print("-----------------", name)
    
#     return train_category_name, test_category_name

# if __name__ == "__main__":
#     save_path = 'well_3000_cluster.jpg'
#     cluster(all_well_data, save_path)
#     # cluster_trian_and_test(train_well_data, test_well_data, save_path)
#     # train_well = "../data/well_228_old/train"
#     # test_well = "../data/well_228_old/test"
#     # train_name, test_name = cluster_txt(all_well_data, train_well, test_well)
#     # print(train_name)
#     # print("---------------")
#     # print(test_name)

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
    iris = pd.read_csv('../data/well_coordinate.txt',header=None)
    names = iris.iloc[:,0].values
    iris = iris.iloc[:,3:5].values
    
    # iris = load_iris().data
    # print(iris)
    iris_db = DBSCAN(eps=500,min_samples=10).fit_predict(iris)


    # 统计每一类的数量
    counts = pd.value_counts(iris_db,sort=True)
    categorized_names = {-1:[],0:[], 1:[], 2:[], 3:[],228:[],28:[],100:[],50:[],20:[],10:[]}
    for name, category in zip(names,iris_db):
        categorized_names[category].append(name)
    
    # 从区块0中随机挑选228口油井
    categorized_names[0] = np.array(categorized_names[0])
    indices_228 = np.random.choice(len(categorized_names[0]),size=228,replace = False)
    categorized_names[228] = categorized_names[0][indices_228]

    # 从228口油井中随机挑选28口测试井
    indices_28 = np.random.choice(indices_228.size,28,replace=False)
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
    ax1.set_ylim(509000, 511800)  # outliers only
    ax2.set_ylim(260500, 271000)  # most of the data

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

    # ax1.scatter(x=iris[:,0],y=iris[:,1], s=40,c=iris_db,cmap='Dark2')
    # ax2.scatter(x=iris[:,0],y=iris[:,1],s=40,c=iris_db,cmap='Dark2')

    # 这里想自定义颜色
    color = ['black' if c == -1
             else '#C55A11' if c == 0
             else '#8FAADC' if c == 1
             else '#F4B183' if c == 2
             else '#5E7594'
             for c in iris_db]
    
    ax1.scatter(x=iris[:,0],y=iris[:,1], s=40,c=color,cmap='Dark2')
    ax2.scatter(x=iris[:,0],y=iris[:,1],s=40,c=color,cmap='Dark2')

    plt.ylabel("区块纵坐标/km", fontproperties=zhfont,fontsize=30)
    plt.xlabel("区块横坐标/km", fontproperties=zhfont,fontsize=30)

    fig.subplots_adjust(hspace=0.05)  # adjust space between axes
    plt.savefig('1.broken_yaxis.jpg', dpi=800, bbox_inches='tight')
    plt.show()

def Cluster1():
    # 导入数据
    iris = pd.read_csv('../data/well_coordinate.txt',header=None)
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
    Cluster()