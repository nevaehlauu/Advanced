"""
训练用配置文件
使用yacs.config库进行训练参数的配置
在yaml文件里修改具体配置
"""

from yacs.config import CfgNode
from pathlib import Path

########################################################################################################################
cfg = CfgNode()
cfg.name = ''  # Run name
########################################################################################################################
### ARCH  主要的、基础的参数
########################################################################################################################
cfg.arch = CfgNode()
cfg.arch.seed = 1244  # Random seed for Pytorch/Numpy initialization 上限为 4,294,967,295
cfg.arch.epochs = 50  # Maximum number of epochs  训练轮次
cfg.arch.search_epochs = 20 # NAS搜索时训练轮次
cfg.arch.patience = 40  # 40轮之内妹有变化，那就结束
cfg.arch.device = 'cpu'  # 设备 例如 cuda:0  cuda:1  cpu
cfg.arch.platform = ''  # 记录一下操作平台
########################################################################################################################
### MODEL
########################################################################################################################
cfg.model = CfgNode()
cfg.model.author = "author's name"
cfg.model.type = "mae"  # 或者 "vit"，该字段暂时没有用
cfg.model.log_dir_path = 'output/mae'  # 该模型的日志的根目录
cfg.model.log_dir_name = None  # 为空，则加入时间戳，反之，就是这个名字
cfg.model.pretrained_filepath = None  # 预训练模型路径，vit可以使用mae的，只要进行类继承就行
cfg.model.save_checkpoint = True  # 是否保存checkpoint
cfg.model.save_criterion_list = ["loss", "acc"]
cfg.model.best_acc = 0  # 记录最佳精确度 base on vals，best_acc对应最好的checkpoint
cfg.model.cur_acc = 0  # 记录最佳精确度 base on vals，cur_acc对应当前的checkpoint
cfg.model.best_acc_epochs = -1  # 记录最佳精确度的epoch
cfg.model.min_loss = 0  # 记录最小loss
cfg.model.cur_loss = 0  # 记录目前loss
cfg.model.min_loss_epochs = -1  # 记录最小损失的epoch
cfg.model.end_epochs = -1  # 当前的轮次
########################################################################################################################
### MODEL.SCHEDULER  学习率衰减
########################################################################################################################
cfg.model.scheduler = CfgNode()
cfg.model.scheduler.decay = 0.5  # Scheduler decay rate
cfg.model.scheduler.lr_epoch_divide_frequency = 2  # 几轮衰减一次
########################################################################################################################
### MODEL.OPTIMIZER
########################################################################################################################
cfg.model.optimizer = CfgNode()
cfg.model.optimizer.learning_rate = 0.001
cfg.model.optimizer.weight_decay = 0.0
cfg.model.optimizer.min_lr = 0.0
########################################################################################################################
### MODEL.PARAMS  模型参数
########################################################################################################################
cfg.model.params = CfgNode()
# 通用的参数
cfg.model.params.generic = CfgNode()
cfg.model.params.generic.slice_length = 480  # 切片长度
cfg.model.params.generic.patch_height = 16
cfg.model.params.generic.features_name = []  # 选取哪些特征曲线，跟着模型比较好
cfg.model.params.generic.label_name = "" # 选择不同任务标签
cfg.model.params.generic.classification_name = "" # 不同任务名称
cfg.model.params.generic.pretrained_filepath = None #预训练文件保存位置
cfg.model.params.generic.draw_plt = True # 是否保存图片

# 训练vqvae的参数 --> tokenizer的训练
cfg.model.params.vqvae = CfgNode()  # mae用的参数
cfg.model.params.vqvae.hidden_size = 256  # size of the latent vectors (default: 256)
cfg.model.params.vqvae.k = 512  # number of latent vectors (default: 512)
cfg.model.params.vqvae.scale_factor = 4  # 最小是4，必须是2的倍数！
cfg.model.params.vqvae.beta = 1.0  # contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)
cfg.model.params.vqvae.delete_codebook = False  # 删掉codebook之后就变成AE了
cfg.model.params.vqvae.customization = False  # 删掉codebook之后就变成AE了
cfg.model.params.vqvae.res_block_nbr = 2  # 残差链接数量
cfg.model.params.vqvae.encoder_type = "normal"  # normal或者senet
cfg.model.params.vqvae.normalization = False  # 是否在decoder的最后面加一层，让输出归一化到 -1, 1之间（与数据集处理保持一致）
cfg.model.params.vqvae.loss_fn = "mse_loss"  # l1_loss or mse_loss

# 训练mae的参数，mae的训练
cfg.model.params.mae = CfgNode()  # mae用的参数
cfg.model.params.mae.embed_dim = 1024
cfg.model.params.mae.depth = 24
cfg.model.params.mae.num_heads = 16
cfg.model.params.mae.decoder_embed_dim = 512
cfg.model.params.mae.decoder_depth = 8
cfg.model.params.mae.decoder_num_heads = 16
cfg.model.params.mae.mlp_ratio = 4.0
cfg.model.params.mae.mask_ratio = 0.75  # 遮掩比例，按照mae源码的处理，这个跟着模型，禁止超过1

# 训练vit的参数，vit模型基于mae模型，与mae共用参数
cfg.model.params.vit = CfgNode()  # vit用的参数，vit部分参数基于mae
cfg.model.params.vit.classes = []  # 有哪些类别？
cfg.model.params.vit.label_name = ''  # 标签的名字叫什么？

# 后面名字改成vit
cfg.model.params.transformer = CfgNode()  # vit用的参数，vit部分参数基于mae
cfg.model.params.transformer.embed_dim = 1024
cfg.model.params.transformer.depth = 24
cfg.model.params.transformer.num_heads = 16
cfg.model.params.transformer.mlp_ratio = 4.0
cfg.model.params.transformer.qkv_bias = True
# cfg.model.params.transformer.num_classes = 100  # 即len(classes)

cfg.model.params.mlp = CfgNode()  # vit用的参数，vit部分参数基于mae
cfg.model.params.mlp.fc_layers = [256, 256]

# classification，和vit一样也是下游任务，网络用的不一样
cfg.model.params.classification = CfgNode()  # 节点
cfg.model.params.classification.classes = []  # 有哪些类别
cfg.model.params.classification.label_name = ''  # 标签的名字叫什么？
cfg.model.params.classification.tokenizer = 'VQVAE'  # tokenizer用什么（如果没有tokenizer_pretrained，将从上面的网络参数里获取）
cfg.model.params.classification.tokenizer_cfg = cfg.model.params.vqvae  # 引用上面的配置
cfg.model.params.classification.tokenizer_frozen = True  # 冻结参数和bn
cfg.model.params.classification.tokenizer_lr_factor = 1.  # 冻结参数和bn
cfg.model.params.classification.tokenizer_pretrained = []  # 分组的特征曲线，一组的曲线共用一个tokenizer，里面是元组
cfg.model.params.classification.features_name_by_grp = [["AC", "GR"], ["AT"]]  # 表示AC,GR共用一个tokenizer, AT用一个tokenizer
cfg.model.params.classification.downstream_input_size = (16, 512)  # 下游任务输入的尺寸
cfg.model.params.classification.downstream_model = "SENet"  # 下游任务用什么模型

# 回归任务
cfg.model.params.regression = CfgNode()  # 节点
cfg.model.params.regression.reg_feature_name = None  # 只允许回归一条线
cfg.model.params.regression.tokenizer = 'VQVAE'  # tokenizer用什么（如果没有tokenizer_pretrained，将从上面的网络参数里获取）
cfg.model.params.regression.tokenizer_frozen = True  # 冻结参数和bn
cfg.model.params.regression.tokenizer_pretrained = []  # 分组的特征曲线，一组的曲线共用一个tokenizer，里面是元组
cfg.model.params.regression.features_name_by_grp = [["AC", "GR"], ["AT"]]  # 表示AC,GR共用一个tokenizer, AT用一个tokenizer
cfg.model.params.regression.downstream_input_size = (16, 512)  # 下游任务输入的尺寸
cfg.model.params.regression.downstream_model = "SENet"  # 下游任务用什么模型

########################################################################################################################
### DATASETS
########################################################################################################################
cfg.datasets = CfgNode()
########################################################################################################################
### DATASETS.AUGMENTATION  暂无
########################################################################################################################
cfg.datasets.augmentation = CfgNode()
########################################################################################################################
### DATASETS.TRAIN 训练集
########################################################################################################################
cfg.datasets.train = CfgNode()
cfg.datasets.train.batch_size = 128  # 批次大小
cfg.datasets.train.num_workers = 0  # pytorch加载数据集的进程数，仅对linux有效，windows强制为0
cfg.datasets.train.slice_step = 100  # 切片步长
cfg.datasets.train.which_wells = None  # 指定井名字
cfg.datasets.train.filepath = '../TL_Dataset/'  # 训练集路径
cfg.datasets.train.desc_filepath = None  # 训练集路径
cfg.datasets.train.add_padding = False  # 训练集路径
cfg.datasets.train.drop_outlier = False  # 是否抛弃异常值
cfg.datasets.train.add_filter = False  # 数据集的格式
cfg.datasets.train.data_format = "dat"  # 数据集的格式
########################################################################################################################
### DATASETS.val 验证集，与训练集关键字完全一致
########################################################################################################################
cfg.datasets.val = CfgNode()
cfg.datasets.val.batch_size = 128  # valid batch size
cfg.datasets.val.num_workers = 0  # 仅对linux有效，windows强制为0；
cfg.datasets.val.slice_step = 1  # 切片步长
cfg.datasets.val.which_wells = None
cfg.datasets.val.filepath = '../TL_Dataset/'
cfg.datasets.val.desc_filepath = None  # 训练集路径
cfg.datasets.val.add_padding = False  # 训练集路径
cfg.datasets.val.drop_outlier = False  # 验证集不允许丢失，该参数固定为false
cfg.datasets.val.add_filter = False  # 数据集的格式
cfg.datasets.val.data_format = "dat"  # 是否抛弃异常值
########################################################################################################################
### THESE SHOULD NOT BE CHANGED
########################################################################################################################
cfg.configs = ''  # Run configuration file
cfg.default = str(Path(__file__).absolute())  # 固定，就是当前文件的路径，这里用的是绝对路径


########################################################################################################################


def get_cfg_defaults():
    return cfg.clone()
