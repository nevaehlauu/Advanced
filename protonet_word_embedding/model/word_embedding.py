"""
单词编码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class wordEmbTransformer(nn.Module):
    """
    将标签转为one-hot向量，然后将其转到高维特征空间
    """
    def __init__(self, n_cls):
        super(wordEmbTransformer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_cls, 256, 3, 1, 1), 
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 512, 3, 1, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 1)
        )
    
    def forward(self, one_hot_labels):
        x = self.conv(one_hot_labels)
        label_feature = F.adaptive_avg_pool1d(x, 1)
        label_feature = label_feature.view(label_feature.size(0), -1)
        lambda_k = torch.sigmoid(self.fc(label_feature))
        return x, lambda_k



class wordEmbLinear(nn.Module):
    def __init__(self, args):
        """
        args：包含配置的对象，需要指定线性层，输入嵌入的维度以及丢弃的概率
        """
        super(wordEmbLinear, self).__init__()
        
        self.args = args
        
        # 定义线性层
        if args.mlp_type == 'linear':
            self.mlp_layer = nn.Linear(args.embedding_dim, 512)
        
        # 定义非线性层
        elif args.mlp_type == 'non-linear':
            self.mlp_layer_1 = nn.Linear(args.embedding_dim, 300)
            self.mlp_layer_2 = nn.Linear(300, 512)
            self.dropout = nn.Dropout(p=args.mlp_dropout)

        # 权重初始化
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, embeddings):
        if self.flags.mlp_type == 'linear':
            h = self.mlp_layer(embeddings)
        elif self.flags.mlp_type == 'non-linear':
            h = F.relu(self.mlp_layer_1(embeddings))
            h = self.dropout(h)
            h = self.mlp_layer_2(h)
        return h