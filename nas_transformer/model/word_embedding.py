"""
单词编码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class wordEmbTransformer(nn.Module):
    def __init__(self, ):
        super(wordEmbTransformer).__init__()


class wordEmbLinear(nn.Module):
    def __init__(self, args):
        """
        flags：包含配置的对象，需要指定线性层，输入嵌入的维度以及丢弃的概率
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