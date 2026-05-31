from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

#定义搜索空间用到的操作

#测井中空洞卷积需要加上

PRIMITIVES = [
    'none',
    'conv_1x1',
    'conv_3x1',
    'conv_5x1',
    'conv_7x1',
    'conv_11x1',
    'res_block_3',
#     'res_block_5',
    'dil_conv_5x1',
    'dil_conv_7x1'
]

# # NASNet搜索得到的结构
# NASNet = Genotype(
#   normal = [
#     ('sep_conv_5x5', 1),
#     ('sep_conv_3x3', 0),
#     ('sep_conv_5x5', 0),
#     ('sep_conv_3x3', 0),
#     ('avg_pool_3x3', 1),
#     ('skip_connect', 0),
#     ('avg_pool_3x3', 0),
#     ('avg_pool_3x3', 0),
#     ('sep_conv_3x3', 1),
#     ('skip_connect', 1),
#   ],
#   normal_concat = [2, 3, 4, 5, 6],
#   reduce = [
#     ('sep_conv_5x5', 1),
#     ('sep_conv_7x7', 0),
#     ('max_pool_3x3', 1),
#     ('sep_conv_7x7', 0),
#     ('avg_pool_3x3', 1),
#     ('sep_conv_5x5', 0),
#     ('skip_connect', 3),
#     ('avg_pool_3x3', 2),
#     ('sep_conv_3x3', 2),
#     ('max_pool_3x3', 1),
#   ],
#   reduce_concat = [4, 5, 6],
# )

# # AmobaNet搜索得到的结构
# AmoebaNet = Genotype(
#   normal = [
#     ('avg_pool_3x3', 0),
#     ('max_pool_3x3', 1),
#     ('sep_conv_3x3', 0),
#     ('sep_conv_5x5', 2),
#     ('sep_conv_3x3', 0),
#     ('avg_pool_3x3', 3),
#     ('sep_conv_3x3', 1),
#     ('skip_connect', 1),
#     ('skip_connect', 0),
#     ('avg_pool_3x3', 1),
#     ],
#   normal_concat = [4, 5, 6],
#   reduce = [
#     ('avg_pool_3x3', 0),
#     ('sep_conv_3x3', 1),
#     ('max_pool_3x3', 0),
#     ('sep_conv_7x7', 2),
#     ('sep_conv_7x7', 0),
#     ('avg_pool_3x3', 1),
#     ('max_pool_3x3', 0),
#     ('max_pool_3x3', 1),
#     ('conv_7x1_1x7', 0),
#     ('sep_conv_3x3', 5),
#   ],
#   reduce_concat = [3, 4, 6]
# )

# # DARTS搜索得到的两个结构
# # nomal指搜索到的Normal Cell结构，nomal_concat指哪些节点会被concat起来作为最终输出（图中的黄色框）,reduce指搜索到的Reduction Cell
# DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1),
#                              ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)],
#                     normal_concat=[2, 3, 4, 5], 
#                     reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), 
#                            ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], 
#                     reduce_concat=[2, 3, 4, 5]
#                    )

# DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), 
#                             ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], 
#                     normal_concat=[2, 3, 4, 5], 
#                     reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), 
#                             ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], 
#                     reduce_concat=[2, 3, 4, 5]
#                    )

# DARTS = DARTS_V2

# DARTS_10 = Genotype(normal=[('conv_3x1', 0), ('dil_conv_7x1', 1), ('dil_conv_7x1', 2), ('conv_11x1', 1), ('res_block_3', 0), ('conv_11x1', 2), ('res_block_3', 4), ('conv_5x1', 2)], normal_concat=range(2, 6), reduce=[('conv_11x1', 1), ('dil_conv_5x1', 0), ('conv_11x1', 1), ('conv_3x1', 2), ('conv_3x1', 1), ('res_block_3', 3), ('conv_3x1', 1), ('conv_7x1', 3)], reduce_concat=range(2, 6))
# DARTS_10 = Genotype(normal=[('conv_3x1', 1), ('res_block_3', 0), ('conv_11x1', 2), ('conv_7x1', 1), ('dil_conv_7x1', 2), ('conv_3x1', 1), ('conv_11x1', 1), ('conv_3x1', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x1', 0), ('conv_11x1', 1), ('dil_conv_5x1', 0), ('dil_conv_5x1', 1), ('conv_1x1', 3), ('dil_conv_7x1', 2), ('conv_5x1', 4), ('dil_conv_7x1', 2)], reduce_concat=range(2, 6))
# DARTS_10 = Genotype(normal=[('conv_11x1', 1), ('conv_5x1', 0), ('res_block_3', 0), ('res_block_3', 2), ('dil_conv_7x1', 0), ('res_block_3', 3), ('dil_conv_7x1', 3), ('conv_5x1', 2)], normal_concat=range(2, 6), reduce=[('dil_conv_5x1', 0), ('conv_3x1', 1), ('conv_1x1', 0), ('res_block_3', 1), ('dil_conv_7x1', 3), ('conv_3x1', 1), ('conv_1x1', 0), ('dil_conv_5x1', 2)], reduce_concat=range(2, 6))
# DARTS_10 =  Genotype(normal=[('conv_1x1', 1), ('conv_3x1', 0), ('res_block_3', 2), ('conv_1x1', 1), ('conv_11x1', 2), ('res_block_3', 1), ('dil_conv_5x1', 3), ('dil_conv_5x1', 1)], normal_concat=range(2, 6), reduce=[('conv_3x1', 0), ('conv_1x1', 1), ('dil_conv_5x1', 0), ('conv_1x1', 2), ('conv_11x1', 2), ('res_block_3', 3), ('conv_5x1', 3), ('conv_5x1', 0)], reduce_concat=range(2, 6))
# DARTS_10 = Genotype(normal=[('conv_7x1', 0), ('conv_7x1', 1), ('dil_conv_7x1', 2), ('conv_3x1', 1), ('conv_5x1', 1), ('conv_3x1', 3), ('conv_5x1', 0), ('conv_3x1', 3)], normal_concat=range(2, 6), reduce=[('conv_5x1', 1), ('conv_5x1', 0), ('conv_1x1', 1), ('res_block_3', 2), ('conv_5x1', 2), ('dil_conv_7x1', 3), ('conv_7x1', 4), ('conv_3x1', 3)], reduce_concat=range(2, 6))
DARTS_2 = Genotype(normal=[('conv_3x1', 1), ('conv_7x1', 0), ('conv_11x1', 2), ('conv_7x1', 1), ('conv_11x1', 1), ('dil_conv_7x1', 3), ('conv_5x1', 3), ('dil_conv_5x1', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_5x1', 0), ('conv_11x1', 1), ('dil_conv_5x1', 0), ('dil_conv_5x1', 1), ('conv_1x1', 3), ('dil_conv_7x1', 2), ('conv_5x1', 4), ('dil_conv_7x1', 2)], reduce_concat=range(2, 6))