import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

batch_size = 2
# 我们用向量序列数据模拟一下训练过程，所以第一步先生成要训练的数据
# max_num_words指的是向量和对应词词表长度
# 目前有的可训练资源是
# “今天天气不错” 到 英语 “It is a good day”
# “我想喝水” 到英语 “I want some water”
# 字典表格：
# src字典{0:P, 1:天, 2:今, 3:喝, 4:不, 5:错, 6:想, 7:我, 8:水, 9:气, 10:去}
# tgt字典{0:P, 1:good, 2:it, 3:day, 4:want, 5:water, 6:some, 7:day, 8:I, 9:is 10: a}
max_num_src_words = 10
max_num_tgt_words = 10
model_dim = 8
max_src_len = 8
# # 构建src序列：
src_seq = torch.Tensor([[2, 1, 1, 9, 4, 5, 0, 0],
                        [7, 6, 3, 8, 0, 0, 0, 0]]).to(torch.int32)
tgt_seq = torch.Tensor([[2, 9, 10, 1, 3, 0, 0, 0],
                        [8, 4, 6, 5, 0, 0, 0, 0]]).to(torch.int32)


# 序列最大长度
max_src_sqe_len = 8
max_tgt_sqe_len = 8
max_pos_len = 8
#
# src_len = torch.Tensor([2, 4]).to(torch.int32)
# tgt_len = torch.Tensor([4, 3]).to(torch.int32)
#
# src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max_src_sqe_len-L)), 0)
#                      for L in src_len])
# tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max_src_sqe_len-L)), 0)
#                      for L in src_len])

# print(src_seq)
src_embedding_table = nn.Embedding(max_num_src_words+1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words+1, model_dim)

src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)



# position embedding
pos_mat = torch.arange(max_pos_len).reshape((-1, 1))
i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1, -1))/model_dim)
pe_embedding_table = torch.zeros(max_pos_len, model_dim)
pe_embedding_table[:, 0::2] = torch.sin(pos_mat/i_mat)
pe_embedding_table[:, 1::2] = torch.cos(pos_mat/i_mat)

pe_embedding = nn.Embedding(max_pos_len, model_dim)

pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False) # 实例化一个embedding, 参数调整为position的参数

src_len = torch.Tensor([2, 8]).to(torch.int32)

src_pos = torch.arange(max(src_len))
src_pos = [torch.unsqueeze(src_pos, 0) for _ in src_len]
src_pos = torch.cat(src_pos).to(torch.int32)

src_pe_embedding = pe_embedding(src_pos)
print(src_pe_embedding)


tgt_pos = torch.arange(max(src_len))
tgt_pos = [torch.unsqueeze(tgt_pos, 0) for _ in src_len]
tgt_pos = torch.cat(tgt_pos).to(torch.int32)
tgt_pe_embedding = pe_embedding(tgt_pos)
print(tgt_pe_embedding)
