import torch
import torch.nn as nn
import math
from torch.autograd import Variable

embedding = nn.Embedding(10, 3)
input1 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 8]])
print(input1)
output1 = embedding(input1)
print(output1)


nn.Embedding(5,6)
# 构建一个embedding层，将词向量维度扩展
class Embeddings(nn.Module):
    def __init__(self, vocab, dim):
        # d_model是词嵌入的维度
        # vocab是词表的大小
        super(Embeddings, self).__init__()
        self.dim = dim
        self.vocab = vocab
        self.lut = nn.Embedding(vocab, dim)

    def forward(self, X):
        return self.lut(X) * sqrt(self.dim)


