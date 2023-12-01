import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        self.WiseFFN = nn.Sequential(
            nn.Linear(ffn_num_input, ffn_num_hiddens),
            nn.ReLU(),
            nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        )

    def forward(self, X):
        # return self.dense2(self.relu(self.dense1(X)))
        return self.WiseFFN(X)


ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
res = ffn(torch.ones((2,3,4)))[0]
print(res)
