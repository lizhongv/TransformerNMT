import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import config


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # FIXME 需要加入GPU中吗，下面register_buffer时会自动加入？
        pe = torch.zeros(max_len, d_model).cuda()  # 加入GPU
        position = torch.arange(0., max_len).cuda().unsqueeze(1)  # 加入GPU
        div_term = torch.exp(torch.arange(0., d_model, 2).cuda()  # 加入GPU
                             * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
