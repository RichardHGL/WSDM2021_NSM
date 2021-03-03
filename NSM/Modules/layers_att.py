import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

VERY_NEG_NUMBER = -100000000000


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)  # 2H -> H
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs, query_mask):
        '''
        :param hidden: (B, 1, d)
        :param encoder_outputs: (B, ML_Q, d)
        :param query_mask: (B, ML_Q, 1)
        :return: attn_weight (B, ML_Q, 1)
        '''
        batch_size = hidden.size(0)
        max_len = encoder_outputs.size(1)
        H = hidden.expand(batch_size, max_len, self.hidden_size)
        att_energies = torch.tanh(self.score(H, encoder_outputs)) + (1 - query_mask) * VERY_NEG_NUMBER
        return F.softmax(att_energies, dim=1)

    def score(self, hidden, encoder_outputs):
        batch_size, max_len, hidden_size = encoder_outputs.size()
        energy = self.attn(torch.cat([hidden, encoder_outputs], -1))  # [B, ML_Q, 2H]->[B, ML_Q, H]
        energy = energy.view(-1, self.hidden_size)  # [B*ML_Q,H]
        v = self.v.unsqueeze(1)  # [H,1]
        energy = energy.mm(v)  # [B*ML_Q,H] x [H,1] -> [T*N*B,1]
        att_energies = energy.view(batch_size, max_len, 1)  # [T,N,B]
        return att_energies
