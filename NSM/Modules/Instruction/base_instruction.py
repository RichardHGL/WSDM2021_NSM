import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class BaseInstruction(torch.nn.Module):

    def __init__(self, args):
        super(BaseInstruction, self).__init__()
        self._parse_args(args)
        self.share_module_def()

    def _parse_args(self, args):
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        # self.share_encoder = args['share_encoder']
        self.q_type = args['q_type']
        self.num_step = args['num_step']
        self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

        self.reset_time = 0

    def share_module_def(self):
        # dropout
        self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (torch.zeros(num_layer, batch_size, hidden_size).to(self.device),
                torch.zeros(num_layer, batch_size, hidden_size).to(self.device))

    def encode_question(self, *args):
        # constituency tree or query_text
        pass

    def get_instruction(self, *args):
        # expected return : question_emb, attn_weight
        pass

    @staticmethod
    def get_node_emb(query_hidden_emb, action):
        '''

        :param query_hidden_emb: (batch_size, max_hyper, emb)
        :param action: (batch_size)
        :return: (batch_size, 1, emb)
        '''
        batch_size, max_hyper, _ = query_hidden_emb.size()
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor)
        q_rep = query_hidden_emb[row_idx, action, :]
        return q_rep.unsqueeze(1)
