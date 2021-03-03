import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.layer_nsm import TypeLayer
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class BaseReasoning(torch.nn.Module):

    def __init__(self, args, num_entity, num_relation):
        super(BaseReasoning, self).__init__()
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.num_step = args['num_step']
        self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        self.reason_kb = args['reason_kb']

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

    def build_matrix(self):
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = self.edge_list
        num_fact = len(fact_ids)
        num_relation = self.num_relation
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        self.num_fact = num_fact
        fact2head = torch.LongTensor([batch_heads, fact_ids]).to(self.device)
        fact2tail = torch.LongTensor([batch_tails, fact_ids]).to(self.device)
        head2fact = torch.LongTensor([fact_ids, batch_heads]).to(self.device)
        tail2fact = torch.LongTensor([fact_ids, batch_tails]).to(self.device)
        rel2fact = torch.LongTensor([batch_rels + batch_ids * num_relation, fact_ids]).to(self.device)
        self.batch_rels = torch.LongTensor(batch_rels).to(self.device)
        self.batch_ids = torch.LongTensor(batch_ids).to(self.device)
        self.batch_heads = torch.LongTensor(batch_heads).to(self.device)
        self.batch_tails = torch.LongTensor(batch_tails).to(self.device)
        # self.batch_ids = batch_ids
        val_one = torch.ones_like(self.batch_ids).float().to(self.device)

        # Sparse Matrix for reason on graph
        self.fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))
        self.head2fact_mat = self._build_sparse_tensor(head2fact, val_one, (num_fact, batch_size * max_local_entity))
        self.fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        self.tail2fact_mat = self._build_sparse_tensor(tail2fact, val_one, (num_fact, batch_size * max_local_entity))
        self.rel2fact_mat = self._build_sparse_tensor(rel2fact, val_one, (batch_size * num_relation, num_fact))

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to(self.device)

