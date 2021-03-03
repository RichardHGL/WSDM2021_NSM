import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class BaseAgent(nn.Module):
    def __init__(self, args, logger, num_entity, num_relation, num_word):
        super(BaseAgent, self).__init__()
        self.parse_args(args, num_entity, num_relation, num_word)

    def parse_args(self, args, num_entity, num_relation, num_word):
        self.args = args
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        print("Entity: {}, Relation: {}, Word: {}".format(num_entity, num_relation, num_word))
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.learning_rate = self.args['lr']
        self.q_type = args['q_type']
        self.num_step = args['num_step']
        # self.lambda_label = args['lambda_label']

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

        self.reset_time = 0

    @staticmethod
    def get_node_emb(query_hidden_emb, action):
        '''

        :param query_hidden_emb: (batch_size, max_hyper, emb)
        :param action: (batch_size)
        :return: (batch_size, emb)
        '''
        batch_size, max_hyper, _ = query_hidden_emb.size()
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor)
        q_rep = query_hidden_emb[row_idx, action, :]
        return q_rep

    def deal_input_seq(self, batch):
        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        query_text = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        query_mask = (query_text != self.num_word).float()

        return current_dist, query_text, query_mask, kb_adj_mat, answer_dist, \
               local_entity, query_entities, true_batch_id

    def forward(self, *args):
        pass

    @staticmethod
    def mask_max(values, mask, keepdim=True):
        return torch.max(values + (1 - mask) * VERY_NEG_NUMBER, dim=-1, keepdim=keepdim)[0]

    @staticmethod
    def mask_argmax(values, mask):
        return torch.argmax(values + (1 - mask) * VERY_NEG_NUMBER, dim=-1, keepdim=True)
