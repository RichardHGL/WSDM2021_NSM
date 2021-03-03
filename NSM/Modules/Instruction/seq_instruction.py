import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.Instruction.base_instruction import BaseInstruction
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class LSTMInstruction(BaseInstruction):

    def __init__(self, args, word_embedding, num_word):
        super(LSTMInstruction, self).__init__(args)
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.encoder_def()
        entity_dim = self.entity_dim
        self.cq_linear = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

    def encoder_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim,
                                    batch_first=True, bidirectional=False)

    def encode_question(self, query_text):
        batch_size = query_text.size(0)
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (h_n, c_n) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                         self.init_hidden(1, batch_size,
                                                                          self.entity_dim))  # 1, batch_size, entity_dim
        self.instruction_hidden = h_n
        self.instruction_mem = c_n
        self.query_node_emb = h_n.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        self.query_hidden_emb = query_hidden_emb
        self.query_mask = (query_text != self.num_word).float()
        return query_hidden_emb, self.query_node_emb

    def init_reason(self, query_text):
        batch_size = query_text.size(0)
        self.encode_question(query_text)
        self.relational_ins = torch.zeros(batch_size, self.entity_dim).to(self.device)
        self.instructions = []
        self.attn_list = []

    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        query_hidden_emb = self.query_hidden_emb
        query_mask = self.query_mask
        if query_node_emb is None:
            query_node_emb = self.query_node_emb
        relational_ins = relational_ins.unsqueeze(1)
        question_linear = getattr(self, 'question_linear' + str(step))
        q_i = question_linear(self.linear_drop(query_node_emb))
        cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i), dim=-1)))
        # batch_size, 1, entity_dim
        ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb))
        # batch_size, max_local_entity, 1
        # cv = self.softmax_d1(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER)
        attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
        # batch_size, max_local_entity, 1
        relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1)
        return relational_ins, attn_weight

    def forward(self, query_text):
        self.init_reason(query_text)
        for i in range(self.num_step):
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins
        return self.instructions, self.attn_list

    # def __repr__(self):
    #     return "LSTM + token-level attention"
