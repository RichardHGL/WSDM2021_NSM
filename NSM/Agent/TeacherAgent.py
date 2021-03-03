import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
from NSM.Agent.BaseAgent import BaseAgent
from NSM.Model.hybrid_model import HybridModel
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class TeacherAgent_hybrid(BaseAgent):
    def __init__(self, args, logger, num_entity, num_relation, num_word):
        super(TeacherAgent_hybrid, self).__init__(args, logger, num_entity, num_relation, num_word)
        self.q_type = args['q_type']
        # model_name = args['model_name'].lower()
        # teacher_type = args['teacher_type'].lower()
        self.label_f1 = args['label_f1']
        self.model = HybridModel(args, num_entity, num_relation, num_word)

    def forward(self, batch, training=False):
        batch = self.deal_input(batch)
        return self.model(batch, training=training)

    def label_data(self, batch):
        batch = self.deal_input(batch)
        middle_dist = self.model.label_data(batch)
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        pred_dist = self.model.dist_history[-1]
        label_valid = self.model.get_label_valid(pred_dist, answer_dist, label_f1=self.label_f1)
        return middle_dist, label_valid

    def deal_input(self, batch):
        return self.deal_input_seq(batch)
