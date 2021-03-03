import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
from NSM.Agent.BaseAgent import BaseAgent
from NSM.Model.nsm_model import GNNModel
from NSM.Model.backward_model import BackwardReasonModel
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class NsmAgent(BaseAgent):
    def __init__(self, args, logger, num_entity, num_relation, num_word):
        super(NsmAgent, self).__init__(args, logger, num_entity, num_relation, num_word)
        self.q_type = "seq"
        model_name = args['model_name'].lower()
        self.label_f1 = args['label_f1']
        self.model_name = model_name
        if model_name.startswith('gnn'):
            self.model = GNNModel(args, num_entity, num_relation, num_word)
        elif model_name.startswith('back'):
            self.model = BackwardReasonModel(args, num_entity, num_relation, num_word)
        else:
            raise NotImplementedError

    def forward(self, batch, training=False):
        batch = self.deal_input(batch)
        return self.model(batch, training=training)

    def label_data(self, batch):
        batch = self.deal_input(batch)
        # middle_dist = self.model.label_data(batch)
        middle_dist = []
        self.model(batch, training=False)
        forward_history = self.model.dist_history
        for i in range(self.num_step - 1):
            middle_dist.append(forward_history[i + 1])
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        if self.model_name == "back":
            pred_dist = self.model.dist_history[0]
            label_valid = self.model.get_label_valid(pred_dist, query_entities, label_f1=self.label_f1)
        else:
            pred_dist = self.model.dist_history[-1]
            label_valid = self.model.get_label_valid(pred_dist, answer_dist, label_f1=self.label_f1)
        # label_valid = None
        return middle_dist, label_valid

    def train_batch(self, batch, middle_dist, label_valid=None):
        batch = self.deal_input(batch)
        return self.model.train_batch(batch, middle_dist, label_valid)

    def deal_input(self, batch):
        return self.deal_input_seq(batch)
