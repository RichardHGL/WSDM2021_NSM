import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
from NSM.Agent.BaseAgent import BaseAgent
from NSM.Model.forward_model import ForwardReasonModel
from NSM.Model.backward_model import BackwardReasonModel
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class TeacherAgent_parallel(BaseAgent):
    def __init__(self, args, logger, num_entity, num_relation, num_word):
        super(TeacherAgent_parallel, self).__init__(args, logger, num_entity, num_relation, num_word)
        self.q_type = args['q_type']
        self.label_f1 = args['label_f1']
        self.model = ForwardReasonModel(args, num_entity, num_relation, num_word)
        self.back_model = BackwardReasonModel(args, num_entity, num_relation, num_word, self.model)
        self.lambda_back = args['lambda_back']
        self.lambda_constrain = args['lambda_constrain']
        self.constrain_type = args['constrain_type']
        self.constraint_loss = torch.nn.MSELoss(reduction='none')
        self.kld_loss_1 = nn.KLDivLoss(reduction='none')
        self.num_step = args['num_step']

    def get_js_div(self, dist_1, dist_2):
        mean_dist = (dist_1 + dist_2) / 2
        log_mean_dist = torch.log(mean_dist + 1e-8)
        # loss_kl_1 = self.kld_loss_1(log_mean_dist, dist_1)
        # loss_kl_2 = self.kld_loss_1(log_mean_dist, dist_2)
        # print(loss_kl_1.item(), loss_kl_2.item())
        loss = 0.5 * (self.kld_loss_1(log_mean_dist, dist_1) + self.kld_loss_1(log_mean_dist, dist_2))
        return loss

    def get_kl_div(self, dist_1, dist_2):
        log_dist_1 = torch.log(dist_1 + 1e-8)
        log_dist_2 = torch.log(dist_2 + 1e-8)
        loss = 0.5 * (self.kld_loss_1(log_dist_1, dist_2) + self.kld_loss_1(log_dist_2, dist_1))
        return loss

    def get_constraint_loss(self, forward_dist, backward_dist, case_valid):
        loss_constraint = None
        for i in range(self.num_step - 1):
            cur_forward_dist = forward_dist[i + 1]
            cur_backward_dist = backward_dist[i + 1]
            tp_loss = self.get_js_div(cur_forward_dist, cur_backward_dist)
            tp_loss = torch.sum(tp_loss * case_valid) / cur_forward_dist.size(0)
            if loss_constraint is None:
                loss_constraint = tp_loss
            else:
                loss_constraint += tp_loss
        return loss_constraint

    def forward(self, batch, training=False):
        batch = self.deal_input(batch)
        loss, pred, pred_dist, tp_list = self.model(batch, training=training)
        extras = [loss.item()]
        if training:
            current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
            local_entity, query_entities, true_batch_id = batch
            answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
            case_valid = (answer_number > 0).float()
            back_loss, _, _, _ = self.back_model(batch, training=training)
            forward_history = self.model.dist_history
            backward_history = self.back_model.dist_history
            constrain_loss = self.get_constraint_loss(forward_history, backward_history, case_valid)
            loss = loss + self.lambda_back * back_loss + self.lambda_constrain * constrain_loss
            extras.append(back_loss.item())
            extras.append(constrain_loss.item())
        return loss, extras, pred_dist, tp_list

    def label_data(self, batch):
        batch = self.deal_input(batch)
        # middle_dist = self.model.label_data(batch)
        middle_dist = []
        self.model(batch, training=False)
        self.back_model(batch, training=False)
        forward_history = self.model.dist_history
        backward_history = self.back_model.dist_history
        for i in range(self.num_step - 1):
            middle_dist.append((forward_history[i + 1] + backward_history[i + 1]) / 2)
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        pred_dist = self.model.dist_history[-1]
        label_valid = self.model.get_label_valid(pred_dist, answer_dist, label_f1=self.label_f1)
        return middle_dist, label_valid

    def deal_input(self, batch):
        return self.deal_input_seq(batch)
