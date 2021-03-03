import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from NSM.Model.base_model import BaseModel
from NSM.Modules.Instruction.seq_instruction import LSTMInstruction
from NSM.Modules.Reasoning.gnn_backward_reasoning import GNNBackwardReasoning

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class BackwardReasonModel(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word, forward_model=None):
        """
        num_relation: number of relation including self-connection
        """
        super(BackwardReasonModel, self).__init__(args, num_entity, num_relation, num_word)
        share_embedding = args['share_embedding']
        share_instruction = args['share_instruction']
        if share_embedding:
            self.share_embedding(forward_model)
        else:
            self.embedding_def()
            self.share_module_def()
        if share_instruction:
            self.instruction = forward_model.instruction
        else:
            self.instruction_def(args)
        self.reasoning_def(args, num_entity, num_relation)
        self.loss_type = "kl"
        self.to(self.device)

    def instruction_def(self, args):
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def reasoning_def(self, args, num_entity, num_relation):
        self.reasoning = GNNBackwardReasoning(args, num_entity, num_relation)

    def share_embedding(self, model):
        self.relation_embedding = model.relation_embedding
        self.word_embedding = model.word_embedding
        self.type_layer = model.type_layer
        self.entity_linear = model.entity_linear
        self.relation_linear = model.relation_linear
        self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        # self.query_entities = query_entities
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.query_node_emb = self.instruction.query_node_emb
        self.rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=self.rel_features,
                                   query_entities=query_entities,
                                   query_node_emb=self.query_node_emb)

    def get_loss_constraint(self, forewad_dist, backward_dist):
        log_prob = torch.log(forewad_dist + 1e-8)
        loss = torch.mean(-(log_prob * backward_dist.detach()))
        return loss

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def label_data(self, batch):
        middle_dist = []
        self.model(batch, training=False)
        self.back_model(batch, training=False)
        forward_history = self.model.dist_history
        backward_history = self.back_model.dist_history
        for i in range(self.num_step - 1):
            middle_dist.append(forward_history[i + 1] + backward_history[i + 1] / 2)
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        pred_dist = self.model.dist_history[-1]
        label_valid = self.model.get_label_valid(pred_dist, answer_dist, label_f1=self.label_f1)
        return middle_dist, label_valid

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
        answer_len[answer_len == 0] = 1.0
        answer_prob = answer_dist.div(answer_len)
        dist_history, score_list = self.reasoning.forward_all(curr_dist=answer_prob,
                                                              instruction_list=self.instruction_list)
        self.dist_history = dist_history
        self.score_list= score_list
        pred_dist = dist_history[0]
        # main_loss = self.get_loss_new(pred_dist, answer_dist)
        # loss = self.get_loss_new(pred_dist, query_entities)
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=query_entities, label_valid=case_valid)
        extras = [loss.item(), 0.0, 0.0]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, query_entities)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        # pred = torch.max(pred_dist, dim=1)[1]
        return loss, extras, pred_dist, tp_list
