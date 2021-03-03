import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from NSM.Model.base_model import BaseModel
from NSM.Modules.Instruction.seq_instruction import LSTMInstruction
from NSM.Modules.Reasoning.gnn_reasoning import GNNReasoning
from NSM.Modules.Reasoning.gnn_backward_reasoning import GNNBackwardReasoning

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class HybridModel(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        num_relation: number of relation including self-connection
        """
        super(HybridModel, self).__init__(args, num_entity, num_relation, num_word)
        self.embedding_def()
        self.share_module_def()
        self.private_module_def(args, num_entity, num_relation)
        self.loss_type = "kl"
        self.model_name = args['model_name'].lower()
        self.lambda_back = args['lambda_back']
        self.lambda_constrain = args['lambda_constrain']
        self.constrain_type = args['constrain_type']
        self.constraint_loss = torch.nn.MSELoss(reduction='none')
        self.kld_loss_1 = nn.KLDivLoss(reduction='none')
        self.num_step = args['num_step']
        self.to(self.device)

    def reasoning_def(self, args, num_entity, num_relation):
        self.reasoning = GNNReasoning(args, num_entity, num_relation)
        self.back_reasoning = GNNBackwardReasoning(args, num_entity, num_relation)

    def instruction_def(self, args):
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def private_module_def(self, args, num_entity, num_relation):
        # initialize entity embedding
        self.instruction_def(args)
        self.reasoning_def(args, num_entity, num_relation)
        self.constraint_loss = torch.nn.MSELoss()

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
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
                                   query_node_emb=self.query_node_emb)

    def one_step(self, num_step):
        # relational_ins, attn_weight = self.instruction.get_instruction(self.relational_ins, query_mask, step=num_step)
        relational_ins = self.instruction_list[num_step]
        # attn_weight = self.attn_list[num_step]
        # self.relational_ins = relational_ins
        self.curr_dist = self.reasoning(self.curr_dist, relational_ins, step=num_step)
        self.dist_history.append(self.curr_dist)

    def get_js_div(self, dist_1, dist_2):
        mean_dist = (dist_1 + dist_2) / 2
        log_mean_dist = torch.log(mean_dist + 1e-8)
        # loss_kl_1 = self.kld_loss_1(log_mean_dist, dist_1)
        # loss_kl_2 = self.kld_loss_1(log_mean_dist, dist_2)
        # print(loss_kl_1.item(), loss_kl_2.item())
        loss = 0.5 * (self.kld_loss_1(log_mean_dist, dist_1) + self.kld_loss_1(log_mean_dist, dist_2))
        return loss

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def calc_loss_backward(self, case_valid):
        back_loss = None
        constrain_loss = None
        for i in range(self.num_step):
            forward_dist = self.dist_history[i]
            backward_dist = self.backward_history[i]
            if i == 0:
                # back_loss = self.get_loss_new(backward_dist, forward_dist)
                back_loss = self.calc_loss_label(curr_dist=backward_dist,
                                                 teacher_dist=forward_dist,
                                                 label_valid=case_valid)
                # backward last step should be similar with seed distribution
            else:
                tp_loss = self.get_js_div(forward_dist, backward_dist)
                tp_loss = torch.sum(tp_loss * case_valid) / forward_dist.size(0)
                if constrain_loss is None:
                    constrain_loss = tp_loss
                else:
                    constrain_loss += tp_loss
        return back_loss, constrain_loss

    def label_data(self, batch):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        # for i in range(self.num_step):
        #     self.one_step(num_step=i)
        self.dist_history, score_list = self.reasoning.forward_all(current_dist, self.instruction_list)
        final_emb = self.reasoning.local_entity_emb
        # # detach()?
        self.back_reasoning.init_reason(local_entity=local_entity,
                                        kb_adj_mat=kb_adj_mat,
                                        local_entity_emb=final_emb,
                                        rel_features=self.rel_features,
                                        query_entities=query_entities,
                                        query_node_emb=self.query_node_emb)
        # # if self.back_from_answer:
        answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
        answer_len[answer_len == 0] = 1.0
        answer_prob = answer_dist.div(answer_len)
        self.backward_history, back_score_list = self.back_reasoning.forward_all(answer_prob, self.instruction_list)
        middle_dist = []
        # # back_score_list[0] -> seed scores
        for i in range(self.num_step - 1):
            # forward_score = score_list[i]
            # backward_score = back_score_list[i + 1]
            # mix_score = (forward_score + backward_score) / 2
            mix_dist = (self.dist_history[i + 1] + self.backward_history[i+1]) / 2
            # mix_dist = F.softmax(mix_score, dim=1)
            # middle_dist.append(self.dist_history[i+1])
            middle_dist.append(mix_dist)
        return middle_dist

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        # for i in range(self.num_step):
        #     self.one_step(num_step=i)
        self.dist_history, score_list = self.reasoning.forward_all(current_dist, self.instruction_list)
        # loss, extras = self.calc_loss_basic(answer_dist)
        extras = []
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # loss = self.get_loss_new(pred_dist, answer_dist)
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        extras.append(loss.item())
        if training:
            final_emb = self.reasoning.local_entity_emb
            # detach()?
            self.back_reasoning.init_reason(local_entity=local_entity,
                                            kb_adj_mat=kb_adj_mat,
                                            local_entity_emb=final_emb,
                                            rel_features=self.rel_features,
                                            query_entities=query_entities,
                                            query_node_emb=self.query_node_emb)
            # if self.back_from_answer:
            answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
            answer_len[answer_len == 0] = 1.0
            answer_prob = answer_dist.div(answer_len)
            self.backward_history, back_score_list = self.back_reasoning.forward_all(answer_prob, self.instruction_list)
            # else:
            #     self.backward_history = self.back_reasoning.forward_all(self.dist_history[-1], self.instruction_list)
            # loss, extras = self.calc_loss_basic(answer_dist)
            back_loss, constrain_loss = self.calc_loss_backward(case_valid)
            extras.append(back_loss.item())
            extras.append(constrain_loss.item())
            loss = loss + self.lambda_back * back_loss + self.lambda_constrain * constrain_loss
        # pred_dist = self.dist_history[-1]
        # pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, extras, pred_dist, tp_list
