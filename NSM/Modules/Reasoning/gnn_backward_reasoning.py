import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.Reasoning.base_reasoning import BaseReasoning
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class GNNBackwardReasoning(BaseReasoning):

    def __init__(self, args, num_entity, num_relation):
        super(GNNBackwardReasoning, self).__init__(args, num_entity, num_relation)
        self.share_module_def()
        self.private_module_def()

    def private_module_def(self):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2 * entity_dim, out_features=entity_dim))
            # self.add_module('score_func' + str(i), nn.Linear(in_features=entity_dim, out_features=1))

    def reason_layer(self, curr_dist, instruction, rel_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        # num_relation = self.num_relation
        rel_features = self.rel_features
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        # fact_val = F.relu(self.kb_self_linear(fact_rel) + self.kb_head_linear(self.linear_drop(fact_ent)))
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))

        possible_head = torch.sparse.mm(self.fact2head_mat, fact_prior)
        # (batch_size *max_local_entity, num_fact) (num_fact, 1)
        possible_head = (possible_head > VERY_SMALL_NUMBER).float().view(batch_size, max_local_entity)

        fact_val = fact_val * fact_prior
        # neighbor_rep = torch.sparse.mm(fact2tail_mat, self.kb_tail_linear(self.linear_drop(fact_val)))
        f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()

        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        return neighbor_rep, possible_head

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, query_entities, query_node_emb):
        batch_size, max_local_entity = local_entity.size()
        self.query_entities = query_entities
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.local_entity_mask = ((self.local_entity_mask + self.query_entities) > 0).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.build_matrix()

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        score_func = self.score_func
        relational_ins = relational_ins.squeeze(1)
        neighbor_rep, possible_head = self.reason_layer(current_dist, relational_ins, rel_linear)
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_rep), dim=2)
        self.local_entity_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))

        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        if self.reason_kb:
            answer_mask = possible_head * self.local_entity_mask
            score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        else:
            answer_mask = self.local_entity_mask
            score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return score_tp, current_dist
        return current_dist

    def forward_all(self, curr_dist, instruction_list):
        dist_history = [curr_dist]
        score_list = []
        for i in range(self.num_step):
            cur_step = self.num_step - 1 - i
            score_tp, curr_dist = self.forward(curr_dist, instruction_list[cur_step], step=cur_step, return_score=True)
            # score_list.append(score_tp)
            # dist_history.append(curr_dist)
            dist_history.insert(0, curr_dist)
            score_list.insert(0, score_tp)
        return dist_history, score_list

    # def __repr__(self):
    #     return "GNN based reasoning"
