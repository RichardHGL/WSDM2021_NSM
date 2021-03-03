import json
import numpy as np
import re
import os
from tqdm import tqdm
import torch
from collections import Counter
from NSM.data.basic_dataset import BasicDataLoader


class SingleDataLoader(BasicDataLoader):
    def __init__(self, config, word2id, relation2id, entity2id, data_type="train"):
        super(SingleDataLoader, self).__init__(config, word2id, relation2id, entity2id, data_type)
        self.use_label = config['use_label']
        self.label_f1 = config['label_f1']
        if data_type == "train" and self.use_label:
            label_file = os.path.join(config['checkpoint_dir'], config['label_file'])
            self.load_label(label_file)

    def _build_graph(self, tp_graph):
        head_list, rel_list, tail_list = tp_graph
        length = len(head_list)
        out_degree = {}
        in_degree = {}
        for i in range(length):
            head = head_list[i]
            rel = rel_list[i]
            tail = tail_list[i]
            out_degree.setdefault(head, {})
            out_degree[head].setdefault(rel, set())
            out_degree[head][rel].add(tail)
            in_degree.setdefault(tail, {})
            in_degree[tail].setdefault(rel, set())
            in_degree[tail][rel].add(head)
        return in_degree, out_degree

    def backward_step(self, possible_heads, cur_action, target_tail, in_degree):
        '''
        input: graph_edge, cur answers, cur relation
        output: edges used, possible heads
        '''
        tp_list = []
        available_heads = set()
        flag = False
        if self.use_self_loop and cur_action == self.num_kb_relation - 1:
            for ent in target_tail:
                tp_list.append((ent, self.num_kb_relation - 1, ent))
            available_heads |= target_tail
            # print("self-loop")
        else:
            # print("non self-loop")
            # print(target_tail)
            for ent in target_tail:
                # print("have target")
                if ent in in_degree and cur_action in in_degree[ent]:
                    # print("enter case")
                    legal_set = in_degree[ent][cur_action] & possible_heads
                    for legal_head in legal_set:
                        tp_list.append((legal_head, cur_action, ent))
                        available_heads.add(legal_head)
                else:
                    flag = True
                    print("debug")
                    print(ent in in_degree)
                    if ent in in_degree:
                        print(cur_action in in_degree[ent])
        return available_heads, tp_list, flag

    def forward_step(self, hop_edge_list, tp_weight_dict):
        new_weight_dict = {}
        if len(hop_edge_list) == 0:
            return new_weight_dict
        # tp_weight_dict = hop_weight_dict[step]
        out_degree = {}
        for head, rel, tail in hop_edge_list:
            if head in tp_weight_dict:
                out_degree.setdefault(head, 0.0)
                out_degree[head] += 1.0
        for head, rel, tail in hop_edge_list:
            if head in tp_weight_dict:
                edge_weight = tp_weight_dict[head] / out_degree[head]
                new_weight_dict.setdefault(tail, 0.0)
                new_weight_dict[tail] += edge_weight
        return new_weight_dict

    def multi_hop_trace(self, tp_obj, acc_reason_answers, in_degree, seed_ent=0):
        hop_dict = {}
        tp_key = "seed_%d" % (seed_ent)
        pred_entities = set(tp_obj[tp_key][str(self.num_step - 1)]["answer"])
        common = pred_entities & acc_reason_answers
        hop_edges = {}
        if len(common) == 0:
            for step in range(self.num_step):
                hop_edges[step] = []
            return hop_edges, True
        action_list = []
        order_list = reversed(range(self.num_step))
        target_tail = acc_reason_answers
        # hop_dict[self.num_step] = target_tail
        exist_flag = False
        for step in order_list:
            # if step == self.num_step - 1:
            cur_action = int(tp_obj[tp_key][str(step)]["action"])
            action_list.append(cur_action)
            if step > 0:
                possible_heads = set(tp_obj[tp_key][str(step - 1)]["answer"])
            else:
                possible_heads = set([seed_ent])
            # print("step", step, possible_heads, cur_action)
            target_tail, tp_triple_list, flag = self.backward_step(possible_heads, cur_action, target_tail, in_degree)
            if flag or exist_flag:
                exist_flag = True
                # print(target_tail, tp_triple_list)
                # hop_dict[step] = target_tail
            # print(target_tail, tp_triple_list)
            # hop_dict[step] = target_tail
            hop_edges[step] = tp_triple_list
        # print(hop_edges)
        return hop_edges, exist_flag

    def load_label(self, label_file):
        if not self.use_label:
            return None
        if self.num_step == 1:
            return None
        label_dist = np.zeros((self.num_data, self.num_step, self.max_local_entity), dtype=float)
        label_valid = np.zeros((self.num_data, 1), dtype=float)
        index = 0
        num_labelled_case = 0
        with open(label_file) as f_in:
            for line in f_in:
                tp_obj = json.loads(line)
                hit = tp_obj['hit']
                f1 = tp_obj['f1']
                tp_seed_list = self.seed_list[index]
                tp_edge_list = self.kb_adj_mats[index]
                in_degree, out_degree = self._build_graph(tp_edge_list)
                real_answer_list = []
                g2l = self.global2local_entity_maps[index]
                for global_ent in self.answer_lists[index]:
                    if global_ent in g2l:
                        real_answer_list.append(g2l[global_ent])
                accurate_answer_set = set(real_answer_list)
                merge_result = tp_obj["merge_pred"]
                acc_reason_answers = set(merge_result) & accurate_answer_set
                num_seed = len(tp_seed_list)
                if hit > 0 and f1 >= self.label_f1:
                    label_valid[index, 0] = 1.0
                    num_labelled_case += 1
                    # good case, we will label it with care
                    label_flag = False
                    for seed_ent in tp_seed_list:
                        hop_edges, flag = self.multi_hop_trace(tp_obj, acc_reason_answers, in_degree, seed_ent=seed_ent)
                        tp_weight_dict = {seed_ent: 1.0 / len(tp_seed_list)}
                        if not flag:
                            label_flag = True
                        for i in range(self.num_step):
                            hop_edge_list = hop_edges[i]
                            curr_weight_dict = self.forward_step(hop_edge_list, tp_weight_dict)
                            for local_ent in curr_weight_dict:
                                label_dist[index, i, local_ent] += curr_weight_dict[local_ent]
                            tp_weight_dict = curr_weight_dict
                    if not label_flag:
                        print(index, "can't label")
                        num_labelled_case -= 1
                        # print(line.strip())
                        label_valid[index, 0] = 0.0
                        for i in range(self.num_step):
                            ent_ct = {}
                            for seed_ent in tp_seed_list:
                                tp_key = "seed_%d" % (seed_ent)
                                tp_answer_list = tp_obj[tp_key][str(i)]["answer"]
                                for local_ent in tp_answer_list:
                                    ent_ct.setdefault(local_ent, 0.0)
                                    ent_ct[local_ent] += 1.0 / len(tp_answer_list)
                            # for more detailed labeling, we can deduce it from final aggregated results
                            for local_ent in ent_ct:
                                label_dist[index, i, local_ent] = ent_ct[local_ent] / num_seed
                                # dist sum 1.0
                else:
                    # bad case, we will label it simple, because we don't use it
                    label_valid[index, 0] = 0.0
                    for i in range(self.num_step):
                        ent_ct = {}
                        for seed_ent in tp_seed_list:
                            tp_key = "seed_%d" % (seed_ent)
                            tp_answer_list = tp_obj[tp_key][str(i)]["answer"]
                            for local_ent in tp_answer_list:
                                ent_ct.setdefault(local_ent, 0.0)
                                ent_ct[local_ent] += 1.0 / len(tp_answer_list)
                        # for more detailed labeling, we can deduce it from final aggregated results
                        for local_ent in ent_ct:
                            label_dist[index, i, local_ent] = ent_ct[local_ent] / num_seed
                            # dist sum 1.0
                index += 1
        assert index == self.num_data
        self.label_dist = label_dist
        self.label_valid = label_valid
        print('--------------------------------')
        print("{} cases among {} cases can be labelled".format(num_labelled_case, self.num_data))
        print('--------------------------------')

    def get_label(self):
        if not self.use_label or self.num_step == 1:
            return None, None
        label_valid = self.label_valid[self.sample_ids]
        # print(label_valid)
        labeL_dist_list = []
        for i in range(self.num_step):
            label_dist = self.label_dist[self.sample_ids, i]
            labeL_dist_list.append(label_dist)
        return labeL_dist_list, label_valid

    def deal_multi_seed(self, sample_ids):
        true_sample_ids = []
        tp_seed_list = self.seed_list[sample_ids]
        true_batch_id = []
        true_seed_ids = []
        # multi_seed_maks = []
        for i, seed_list in enumerate(tp_seed_list):
            true_batch_id.append([])
            for seed_ent in seed_list:
                true_batch_id[i].append(len(true_sample_ids))
                true_sample_ids.append(sample_ids[i])
                true_seed_ids.append(seed_ent)
                # if len(seed_list) > 1:
                #     multi_seed_maks.append(1.0)
                # else:
                #     multi_seed_maks.append(0.0)
        # print(tp_seed_list)
        # print(true_sample_ids, len(true_sample_ids))
        seed_dist = np.zeros((len(true_sample_ids), self.max_local_entity), dtype=float)
        for j, local_ent in enumerate(true_seed_ids):
            seed_dist[j, local_ent] = 1.0
            # single seed entity
        return true_batch_id, true_sample_ids, seed_dist

    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        # true_batch_id, sample_ids, seed_dist = self.deal_multi_seed(ori_sample_ids)
        # self.sample_ids = sample_ids
        # self.true_sample_ids = ori_sample_ids
        # self.batch_ids = true_batch_id
        true_batch_id = None
        seed_dist = self.seed_distribution[sample_ids]
        q_input = self.deal_q_type(q_type)
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   (self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)), \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids],\

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               (self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)), \
               q_input, \
               seed_dist, \
               true_batch_id, \
               self.answer_dists[sample_ids]