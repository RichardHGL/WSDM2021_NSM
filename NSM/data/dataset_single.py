import json
import numpy as np
import re
from tqdm import tqdm
import torch
from collections import Counter
from NSM.data.basic_dataset import BasicDataLoader


class SingleDataLoader(BasicDataLoader):
    def __init__(self, config, word2id, relation2id, entity2id, data_type="train"):
        super(SingleDataLoader, self).__init__(config, word2id, relation2id, entity2id, data_type)

    def deal_q_type(self, q_type=None):
        sample_ids = self.sample_ids
        if q_type is None:
            q_type = self.q_type
        if q_type == "seq":
            q_input = self.query_texts[sample_ids]
        else:
            raise NotImplementedError
        return q_input

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
        ori_sample_ids = self.batches[start: end]
        true_batch_id, sample_ids, seed_dist = self.deal_multi_seed(ori_sample_ids)
        self.sample_ids = sample_ids
        self.true_sample_ids = ori_sample_ids
        self.batch_ids = true_batch_id
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