import json
import numpy as np
import re
from tqdm import tqdm
import torch
from NSM.data.read_tree import read_tree
from collections import Counter


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id


class BasicDataLoader(object):
    def __init__(self, config, word2id, relation2id, entity2id, data_type="train"):
        self._parse_args(config, word2id, relation2id, entity2id)
        self._load_file(config, data_type)
        self._load_data()

    def _load_file(self, config, data_type="train"):
        data_file = config['data_folder'] + data_type + "_simple.json"
        dep_file = config['data_folder'] + data_type + ".dep"
        print('loading data from', data_file)
        self.data = []
        self.dep = []
        skip_index = set()
        index = 0
        with open(data_file) as f_in:
            for line in tqdm(f_in):
                index += 1
                line = json.loads(line)
                if len(line['entities']) == 0:
                    skip_index.add(index)
                    continue
                self.data.append(line)
                self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
        print("skip", skip_index)
        index = 0
        with open(dep_file) as f_in:
            for line in f_in:
                index += 1
                if index in skip_index:
                    continue
                line = json.loads(line)
                self.dep.append(line)
        print('max_facts: ', self.max_facts)
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

    def _load_data(self):
        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()

        if self.use_self_loop:
            self.max_facts = self.max_facts + self.max_local_entity

        self.question_id = []
        self.candidate_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.q_adj_mats = np.empty(self.num_data, dtype=object)
        # self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.seed_list = np.empty(self.num_data, dtype=object)
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        # self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_lists = np.empty(self.num_data, dtype=object)

        if self.q_type == "con":
            print('preparing con ...')
            self._prepare_con()
        else:
            print('preparing dep ...')
            self._prepare_dep()
        print('preparing data ...')
        self._prepare_data()

    def _parse_args(self, config, word2id, relation2id, entity2id):
        self.use_inverse_relation = config['use_inverse_relation']
        self.use_self_loop = config['use_self_loop']
        self.num_step = config['num_step']
        self.max_local_entity = 0
        self.max_relevant_doc = 0
        self.max_facts = 0

        print('building word index ...')
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.id2entity = {i: entity for entity, i in entity2id.items()}
        self.q_type = config['q_type']

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
        print("Entity: {}, Relation in KB: {}, Relation in use: {} ".format(len(entity2id),
                                                                            len(self.relation2id),
                                                                            self.num_kb_relation))

    @staticmethod
    def tokenize_sent(question_text):
        question_text = question_text.strip().lower()
        question_text = re.sub('\'s', ' s', question_text)
        words = []
        for w_idx, w in enumerate(question_text.split(' ')):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w == '':
                continue
            words += [w]
        return words


    def get_quest(self, training=False):
        q_list = []
        # if training:
        #     sample_ids = self.sample_ids
        # else:
        #     sample_ids = self.true_sample_ids
        sample_ids = self.sample_ids
        for sample_id in sample_ids:
            tp_str = self.decode_text(self.query_texts[sample_id, :])
            # id2word = self.id2word
            # for i in range(self.max_query_word):
            #     if self.query_texts[sample_id, i] in id2word:
            #         tp_str += id2word[self.query_texts[sample_id, i]] + " "
            q_list.append(tp_str)
        return q_list

    def decode_text(self, np_array_x):
        id2word = self.id2word
        tp_str = ""
        for i in range(self.max_query_word):
            if np_array_x[i] in id2word:
                tp_str += id2word[np_array_x[i]] + " "
        return tp_str

    def _prepare_dep(self):
        max_count = 0
        for line in self.dep:
            word_list = line["dep"]
            max_count = max(max_count, len(word_list))
        self.max_query_word = max_count
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        next_id = 0
        # self.layer2node = []
        self.node2layer = []
        self.dep_parents = []
        self.dep_relations = []
        for sample in tqdm(self.dep):
            tp_dep = sample["dep"]
            node_layer, parents, relations = read_tree(tp_dep)
            # self.layer2node.append(layer2node)
            self.node2layer.append(node_layer)
            self.dep_parents.append(parents)
            self.dep_relations.append(relations)
            tokens = [item[0] for item in tp_dep]
            for j, word in enumerate(tokens):
                # if j < self.max_query_word:
                if word in self.word2id:
                    self.query_texts[next_id, j] = self.word2id[word]
                else:
                    self.query_texts[next_id, j] = len(self.word2id)
            # head_list = []
            # tail_list = []
            # for i in range(len(parents)):
            #     if parents[i] == -1:
            #         continue
            #     head_list.append(i)
            #     tail_list.append(parents[i])
            # self.q_adj_mats[next_id] = (np.array(head_list, dtype=int), np.array(tail_list, dtype=int))
            next_id += 1

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        num_query_entity = {}
        for sample in tqdm(self.data):
            self.question_id.append(sample["id"])
            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            if len(g2l) == 0:
                print(next_id)
                continue
            # build connection between question and entities in it
            tp_set = set()
            seed_list = []
            for j, entity in enumerate(sample['entities']):
                # if entity['text'] not in self.entity2id:
                #     continue
                global_entity = entity  # self.entity2id[entity['text']]
                if global_entity not in g2l:
                    continue
                local_ent = g2l[global_entity]
                self.query_entities[next_id, local_ent] = 1.0
                seed_list.append(local_ent)
                tp_set.add(local_ent)
            self.seed_list[next_id] = seed_list
            num_query_entity[next_id] = len(tp_set)
            for global_entity, local_entity in g2l.items():
                if local_entity not in tp_set:  # skip entities in question
                    self.candidate_entities[next_id, local_entity] = global_entity
                # if local_entity != 0:  # skip question node
                #     self.candidate_entities[next_id, local_entity] = global_entity

            # relations in local KB
            head_list = []
            rel_list = []
            tail_list = []
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                # head = g2l[self.entity2id[sbj['text']]]
                # rel = self.relation2id[rel['text']]
                # tail = g2l[self.entity2id[obj['text']]]
                head = g2l[sbj]
                rel = int(rel)
                tail = g2l[obj]
                head_list.append(head)
                rel_list.append(rel)
                tail_list.append(tail)
                if self.use_inverse_relation:
                    head_list.append(tail)
                    rel_list.append(rel + len(self.relation2id))
                    tail_list.append(head)
            if len(tp_set) > 0:
                for local_ent in tp_set:
                    self.seed_distribution[next_id, local_ent] = 1.0 / len(tp_set)
            else:
                for index in range(len(g2l)):
                    self.seed_distribution[next_id, index] = 1.0 / len(g2l)
            try:
                assert np.sum(self.seed_distribution[next_id]) > 0.0
            except:
                print(next_id, len(tp_set))
                exit(-1)

            # tokenize question
            # tokens = self.tokenize_sent(sample['question'])
            # tokens = sample['question'].split()
            # for j, word in enumerate(tokens):
            #     # if j < self.max_query_word:
            #     if word in self.word2id:
            #         self.query_texts[next_id, j] = self.word2id[word]
            #     else:
            #         self.query_texts[next_id, j] = len(self.word2id)# self.word2id['__unk__']

            # construct distribution for answers
            answer_list = []
            for answer in sample['answers']:
                keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                answer_ent = self.entity2id[answer[keyword]]
                answer_list.append(answer_ent)
                if answer_ent in g2l:
                    self.answer_dists[next_id, g2l[answer_ent]] = 1.0
            self.answer_lists[next_id] = answer_list
            self.kb_adj_mats[next_id] = (np.array(head_list, dtype=int),
                                         np.array(rel_list, dtype=int),
                                         np.array(tail_list, dtype=int))

            next_id += 1
        num_no_query_ent = 0
        num_one_query_ent = 0
        num_multiple_ent = 0
        for i in range(next_id):
            ct = num_query_entity[i]
            if ct == 1:
                num_one_query_ent += 1
            elif ct == 0:
                num_no_query_ent += 1
            else:
                num_multiple_ent += 1
        print("{} cases in total, {} cases without query entity, {} cases with single query entity,"
              " {} cases with multiple query entities".format(next_id, num_no_query_ent,
                                                              num_one_query_ent, num_multiple_ent))

    def _build_query_graph_new(self, sample_ids):
        word_ids = np.array([], dtype=int)
        layer_heads = {}
        layer_tails = {}
        layer_map = {}
        root_pos = []
        for i, sample_id in enumerate(sample_ids):
            word_ids = np.append(word_ids, self.query_texts[sample_id, :])
            index_bias = i * self.max_query_word
            node_layer = self.node2layer[sample_id]
            parents = self.dep_parents[sample_id]
            for j, par in enumerate(parents):
                if par == -1:   # root node, par = -1, layer = 1
                    root_pos.append(index_bias + j)
                    continue
                cur_layer = node_layer[j]
                node_now = j + index_bias
                parent_node = par + index_bias
                layer_heads.setdefault(cur_layer - 1, [])
                layer_tails.setdefault(cur_layer - 1, [])
                layer_map.setdefault(cur_layer, {})
                layer_map.setdefault(cur_layer - 1, {})
                if node_now not in layer_map[cur_layer]:
                    layer_map[cur_layer][node_now] = len(layer_map[cur_layer])
                if parent_node not in layer_map[cur_layer - 1]:
                    layer_map[cur_layer - 1][parent_node] = len(layer_map[cur_layer - 1])
                layer_heads[cur_layer - 1].append(layer_map[cur_layer][node_now])
                layer_tails[cur_layer - 1].append(layer_map[cur_layer - 1][parent_node])
                if j not in parents:
                    # if node is leave node, add zero node from previous layer
                    layer_heads.setdefault(cur_layer, [])
                    layer_tails.setdefault(cur_layer, [])
                    layer_heads[cur_layer].append(0)
                    layer_tails[cur_layer].append(layer_map[cur_layer][node_now])
        max_layer = max(list(layer_heads.keys()))
        # organize data layer-wise
        edge_lists = []
        number_node_total = 1  # initial node zero vector
        word_order = [0] * (len(sample_ids) * self.max_query_word)
        for layer in range(max_layer, 0, -1):
            # 1 ~ max_layer
            num_node = len(layer_map[layer])
            id2node = {v: k for k, v in layer_map[layer].items()}
            layer_entities = []
            for id in range(num_node):
                batch_node_idx = id2node[id]
                layer_entities.append(word_ids[batch_node_idx])
                word_order[batch_node_idx] = id + number_node_total
            tp_heads = []
            for node in layer_heads[layer]:
                if node == 0:   # Further check, there may be bug
                    tp_heads.append(0)
                else:
                    # zero index for leaf node in degree
                    tp_heads.append(node + 1)
            tp_heads = np.array(tp_heads)
            number_node_total += num_node
            fact_ids = np.array(range(len(layer_heads[layer])), dtype=int)
            tp_tails = np.array(layer_tails[layer])  # + number_node_total
            edge_list = (tp_heads, None, tp_tails, fact_ids)
            edge_lists.append((edge_list, layer_entities))
        root_order = [word_order[item] for item in root_pos]
        return edge_lists, word_order, root_order

    def _build_fact_mat(self, sample_ids, fact_dropout):
        batch_heads = np.array([], dtype=int)
        batch_rels = np.array([], dtype=int)
        batch_tails = np.array([], dtype=int)
        batch_ids = np.array([], dtype=int)
        for i, sample_id in enumerate(sample_ids):
            index_bias = i * self.max_local_entity
            head_list, rel_list, tail_list = self.kb_adj_mats[sample_id]
            num_fact = len(head_list)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[: num_keep_fact]

            real_head_list = head_list[mask_index] + index_bias
            real_tail_list = tail_list[mask_index] + index_bias
            real_rel_list = rel_list[mask_index]
            batch_heads = np.append(batch_heads, real_head_list)
            batch_rels = np.append(batch_rels, real_rel_list)
            batch_tails = np.append(batch_tails, real_tail_list)
            batch_ids = np.append(batch_ids, np.full(len(mask_index), i, dtype=int))
            if self.use_self_loop:
                num_ent_now = len(self.global2local_entity_maps[sample_id])
                ent_array = np.array(range(num_ent_now), dtype=int) + index_bias
                rel_array = np.array([self.num_kb_relation - 1] * num_ent_now, dtype=int)
                batch_heads = np.append(batch_heads, ent_array)
                batch_tails = np.append(batch_tails, ent_array)
                batch_rels = np.append(batch_rels, rel_array)
                batch_ids = np.append(batch_ids, np.full(num_ent_now, i, dtype=int))
        fact_ids = np.array(range(len(batch_heads)), dtype=int)
        head_count = Counter(batch_heads)
        # tail_count = Counter(batch_tails)
        weight_list = [1.0 / head_count[head] for head in batch_heads]
        # entity2fact_index = torch.LongTensor([batch_heads, fact_ids])
        # entity2fact_val = torch.FloatTensor(weight_list)
        # entity2fact_mat = torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
        #     [len(sample_ids) * self.max_local_entity, len(batch_heads)]))
        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list

    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        print('max local entity: ', self.max_local_entity)
        return global2local_entity_maps

    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        for entity_global_id in entities:
            # entity_text = entity['text']
            # if entity_text not in entity2id:
            #     continue
            # entity_global_id = entity2id[entity_text]
            # print(entity_global_id)
            # print(entity_global_id)
            if entity_global_id not in g2l:
                g2l[entity_global_id] = len(g2l)

    def deal_q_type(self, q_type=None):
        sample_ids = self.sample_ids
        if q_type is None:
            q_type = self.q_type
        if q_type == "seq":
            q_input = self.query_texts[sample_ids]
        else:
            raise NotImplementedError
        return q_input
