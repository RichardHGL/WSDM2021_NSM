import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.layer_nsm import TypeLayer
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


def f1_and_hits_new(answers, candidate2prob, eps=0.5):
    retrieved = []
    correct = 0
    cand_list = sorted(candidate2prob, key=lambda x:x[1], reverse=True)
    if len(cand_list) == 0:
        best_ans = -1
    else:
        best_ans = cand_list[0][0]
    # max_prob = cand_list[0][1]
    tp_prob = 0.0
    for c, prob in cand_list:
        retrieved.append((c, prob))
        tp_prob += prob
        if c in answers:
            correct += 1
        if tp_prob > eps:
            break
    if len(answers) == 0:
        if len(retrieved) == 0:
            return 1.0, 1.0, 1.0, 1.0  # precision, recall, f1, hits
        else:
            return 0.0, 1.0, 0.0, 1.0  # precision, recall, f1, hits
    else:
        hits = float(best_ans in answers)
        if len(retrieved) == 0:
            return 1.0, 0.0, 0.0, hits  # precision, recall, f1, hits
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return p, r, f1, hits


class BaseModel(torch.nn.Module):

    def __init__(self, args, num_entity, num_relation, num_word):
        super(BaseModel, self).__init__()
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self._parse_args(args)
        self.embedding_def()
        self.share_module_def()
        self.model_name = args['model_name'].lower()
        print("Entity: {}, Relation: {}, Word: {}".format(num_entity, num_relation, num_word))

    def _parse_args(self, args):
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.has_entity_kge = False
        self.has_relation_kge = False
        # self.share_encoder = args['share_encoder']
        # self.use_gnn = args['use_gnn']
        self.q_type = args['q_type']
        # self.RL_decay = args['RL_decay']
        self.num_layer = args['num_layer']
        self.num_step = args['num_step']
        self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        self.encode_type = args["encode_type"]
        self.reason_kb = args['reason_kb']
        self.eps = args['eps']

        self.loss_type = args['loss_type']
        # self.lambda_label = args['lambda_label']
        self.label_f1 = args['label_f1']
        self.entropy_weight = args['entropy_weight']

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

        self.reset_time = 0

    def share_module_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim

        if self.has_entity_kge:
            self.entity_linear = nn.Linear(in_features=kg_dim + kge_dim, out_features=entity_dim)
        else:
            self.entity_linear = nn.Linear(in_features=kg_dim, out_features=entity_dim)

        if self.has_relation_kge:
            self.relation_linear = nn.Linear(in_features=2 * kg_dim + kge_dim, out_features=entity_dim)
        else:
            self.relation_linear = nn.Linear(in_features=2 * kg_dim, out_features=entity_dim)

        # dropout
        self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        # batch_size, max_local_entity = local_entity.size()
        # hidden_size = self.entity_dim
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            if self.has_entity_kge:
                local_entity_emb = torch.cat((local_entity_emb, self.entity_kge(local_entity)),
                                             dim=2)  # batch_size, max_local_entity, word_dim + kge_dim
            if self.word_dim != self.entity_dim:
                local_entity_emb = self.entity_linear(local_entity_emb)  # batch_size, max_local_entity, entity_dim
        return local_entity_emb

    def embedding_def(self):
        word_dim = self.word_dim
        kge_dim = self.kge_dim
        kg_dim = self.kg_dim
        num_entity = self.num_entity
        num_relation = self.num_relation
        num_word = self.num_word

        if not self.encode_type:
            self.entity_embedding = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kg_dim,
                                                 padding_idx=num_entity)
            if self.entity_emb_file is not None:
                self.entity_embedding.weight = nn.Parameter(
                    torch.from_numpy(np.pad(np.load(self.entity_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
                self.entity_embedding.weight.requires_grad = False

            if self.entity_kge_file is not None:
                self.has_entity_kge = True
                self.entity_kge = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kge_dim,
                                               padding_idx=num_entity)
                self.entity_kge.weight = nn.Parameter(
                    torch.from_numpy(np.pad(np.load(self.entity_kge_file), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
                self.entity_kge.weight.requires_grad = False
            else:
                self.entity_kge = None

        # initialize relation embedding
        self.relation_embedding = nn.Embedding(num_embeddings=num_relation, embedding_dim=2 * kg_dim)
        if self.relation_emb_file is not None:
            np_tensor = self.load_relation_file(self.relation_emb_file)
            self.relation_embedding.weight = nn.Parameter(torch.from_numpy(np_tensor).type('torch.FloatTensor'))
        if self.relation_kge_file is not None:
            self.has_relation_kge = True
            self.relation_kge = nn.Embedding(num_embeddings=num_relation, embedding_dim=kge_dim)
            np_tensor = self.load_relation_file(self.relation_kge_file)
            self.relation_kge.weight = nn.Parameter(torch.from_numpy(np_tensor).type('torch.FloatTensor'))
        else:
            self.relation_kge = None

        # initialize text embeddings
        self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=word_dim,
                                           padding_idx=num_word)
        if self.word_emb_file is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(
                    np.pad(np.load(self.word_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False

    def load_relation_file(self, filename):
        half_tensor = np.load(filename)
        num_pad = 0
        if self.use_self_loop:
            num_pad = 1
        if self.use_inverse_relation:
            load_tensor = np.concatenate([half_tensor, half_tensor])
        else:
            load_tensor = half_tensor
        return np.pad(load_tensor, ((0, num_pad), (0, 0)), 'constant')

    def get_rel_feature(self):
        rel_features = self.relation_embedding.weight
        if self.has_relation_kge:
            rel_features = torch.cat((rel_features, self.relation_kge.weight), dim=-1)
        rel_features = self.relation_linear(rel_features)
        return rel_features

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return self.instruction.init_hidden(num_layer, batch_size, hidden_size)

    def encode_question(self, q_input):
        return self.instruction.encode_question(q_input)

    def get_instruction(self, query_hidden_emb, query_mask, states):
        return self.instruction.get_instruction(query_hidden_emb, query_mask, states)

    def get_loss_bce(self, pred_dist_score, answer_dist):
        answer_dist = (answer_dist > 0).float() * 0.9   # label smooth
        # answer_dist = answer_dist * 0.9  # label smooth
        loss = self.bce_loss_logits(pred_dist_score, answer_dist)
        return loss

    def get_loss_kl(self, pred_dist, answer_dist):
        answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
        answer_len[answer_len == 0] = 1.0
        answer_prob = answer_dist.div(answer_len)
        log_prob = torch.log(pred_dist + 1e-8)
        loss = self.kld_loss(log_prob, answer_prob)
        return loss

    def get_loss_new(self, pred_dist, answer_dist, reduction='mean'):
        if self.loss_type == "bce":
            tp_loss = self.get_loss_bce(pred_dist, answer_dist)
            if reduction == 'none':
                return tp_loss
            else:
                # mean
                return torch.mean(tp_loss)
        else:
            tp_loss = self.get_loss_kl(pred_dist, answer_dist)
            if reduction == 'none':
                return tp_loss
            else:
                # batchmean
                return torch.sum(tp_loss) / pred_dist.size(0)

    def calc_loss_label(self, label_dist, label_valid):
        loss_label = None
        # (batch_size, 1)
        for i in range(self.num_step):
            cur_dist = self.dist_history[i + 1]
            cur_label_dist = label_dist[i]
            cur_label_dist = torch.from_numpy(cur_label_dist).type('torch.FloatTensor').to(self.device)
            # (batch_size, num_entity)
            tp_loss = self.get_loss_new(pred_dist=cur_dist, answer_dist=cur_label_dist, reduction='none')
            # print(tp_loss.size())
            # print(label_valid)
            tp_loss = tp_loss * label_valid
            if self.loss_type == "bce":
                # mean
                cur_loss = torch.mean(tp_loss)
            elif self.loss_type == "kl":
                # batchmean
                cur_loss = torch.sum(tp_loss) / cur_dist.size(0)
            else:
                raise NotImplementedError
            if loss_label is None:
                loss_label = cur_loss
            else:
                loss_label += cur_loss
        return loss_label

    def calc_f1(self, curr_dist, dist_ans, eps=0.01, metric="f1"):
        dist_now = (curr_dist > eps).float()
        dist_ans = (dist_ans > eps).float()
        correct_num = torch.sum(dist_now * dist_ans, dim=-1)
        # guess_num = torch.sum(dist_now, dim=-1)
        answer_num = torch.sum(dist_ans, dim=-1)
        answer_num[answer_num == 0] = 1.0
        pred_num = torch.sum(dist_ans, dim=-1)
        pred_num[pred_num == 0] = 1.0
        recall = correct_num.div(answer_num)
        if metric == 'recall':
            return recall
        precision = correct_num.div(pred_num)
        if metric == 'precision':
            return precision
        mask = (correct_num == 0).float()
        precision_ = precision + mask * VERY_SMALL_NUMBER
        recall_ = recall + mask * VERY_SMALL_NUMBER
        f1 = 2.0 / ((1.0 / precision_) + (1.0 / recall_))
        f1_0 = torch.zeros_like(f1)
        f1 = torch.where(correct_num > 0, f1, f1_0)
        return precision, recall, f1

    def calc_f1_new(self, curr_dist, dist_ans, h1_vec):
        batch_size = curr_dist.size(0)
        max_local_entity = curr_dist.size(1)
        seed_dist = self.dist_history[0]
        local_entity = self.local_entity
        ignore_prob = (1 - self.eps) / max_local_entity
        pad_ent_id = self.num_entity
        # hits_list = []
        f1_list = []
        for batch_id in range(batch_size):
            if h1_vec[batch_id].item() == 0.0:
                f1_list.append(0.0)
                # we consider cases which own hit@1 as prior to reduce computation time
                continue
            candidates = local_entity[batch_id, :].tolist()
            probs = curr_dist[batch_id, :].tolist()
            answer_prob = dist_ans[batch_id, :].tolist()
            seed_entities = seed_dist[batch_id, :].tolist()
            answer_list = []
            candidate2prob = []
            for c, p, p_a, s in zip(candidates, probs, answer_prob, seed_entities):
                if s > 0:
                    # ignore seed entities
                    continue
                if c == pad_ent_id:
                    continue
                if p_a > 0:
                    answer_list.append(c)
                if p < ignore_prob:
                    continue
                candidate2prob.append((c, p))
            precision, recall, f1, hits = f1_and_hits_new(answer_list, candidate2prob, self.eps)
            # hits_list.append(hits)
            f1_list.append(f1)
        # hits_vec = torch.FloatTensor(hits_list).to(self.device)
        f1_vec = torch.FloatTensor(f1_list).to(self.device)
        return f1_vec

    def calc_h1(self, curr_dist, dist_ans, eps=0.01):
        greedy_option = curr_dist.argmax(dim=-1, keepdim=True)
        dist_top1 = torch.zeros_like(curr_dist).scatter_(1, greedy_option, 1.0)
        dist_ans = (dist_ans > eps).float()
        h1 = torch.sum(dist_top1 * dist_ans, dim=-1)
        return (h1 > 0).float()
    
    def get_eval_metric(self, pred_dist, answer_dist):
        with torch.no_grad():
            h1 = self.calc_h1(curr_dist=pred_dist, dist_ans=answer_dist, eps=VERY_SMALL_NUMBER)
            f1 = self.calc_f1_new(pred_dist, answer_dist, h1)
        return h1, f1

    def get_label_valid(self, pred_dist, answer_dist, label_f1=0.8):
        # precision, recall, f1 = self.calc_f1(curr_dist=pred_dist, dist_ans=answer_dist,
        #                                      eps=VERY_SMALL_NUMBER, metric='f1')
        with torch.no_grad():
            h1 = self.calc_h1(curr_dist=pred_dist, dist_ans=answer_dist, eps=VERY_SMALL_NUMBER)
            f1 = self.calc_f1_new(pred_dist, answer_dist, h1)
        f1_valid = (f1 > label_f1).float()
        return (h1 * f1_valid).unsqueeze(1)

    def get_attn_align_loss(self, attn_list):
        align_loss = None
        for i in range(self.num_step):
            other_step = self.num_step - 1 - i
            cur_dist = self.attn_list[i]
            other_dist = attn_list[other_step].detach()
            if align_loss is None:
                align_loss = self.mse_loss(cur_dist, other_dist)
            else:
                align_loss += self.mse_loss(cur_dist, other_dist)
        return align_loss

    def get_dist_align_loss(self, dist_history):
        align_loss = None
        for i in range(self.num_step - 1):
            forward_pos = i + 1
            backward_pos = self.num_step - 1 - i
            cur_dist = self.dist_history[forward_pos]
            back_dist = dist_history[backward_pos].detach()
            if align_loss is None:
                align_loss = self.mse_loss(cur_dist, back_dist)
            else:
                align_loss += self.mse_loss(cur_dist, back_dist)
        return align_loss

    def get_cotraining_loss(self, target_dist, answer_dist):
        # loss_merge = None
        # loss_constraint = None
        # label_valid = self.get_label_valid(pred_dist=target_dist[-1], answer_dist=answer_dist, label_f1=self.label_f1)
        pred_dist = self.dist_history[-1]
        cur_label_dist = target_dist[-1].detach()
        avg_dist = (pred_dist + cur_label_dist) / 2
        loss_merge = self.get_loss_new(pred_dist=avg_dist, answer_dist=answer_dist)
        loss_constraint = self.mse_loss(pred_dist, cur_label_dist)
        # for i in range(self.num_step):
        #     pred_dist = self.dist_history[i + 1]
        #     cur_label_dist = target_dist[i + 1].detach()
        #     if i == self.num_step - 1:
        #         avg_dist = (pred_dist + cur_label_dist) / 2
        #         loss_merge = self.get_loss_new(pred_dist=avg_dist, answer_dist=answer_dist)
        #     else:
        #         if loss_constraint is None:
        #             loss_constraint = self.mse_loss(pred_dist, cur_label_dist)
        #         else:
        #             loss_constraint += self.mse_loss(pred_dist, cur_label_dist)
        return loss_merge, loss_constraint

    def get_constraint_loss(self, target_dist, answer_dist, consider_last=True):
        loss_constraint = None
        label_valid = self.get_label_valid(pred_dist=target_dist[-1], answer_dist=answer_dist, label_f1=self.label_f1)
        # (batch_size, 1)
        if consider_last:
            total_step = self.num_step
        else:
            total_step = self.num_step - 1
        for i in range(total_step):
            pred_dist = self.dist_history[i + 1]
            cur_label_dist = target_dist[i + 1].detach()
            tp_loss = self.get_loss_new(pred_dist=pred_dist, answer_dist=cur_label_dist, reduction='none')
            tp_loss = tp_loss * label_valid
            if self.loss_type == "bce":
                # mean
                cur_loss = torch.mean(tp_loss)
            elif self.loss_type == "kl":
                # batchmean
                cur_loss = torch.sum(tp_loss) / answer_dist.size(0)
            else:
                raise NotImplementedError
            if loss_constraint is None:
                loss_constraint = cur_loss
            else:
                loss_constraint += cur_loss
        return loss_constraint

    def calc_loss_basic(self, answer_dist):
        extras = []
        pred_dist = self.dist_history[-1]
        loss = self.get_loss_new(pred_dist, answer_dist)
        extras.append(loss.item())
        if self.entropy_weight > 0:
            ent_loss = None
            for action_prob in self.action_probs:
                dist = torch.distributions.Categorical(probs=action_prob)
                entropy = dist.entropy()
                if ent_loss is None:
                    ent_loss = torch.mean(entropy)
                else:
                    ent_loss += torch.mean(entropy)
            loss = loss + ent_loss * self.entropy_weight
            extras.append(ent_loss.item())
        else:
            extras.append(0.0)
        return loss, extras

    def calc_loss(self, answer_dist, use_label=False, label_dist=None, label_valid=None):
        extras = []
        pred_dist = self.dist_history[-1]
        if use_label and self.num_step > 1:
            label_valid = torch.from_numpy(label_valid).type('torch.FloatTensor').to(self.device)
            main_loss = self.get_loss_new(pred_dist, answer_dist, reduction='none')
            main_loss = main_loss * (1 - label_valid)
            if self.loss_type == "bce":
                # mean
                main_loss = torch.mean(main_loss)
            elif self.loss_type == "kl":
                # batchmean
                main_loss = torch.sum(main_loss) / batch_size
            else:
                raise NotImplementedError
            # label_valid = torch.from_numpy(label_valid).type('torch.FloatTensor').to(self.device)
            loss_label = self.calc_loss_label(label_dist, label_valid)
            loss = main_loss + loss_label * self.lambda_label
            extras.append(main_loss.item())
            extras.append(loss_label.item())
        else:
            loss = self.get_loss_new(pred_dist, answer_dist)
            extras.append(loss.item())
            extras.append(0.0)
        if self.entropy_weight > 0:
            ent_loss = None
            for action_prob in self.action_probs:
                dist = torch.distributions.Categorical(probs=action_prob)
                entropy = dist.entropy()
                if ent_loss is None:
                    ent_loss = torch.mean(entropy)
                else:
                    ent_loss += torch.mean(entropy)
            loss = loss + ent_loss * self.entropy_weight
            extras.append(ent_loss.item())
        else:
            extras.append(0.0)
        return loss, extras
