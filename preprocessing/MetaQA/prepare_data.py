import os
import sys
import json
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.preprocessing import normalize
from ppr_util import personalized_pagerank


def load_kb(kb_file):
    f = open(kb_file)
    ent2id = {}
    rel2id = {}
    ent2triple = {}
    rows = []
    cols = []
    for line in f:
        head, rel, tail = line.strip().split("|")
        if head not in ent2id:
            ent2id[head] = len(ent2id)
            ent2triple.setdefault(ent2id[head], set())
        if tail not in ent2id:
            ent2id[tail] = len(ent2id)
            ent2triple.setdefault(ent2id[tail], set())
        if rel not in rel2id:
            rel2id[rel] = len(rel2id)
        head_id = ent2id[head]
        # rel_id = rel2id[rel]
        tail_id = ent2id[tail]
        ent2triple[head_id].add((head, rel, tail))
        ent2triple[tail_id].add((head, rel, tail))
        rows.append(head_id)
        rows.append(tail_id)
        cols.append(tail_id)
        cols.append(head_id)
    f.close()
    print("Number Entity : {}, Relation : {}".format(len(ent2id), len(rel2id)))
    vals = np.ones((len(rows),))
    rows = np.array(rows)
    cols = np.array(cols)
    sp_mat = csr_matrix((vals, (rows, cols)), shape=(len(ent2id), len(ent2id)))

    return ent2id, rel2id, ent2triple, normalize(sp_mat, norm="l1", axis=1)


def get_subgraph(cand_ents, ent2triple, ent2id, rel2id):
    triple_set = set()
    cand_set = set(cand_ents)
    for ent in cand_ents:
        triple_set |= ent2triple[ent]
    readable_tuples = []
    for triple in triple_set:
        head, rel, tail = triple
        head_id = ent2id[head]
        if head_id not in cand_set:
            continue
        rel_id = rel2id[rel]
        tail_id = ent2id[tail]
        if tail_id not in cand_set:
            continue
        readable_tuples.append([
            {"kb_id": head_id, "text": head},
            {"rel_id": rel_id, "text": rel},
            {"kb_id": tail_id, "text": tail},
        ])
    return readable_tuples


def find_linked_ents(query_text):
    length = len(query_text)
    start_pos_list = []
    end_pos_list = []
    num_ent = 0
    for i in range(length):
        if query_text[i] == "[":
            start_pos = i - (num_ent * 2)
            start_pos_list.append(start_pos)
            num_ent += 1
        if query_text[i] == "]":
            end_pos = i - (num_ent * 2 - 1)
            end_pos_list.append(end_pos)
    query_text = query_text.replace("[", "")
    query_text = query_text.replace("]", "")
    return start_pos_list, end_pos_list, query_text


def rank_ppr_ents(seed_list, sp_mat, mode="fixed", max_ent=500, min_ppr=0.005):
    seed = np.zeros((sp_mat.shape[0], 1))
    seed[seed_list] = 1. / len(set(seed_list))
    ppr = personalized_pagerank(seed, sp_mat, restart_prob=0.8, max_iter=20)
    if mode == "fixed":
        sorted_idx = np.argsort(ppr)[::-1]
        extracted_ents = sorted_idx[:max_ent]
        # check if any ppr values are nearly zero
        zero_idx = np.where(ppr[extracted_ents] < 1e-6)[0]
        if zero_idx.shape[0] > 0:
            extracted_ents = extracted_ents[:zero_idx[0]]
    else:
        extracted_ents = np.where(ppr > min_ppr)[0]
    return extracted_ents


def _readable_entities(ent_list, ent2id):
    id2ent = {v: k for k, v in ent2id.items()}
    readable_entities = []
    ent_set = set(ent_list.tolist())
    for ent in ent_set:
        readable_entities.append({"text": id2ent[ent], "kb_id": ent})
    return readable_entities


def _get_answer_coverage(answers, entities):
    found, total = 0., 0
    all_entities = set(entities)
    for answer in answers:
        if answer["kb_id"] in all_entities:
            found += 1.
        total += 1
    return found / total


def output_dict(ele2id, filename):
    num_ele = len(ele2id)
    id2ele = {v: k for k, v in ele2id.items()}
    f = open(filename, "w")
    for i in range(num_ele):
        f.write(id2ele[i] + "\n")
    f.close()


def process_metaqa(input_path, output_path, kb_file, max_ent=500):
    ent2id, rel2id, ent2triple, sp_mat = load_kb(kb_file)
    output_dict(ent2id, os.path.join(output_path, "entities.txt"))
    output_dict(rel2id, os.path.join(output_path, "relations.txt"))
    for split in ["dev", "train", "test"]:
        infile = "qa_" + split + ".txt"
        outfile = split + ".json"
        input = os.path.join(input_path, infile)
        output = os.path.join(output_path, outfile)
        answer_coverage = []
        f_in = open(input)
        f_out = open(output, "w")
        for index, line in tqdm(enumerate(f_in)):
            tp_obj = {}
            seed_obj = []
            answer_obj = []
            tp_obj["id"] = split + "_" + str(index)
            question_text, answer_text = line.strip().split("\t")
            start_pos_list, end_pos_list, question_text = find_linked_ents(question_text)
            tp_obj["question"] = question_text
            # To ensure token in vocab, tokenize may be need
            assert len(start_pos_list) == len(end_pos_list)
            for i, start_pos in enumerate(start_pos_list):
                end_pos = end_pos_list[i]
                assert start_pos < end_pos
                ent_str = question_text[start_pos:end_pos]
                assert ent_str in ent2id
                seed_obj.append({
                    "start": start_pos,
                    "end": end_pos,
                    "text": ent_str,
                    "kb_id": ent2id[ent_str]
                })
            answers = answer_text.split("|")
            for answer in answers:
                assert answer in ent2id
                answer_obj.append({
                    "text": answer,
                    "kb_id": ent2id[answer]
                })
            tp_obj["entities"] = seed_obj
            tp_obj["answers"] = answer_obj
            seed_list = [ent["kb_id"] for ent in seed_obj]
            if len(seed_obj) == 0:
                extracted_tuples = []
                extracted_ents = []
            else:
                extracted_ents = rank_ppr_ents(seed_list, sp_mat, max_ent=max_ent)
                extracted_tuples = get_subgraph(extracted_ents, ent2triple, ent2id, rel2id)
            tp_obj["subgraph"] = {}
            tp_obj["subgraph"]["tuples"] = extracted_tuples
            tp_obj["subgraph"]["entities"] = _readable_entities(extracted_ents, ent2id)
            f_out.write(json.dumps(tp_obj) + "\n")
            answer_coverage.append(_get_answer_coverage(answer_obj, extracted_ents))
        print("Answer coverage in retrieved subgraphs = %.3f" % (np.mean(answer_coverage)))
        f_in.close()
        f_out.close()


data_folder = sys.argv[1]
max_ent = eval(sys.argv[2])
kb_file = os.path.join(data_folder, "kb.txt")
# for dir in ["1-hop"]:
for dir in ["1-hop", "2-hop", "3-hop"]:
    inpath = os.path.join(data_folder, dir + "/vanilla")
    outpath = os.path.join(data_folder, dir + "/big")
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    process_metaqa(inpath, outpath, kb_file, max_ent=max_ent)
