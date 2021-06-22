import sys
import os
import json
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from util.ppr_util import rank_ppr_ents
from util.deal_cvt import load_cvt, is_cvt
import time
import numpy as np

cvt_nodes = load_cvt()


def build_sp_mat(subgraph):
    ent2id = {}
    rel2id = {}
    rows = []
    cols = []
    for triple in subgraph:
        head, rel, tail = triple
        if head not in ent2id:
            ent2id[head] = len(ent2id)
        if tail not in ent2id:
            ent2id[tail] = len(ent2id)
        if rel not in rel2id:
            rel2id[rel] = len(rel2id)
        head_id = ent2id[head]
        tail_id = ent2id[tail]
        rows.append(head_id)
        rows.append(tail_id)
        cols.append(tail_id)
        cols.append(head_id)
    print("Number Entity : {}, Relation : {}".format(len(ent2id), len(rel2id)))
    print("Number Triple : {}".format(len(subgraph)))
    vals = np.ones((len(rows),))
    rows = np.array(rows)
    cols = np.array(cols)
    sp_mat = csr_matrix((vals, (rows, cols)), shape=(len(ent2id), len(ent2id)))

    return ent2id, rel2id, normalize(sp_mat, norm="l1", axis=1)


# def load_kb(kb_file):
#     f = open(kb_file)
#     triples = {}
#     back_triples = {}
#     for line in f:
#         head, rel, tail = line.strip().split("\t")
#         triples.setdefault(head, set())
#         triples[head].add((head, rel, tail))
#         back_triples.setdefault(tail, set())
#         back_triples[tail].add((head, rel, tail))
#     f.close()
#     return triples, back_triples


def load_kb(kb_file):
    f = open(kb_file)
    triples = {}
    back_triples = {}
    for line in f:
        head, rel, tail = line.strip().split("\t")
        triples.setdefault(head, set())
        triples[head].add((head, rel, tail))
        if is_cvt(tail, cvt_nodes):
            back_triples.setdefault(tail, set())
            back_triples[tail].add((head, rel, tail))
    f.close()
    for cvt_node in back_triples:
        # cvt node give subject entity further step triples
        for triple in back_triples[cvt_node]:
            head, rel, tail = triple
            if tail not in triples:
                triples[head].remove(triple)
                # Single cvt triple is of no sense
                continue
            triples[head] |= triples[tail]
    return triples, back_triples


def _readable_entities(ent_list, ent2id):
    id2ent = {v: k for k, v in ent2id.items()}
    readable_entities = []
    ent_set = set(ent_list.tolist())
    for ent in ent_set:
        readable_entities.append({"text": id2ent[ent], "kb_id": id2ent[ent]})
    return readable_entities


def get_1hop_triples(triples, seed_ent):
    if seed_ent not in triples:
        return set()
    return triples[seed_ent]


def get_1hop_back_triples(back_triples, seed_ent):
    if seed_ent not in back_triples:
        return set()
    return back_triples[seed_ent]


def get_2hop_triples(triples, seed_ent):
    # all cvt case have been taken into consideration
    triple_set = set()
    if seed_ent not in triples:
        return triple_set
    hop1_triples = triples[seed_ent]
    triple_set |= hop1_triples
    for head, rel, tail in hop1_triples:
        if tail in triples:
            triple_set |= triples[tail]
    return triple_set


def get_subgraph(cand_ents, triples, back_triples, cvt_add_flag=True):
    triple_set = set()
    reserve_set = set()
    # ent_list = [ent["text"] for ent in entities]
    # cand_ents = set(ent_list)
    for ent in cand_ents:
        triple_set |= get_1hop_triples(triples, ent)
        # if is_cvt(ent, cvt_nodes):
        #     reserve_set |= get_1hop_triples(triples, ent)
        #     reserve_set |= get_1hop_back_triples(back_triples, ent)
        #     cvt nodes add all triples related to it
        # triple_set |= get_1hop_back_triples(back_triples, ent)
    tp_edges = {}
    for triple in triple_set:
        head, rel, tail = triple
        if head not in cand_ents:
            continue
        if tail not in cand_ents:
            continue
        reserve_set.add((head, rel, tail))
        tp_edges.setdefault(head, set())
        tp_edges[head].add((head, rel, tail))
    for ent in tp_edges:
        for triple in tp_edges[ent]:
            _, __, tail = triple
            if is_cvt(tail, cvt_nodes):
                if tail not in triples:
                    # single hop to cvg entity is of no sense
                    reserve_set.remove(triple)
                elif tail not in tp_edges:
                    if cvt_add_flag:
                        reserve_set |= triples[tail]
                    else:
                        # single hop to cvg entity is of no sense
                        reserve_set.remove(triple)
    readable_tuples = []
    ent_set = set()
    for triple in reserve_set:
        head, rel, tail = triple
        readable_tuples.append([
            {"kb_id": head, "text": head},
            {"rel_id": rel, "text": rel},
            {"kb_id": tail, "text": tail},
        ])
        ent_set.add(head)
        ent_set.add(tail)
    readable_entities = []
    for ent in ent_set:
        readable_entities.append({"text": ent, "kb_id": ent})
    return readable_entities, readable_tuples


def _get_answer_coverage(answers, entities):
    if len(answers) == 0:
        return 1.0
    found, total = 0., 0
    ent_list = [ent["text"] for ent in entities]
    all_entities = set(ent_list)
    for answer in answers:
        if answer["kb_id"] in all_entities:
            found += 1.
        total += 1
    return found / total


def retrieve_subgraph(kb_file, in_file, out_file=None, max_ent=2000):
    st = time.time()
    triples, back_triples = load_kb(kb_file)
    print("Load KB", time.time() - st)
    f = open(in_file)
    data = []
    answer_coverage = []
    for line in f:
        data.append(json.loads(line))
    f.close()
    f_out = open(out_file, "w")
    print("Load data", time.time() - st)
    for i, q_obj in enumerate(data):
        q_obj = data[i]
        entity_set = set()
        triple_set = set()
        for ent in q_obj["entities"]:
            entity = ent['kb_id']
            entity_set.add(entity)
            triple_set |= get_2hop_triples(triples, entity)
        if len(entity_set) == 0 or len(triple_set) == 0:
            extracted_tuples = []
            extracted_ents = []
            readable_entities = []
            print("Bad question", q_obj["id"])
        else:
            ent2id, rel2id, sp_mat = build_sp_mat(triple_set)
            seed_list = []
            for ent in entity_set:
                if ent in ent2id:
                    seed_list.append(ent2id[ent])
            # seed_list = [ent2id[ent] for ent in entity_set]
            extracted_ents = rank_ppr_ents(seed_list, sp_mat, max_ent=max_ent)
            id2ent = {v: k for k, v in ent2id.items()}
            ent_set = set(extracted_ents.tolist())
            ent_set_new = set(id2ent[ent] for ent in ent_set)   # | entity_set
            # readable_entities = _readable_entities(extracted_ents, ent2id)
            # extracted_tuples = get_subgraph(readable_entities, triples, back_triples)
            readable_entities, extracted_tuples = get_subgraph(ent_set_new, triples,
                                                               back_triples, cvt_add_flag=False)
        q_obj["subgraph"] = {}
        q_obj["subgraph"]["tuples"] = extracted_tuples
        # readable_entities = _readable_entities(extracted_ents, ent2id)
        q_obj["subgraph"]["entities"] = readable_entities
        f_out.write(json.dumps(q_obj) + "\n")
        coverage = _get_answer_coverage(q_obj["answers"], readable_entities)
        # print("ID: {}, Answer: {}, Coverage: {}".format(q_obj["id"], len(q_obj["answers"]), coverage))
        # print("entity: {}, triple: {}".format(len(readable_entities), len(extracted_tuples)))
        answer_coverage.append(coverage)
        if i % 100 == 0:
            print("{} samples cost time:{:.1f} with coverage{:.3f}".format(i,
                                                                           time.time() - st, np.mean(answer_coverage)))
    print("Finish", time.time() - st)
    print("Answer coverage in retrieved subgraphs = %.3f" % (np.mean(answer_coverage)))

# kb_file = "webqsp_hop2.txt"
# in_file = "webqsp_step0.json"
# out_file = "webqsp_step1.json"
kb_file = sys.argv[1]
in_file = sys.argv[2]
out_file = sys.argv[3]
retrieve_subgraph(kb_file, in_file, out_file, max_ent=2000)
