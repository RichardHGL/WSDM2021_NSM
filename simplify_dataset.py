import json
import sys
import os
import numpy as np
import re
from tqdm import tqdm


def simplify_entities(entity_list, entity2id):
    ent_id_list = []
    for entity in entity_list:
        entity_text = entity['text'].strip()
        if entity_text not in entity2id:
            print(entity_text)
            assert False
        entity_global_id = entity2id[entity_text]
        ent_id_list.append(entity_global_id)
    return ent_id_list


def simplify_tuples(tuple_list, entity2id, relation2id):
    triple_list = []
    for triple in tuple_list:
        sbj, rel, obj = triple
        head = entity2id[sbj['text']]
        rel = relation2id[rel['text']]
        tail = entity2id[obj['text']]
        triple_list.append([head, rel, tail])
    return triple_list


def simplify_data(input, output, entity2id, relation2id):
    f_in = open(input, "r")
    f_out = open(output, "w")
    for line in tqdm(f_in):
        tp_dict = json.loads(line)
        tp_dict["entities"] = simplify_entities(tp_dict["entities"], entity2id)
        tp_dict["subgraph"]["tuples"] = simplify_tuples(tp_dict["subgraph"]["tuples"], entity2id, relation2id)
        tp_dict["subgraph"]["entities"] = simplify_entities(tp_dict["subgraph"]["entities"], entity2id)
        f_out.write(json.dumps(tp_dict) + "\n")
    f_in.close()
    f_out.close()


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id


def load_id_map(data_path):
    entity2id = load_dict(os.path.join(data_path, "entities.txt"))
    relation2id = load_dict(os.path.join(data_path, "relations.txt"))
    return entity2id, relation2id


data_path = sys.argv[1]
entity2id, relation2id = load_id_map(data_path)
for name in ["train", "dev", "test"]:
    input_file = os.path.join(data_path, name + ".json")
    output_file = os.path.join(data_path, name + "_simple.json")
    print("simplify ", input_file)
    simplify_data(input_file, output_file, entity2id, relation2id)
