import os
import sys
import json


def output_dict(ele2id, filename):
    num_ele = len(ele2id)
    id2ele = {v: k for k, v in ele2id.items()}
    f = open(filename, "w")
    for i in range(num_ele):
        f.write(id2ele[i] + "\n")
    f.close()


def load_kb(in_path, out_path):
    ent2id = {}
    rel2id = {}
    for name in ["train", "dev", "test"]:
        filename = os.path.join(in_path, name+".json")
        f = open(filename)
        for line in f:
            tp_obj = json.loads(line)
            for triple in tp_obj["subgraph"]["tuples"]:
                head, rel, tail = triple
                head = head["kb_id"]
                rel = rel["rel_id"]
                tail = tail["kb_id"]
                if head not in ent2id:
                    ent2id[head] = len(ent2id)
                if tail not in ent2id:
                    ent2id[tail] = len(ent2id)
                if rel not in rel2id:
                    rel2id[rel] = len(rel2id)
            for answer in tp_obj['answers']:
                keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                answer_ent = answer[keyword]
                if answer_ent not in ent2id:
                    ent2id[answer_ent] = len(ent2id)
            for entity in tp_obj['entities']:
                seed_ent = entity['text']
                if seed_ent not in ent2id:
                    ent2id[seed_ent] = len(ent2id)
        f.close()
    print("Number Entity : {}, Relation : {}".format(len(ent2id), len(rel2id)))
    output_dict(ent2id, os.path.join(out_path, "entities.txt"))
    output_dict(rel2id, os.path.join(out_path, "relations.txt"))


load_kb(sys.argv[1], sys.argv[2])