import os
import sys


def output_dict(ele2id, filename):
    num_ele = len(ele2id)
    id2ele = {v: k for k, v in ele2id.items()}
    f = open(filename, "w")
    for i in range(num_ele):
        f.write(id2ele[i] + "\n")
    f.close()


def load_kb(kb_file, out_path):
    f = open(kb_file)
    ent2id = {}
    rel2id = {}
    ent2triple = {}
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
    f.close()
    print("Number Entity : {}, Relation : {}".format(len(ent2id), len(rel2id)))
    output_dict(ent2id, os.path.join(out_path, "entities.txt"))
    output_dict(rel2id, os.path.join(out_path, "relations.txt"))


load_kb(sys.argv[1], sys.argv[2])