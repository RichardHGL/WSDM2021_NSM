import os
import sys
import json
import re
from tqdm import tqdm


def load_vocab(filename):
    f = open(filename)
    voc2id = {}
    for line in f:
        line = line.strip()
        voc2id[line] = len(voc2id)
    return voc2id


def output_dict(ele2id, filename):
    num_ele = len(ele2id)
    id2ele = {v: k for k, v in ele2id.items()}
    f = open(filename, "w")
    for i in range(num_ele):
        f.write(id2ele[i] + "\n")
    f.close()


def deal_rel(tp_str, dataset):
    if dataset == "CWQ" or dataset == "webqsp":
        return tp_str.split(".")
    elif dataset == "metaqa":
        return [tp_str]
    else:
        raise NotImplementedError


def add_word_in_Relation(relation_file, vocab, dataset="webqsp"):
    rel2id = load_vocab(relation_file)
    max_len = 0
    oov = set()
    for rel in rel2id:
        domain_list = deal_rel(rel, dataset)
        tp_list = []
        for domain_str in domain_list:
            tp_list += domain_str.split("_")
        # print(tp_list)
        words = []
        for w_idx, w in enumerate(tp_list):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w != '' and w not in vocab:
                vocab[w] = len(vocab)
                oov.add(w)
                words.append(vocab[w])
        if len(words) > max_len:
            max_len = len(words)
    print("Max length:", max_len)
    print("Total", len(vocab))
    print("OOV relation", len(oov))
    return vocab


def add_word_in_question(inpath):
    vocab = {}
    for split in ["train", "dev", "test"]:
        infile = os.path.join(inpath, split + ".dep")
        f = open(infile)
        for line in f:
            tp_obj = json.loads(line)
            tp_dep = tp_obj["dep"]
            tokens = [item[0] for item in tp_dep]
            for j, word in enumerate(tokens):
                if word not in vocab:
                    vocab[word] = len(vocab)
        f.close()
        print(split, len(vocab))
    # out_file = os.path.join(outpath, "vocab.txt")
    # output_dict(vocab, out_file)
    return vocab


inpath = sys.argv[1]
outpath = sys.argv[2]
dataset = sys.argv[3]
question_vocab = add_word_in_question(inpath)
relation_file = os.path.join(inpath, "relations.txt")
full_vocab = add_word_in_Relation(relation_file, question_vocab, dataset)
out_file = os.path.join(outpath, "vocab_new.txt")
output_dict(full_vocab, out_file)
