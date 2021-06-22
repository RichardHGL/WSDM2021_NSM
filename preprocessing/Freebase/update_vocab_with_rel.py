import sys
import os


def load_vocab(filename):
    f = open(filename)
    voc2id = {}
    for line in f:
        line = line.strip()
        voc2id[line] = len(voc2id)
    return voc2id


def deal_rel(tp_str, dataset):
    if dataset == "CWQ" or dataset == "webqsp":
        return tp_str.split(".")
    else:
        raise NotImplementedError


def word2rel(relation_file, vocab_file, output="full/vocab_new.txt", dataset="webqsp"):
    rel2id = load_vocab(relation_file)
    word2id = load_vocab(vocab_file)
    max_len = 0
    words = set()
    oov = set()
    for rel in rel2id:
        domain_list = deal_rel(rel, dataset)
        tp_list = []
        for domain_str in domain_list:
            tp_list += domain_str.split("_")
        # print(tp_list)
        if len(tp_list) > max_len:
            max_len = len(tp_list)
        for word in tp_list:
            if word not in word2id:
                oov.add(word)
            words.add(word)
    print("Max length:", max_len)
    print(len(words))
    print(len(oov))
    f_out = open(output, "w")
    f_in = open(vocab_file)
    for line in f_in:
        f_out.write(line)
    f_in.close()
    for word in oov:
        f_out.write(word + "\n")
    f_out.close()
data_folder = sys.argv[1]
dataset = sys.argv[2]
assert dataset in ["webqsp", "CWQ"]
relation_file = os.path.join(data_folder, "relations.txt")
old_vocab = os.path.join(data_folder, "vocab.txt")
new_vocab = os.path.join(data_folder, "vocab_new.txt")
word2rel(relation_file, old_vocab, new_vocab, dataset)
