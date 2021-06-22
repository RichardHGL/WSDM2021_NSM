import numpy as np
import os
import sys
from tqdm import tqdm

def load_vocab(filename):
    f = open(filename)
    voc2id = {}
    for line in f:
        line = line.strip()
        voc2id[line] = len(voc2id)
    return voc2id

def output_vec(vocab_file, word2emb, output_file, dim=300):
    f = open(vocab_file)
    lines = f.readlines()
    # vectors = np.array([], dtype=float)
    vectors = np.zeros((len(lines), dim), dtype=float)
    for i, line in tqdm(enumerate(lines)):
        line = line.strip()
        if line in word2emb:
            vectors[i, :] = word2emb[line]
        # if line in word2emb:
        #     vectors = np.append(vectors, word2emb[line])
        # else:
        #     vectors = np.append(vectors, np.zeros(dim, dtype=float))
    np.save(output_file, vectors)

def load_emb(vocab_file, glove_file):
    voc2id = load_vocab(vocab_file)
    words = set(voc2id.keys())
    word2emb = {}
    f = open(glove_file)
    for line in f:
        line = line.strip().split()
        word = line[0]
        if word not in words:
            continue
        try:
            emb = np.array([float(val) for val in line[1:]])
        except ValueError:
            emb = np.array([float(val) for val in line[-300:]])
        word2emb[word] = emb
    print(len(word2emb))
    return word2emb

glove_file = "/mnt/DGX-1-Vol01/gaolehe/data/glove.840B.300d.txt"
dim = 300
data_folder = sys.argv[1]
vocab_file = os.path.join(data_folder, "vocab_new.txt")
output_file = os.path.join(data_folder, "word_emb_300d.npy")
word2emb = load_emb(vocab_file, glove_file)
print("Word emb load done!")
output_vec(vocab_file, word2emb, output_file, dim)
