import json
import numpy as np
from NSM.util.config import get_config
import time
from NSM.data.dataset_super import SingleDataLoader


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id


def load_data(config):
    entity2id = load_dict(config['data_folder'] + config['entity2id'])
    word2id = load_dict(config['data_folder'] + config['word2id'])
    relation2id = load_dict(config['data_folder'] + config['relation2id'])
    if config["is_eval"]:
        train_data = None
        valid_data = None
    else:
        train_data = SingleDataLoader(config, word2id, relation2id, entity2id, data_type="train")
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, data_type="dev")
    test_data = SingleDataLoader(config, word2id, relation2id, entity2id, data_type="test")
    dataset = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "word2id": word2id
    }
    return dataset


if __name__ == "__main__":
    st = time.time()
    args = get_config()
    load_data(args)
