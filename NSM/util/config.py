import os
import yaml
from tqdm import tqdm
import torch
import argparse


def get_config(config_path=None):
    if not config_path:
        parser = argparse.ArgumentParser()

        # datasets
        parser.add_argument('--name', default='webqsp', type=str)
        parser.add_argument('--data_folder', default='datasets/webqsp/kb_03/', type=str)
        parser.add_argument('--train_data', default='train.json', type=str)
        parser.add_argument('--train_documents', default='documents.json', type=str)
        parser.add_argument('--dev_data', default='dev.json', type=str)
        parser.add_argument('--dev_documents', default='documents.json', type=str)
        parser.add_argument('--test_data', default='test.json', type=str)
        parser.add_argument('--test_documents', default='documents.json', type=str)
        parser.add_argument('--max_query_word', default=10, type=int)
        parser.add_argument('--max_document_word', default=50, type=int)
        parser.add_argument('--max_char', default=25, type=int)
        parser.add_argument('--max_num_neighbors', default=100, type=int)
        parser.add_argument('--max_rel_words', default=8, type=int)

        # embeddings
        parser.add_argument('--word2id', default='glove_vocab.txt', type=str)
        parser.add_argument('--relation2id', default='relations.txt', type=str)
        parser.add_argument('--entity2id', default='entities.txt', type=str)
        parser.add_argument('--char2id', default='chars.txt', type=str)
        parser.add_argument('--word_emb_file', default='glove_word_emb.npy', type=str)
        parser.add_argument('--entity_emb_file', default='entity_emb_100d.npy', type=str)
        parser.add_argument('--rel_word_ids', default='rel_word_idx.npy', type=str)

        # dimensions, layers, dropout
        parser.add_argument('--num_layer', default=1, type=int)
        parser.add_argument('--entity_dim', default=100, type=int)
        parser.add_argument('--word_dim', default=300, type=int)
        parser.add_argument('--hidden_drop', default=0.2, type=float)
        parser.add_argument('--word_drop', default=0.2, type=float)

        # optimization
        parser.add_argument('--num_epoch', default=100, type=int)
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--gradient_clip', default=1.0, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--seed', default=19940715, type=int)
        parser.add_argument('--lr_schedule', action='store_true')
        parser.add_argument('--label_smooth', default=0.1, type=float)
        parser.add_argument('--fact_drop', default=0, type=float)

        # model options
        parser.add_argument('--use_doc', action='store_true')
        parser.add_argument('--use_inverse_relation', action='store_true')
        parser.add_argument('--model_id', default='debug', type=str)
        parser.add_argument('--load_model_file', default=None, type=str)
        parser.add_argument('--mode', default='train', type=str)
        parser.add_argument('--eps', default=0.05, type=float) # threshold for f1

        args = parser.parse_args()
        args.use_cuda = torch.cuda.is_available()

        if args.name == 'webqsp':
            args.type_rels = ['<fb:food.dish.type_of_dish1>', '<fb:film.performance.special_performance_type>', '<fb:geography.mountain.mountain_type>', '<fb:base.aareas.schema.administrative_area.administrative_area_type>', '<fb:base.qualia.disability.type_of_disability>', '<fb:common.topic.notable_types>', '<fb:base.events.event_feed.type_of_event>', '<fb:base.disaster2.injury.type_of_event>', '<fb:religion.religion.types_of_places_of_worship>', '<fb:tv.tv_regular_personal_appearance.appearance_type>']
        else:
            args.type_rels = []

        config = vars(args)
        config['to_save_model'] = True # always save model
        config['save_model_file'] = 'model/' + config['name'] + '/best_{}.pt'.format(config['model_id'])
        config['pred_file'] = 'results/' + config['name'] + '/best_{}.pred'.format(config['model_id'])
    else:
        with open(config_path, "r") as setting:
            config = yaml.load(setting)

    print('-'* 10 + 'Experiment Config' + '-' * 10)
    for k, v in config.items():
        print(k + ': ', v)
    print('-'* 10 + 'Experiment Config' + '-' * 10 + '\n')

    return config