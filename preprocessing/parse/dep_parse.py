import sys
import os
import re
import json
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPDependencyParser


parser = CoreNLPParser(url='http://localhost:9000')
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')


def tokenize_sent(question_text):
    question_text = question_text.strip().lower()
    question_text = re.sub('\'s', ' s', question_text)
    words = []
    for w_idx, w in enumerate(question_text.split(' ')):
        w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
        if w == '':
            continue
        words += [w]
    return words


def get_dep_parse(parses):
    dep_parse = []
    best_parse = next(parses)
    x = best_parse.to_conll(4)
    tp_list = x.strip().split("\n")
    for sub_str in tp_list:
        dep_parse.append(sub_str.split("\t"))
    # for parse in parses:
    #     x = parse.to_conll(4)
    #     tp_list = x.strip().split("\n")
    #     for sub_str in tp_list:
    #         dep_parse.append(sub_str.split("\t"))
    return dep_parse


def get_question(input, output):
    f = open(input, encoding='utf8')
    f1 = open(output, "w", encoding='utf8')
    for line in f:
        data = json.loads(line)
        id = data["id"]
        question = data["question"]
        # tokens = question.split()
        tokens = tokenize_sent(question)
        # const_parse = list(parser.parse(tokens))
        parses = dep_parser.parse(tokens)
        dep_parse = get_dep_parse(parses)
        # dep_parse = [[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in parses]
        new_obj = {
            "id": id,
            "dep": dep_parse,
            "question": question
        }
        f1.write(json.dumps(new_obj) + "\n")
    f.close()
    f1.close()


get_question(sys.argv[1], sys.argv[2])
