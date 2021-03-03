import sys
import json
from nltk import Tree


def read_const_tree(tp_str):
    left_b = []
    word_b = []
    right_b = []
    length = len(tp_str)
    i = 0
    while i < length:
        char = tp_str[i]
        if char == "(":
            left_b.append(i)
            j = i + 1
            while tp_str[j] != " ":
                j += 1
            tp_word = tp_str[i+1: j]
            word_b.append(tp_word)
        elif char == ")":
            right_b.append(i)
        i += 1
    assert len(left_b) == len(right_b)
    parents = [-1] * len(left_b)
    end_b = [-1] * len(left_b)
    num_finish = 0
    left_pos = []
    for i, char in enumerate(tp_str):
        if char == "(":
            left_pos.append(i)
        if char == ")":
            num_finish += 1
            l_p = left_pos.pop()
            cur_pos = left_b.index(l_p)
            end_b[cur_pos] = i
            if len(left_pos) > 0:
                parents[cur_pos] = left_b.index(left_pos[-1])
    # print(parents)
    # print(left_b)
    # print(end_b)
    # print(word_b)
    word_list = []
    word_parent = []
    for j, par in enumerate(parents):
        if j not in parents:
            span = tp_str[left_b[j] + 1:end_b[j]]
            word_list.append(span.split(" ")[1])
            word_parent.append(j)
            # print(j, left_b[j], end_b[j])
            # print(span)
    # print(word_list)
    # print([(word_parent[i], word_b[word_parent[i]]) for i in range(len(word_list))])
    # Tree.fromstring(tp_str).pretty_print()
    node2layer = {}
    node2layer[-1] = 0
    # initial add root node at layer 0
    for i in range(len(parents)):
        if i in node2layer:
            continue
        search_hierachy(i, parents, node2layer)
    node_layer = [node2layer[i] for i in range(len(parents))]
    # print(node_layer)
    # print(len(node_layer), len(word_b), len(parents), len(word_list), len(word_parent))
    return node_layer, word_b, parents, word_list, word_parent


def load_const_str(tp_str):
    num_left = 0
    num_right = 0
    parents = []
    left_pos = []
    right_pos = []
    spans = []
    print(len(tp_str))
    child_list = {}
    for i, char in enumerate(tp_str):
        if char == "(":
            num_left += 1
            left_pos.append(i)
        elif char == ")":
            num_right += 1
            right_pos.append(i)
            left_bracket = left_pos.pop() + 1
            spans.append((left_bracket, i))
    print(child_list)
    for left_ind, right_ind in spans:
        print(left_ind, right_ind)
        print(tp_str[left_ind: right_ind])


def search_hierachy(index, parents, node2layer):
    cur_parent = parents[index]
    # print(index, cur_parent)
    if cur_parent not in node2layer:
        parent_layer = search_hierachy(cur_parent, parents, node2layer)
        # this recursive process will annotate graph layers
    else:
        parent_layer = node2layer[cur_parent]
    node2layer[index] = parent_layer + 1
    return parent_layer + 1


def read_tree(edge_list):
    length = len(edge_list)
    parents = []
    relations = []
    for i in range(length):
        parents.append(eval(edge_list[i][2]) - 1)
        relations.append(edge_list[i][3])
    node2layer = {}
    node2layer[-1] = 0
    # initial add root node at layer 0
    for i in range(length):
        if i in node2layer:
            continue
        search_hierachy(i, parents, node2layer)
    node_layer = [node2layer[i] for i in range(length)]
    max_layer = max(node2layer)
    # max_layer = 0
    # layer2node = {}
    # for i in range(length):
    #     if node2layer[i] > max_layer:
    #         max_layer = node2layer[i]
    #     cur_layer = node2layer[i]
    #     layer2node.setdefault(cur_layer, [])
    #     layer2node[cur_layer].append(i)
    return node_layer, parents, relations


if __name__ == "__main__":
    # filename = "/mnt/DGX-1-Vol01/gaolehe/data/KBQA/Freebase/webqsp/train.con"
    # tp_str = "(ROOT (SBARQ (WHNP (WDT which) (NNS countries)) (SQ (VP (VBP border) (S (NP (DT the)) (NP (PRP us)))))))"
    tp_str = "(ROOT (SBARQ (WHNP (WDT what) (NN business) (NNS titles)) (SQ (VBD was) " \
             "(NP (NP (DT the) (ADJP (RBS most) (JJ famous)) (NNS alumni)) (PP (IN of)" \
             " (NP (NN detroit) (NN business) (NN institute)))) (VP (ADVP (RB best)) (VBN known) (PP (IN for))))))"
    read_const_tree(tp_str)
    # filename = sys.argv[1]
    # f = open(filename)
    # for line in f:
    #     tp_obj = json.loads(line)
    #     read_tree(tp_obj["dep"])
    # f.close()