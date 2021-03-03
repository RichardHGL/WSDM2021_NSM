import sys
import json
from nltk import Tree


def check_bracket(tp_str):
    left_b = []
    word_b = []
    right_b = []
    length = len(tp_str)
    # print(len(tp_str), tp_str)
    i = 0
    while i < length:
        char = tp_str[i]
        if char == "(":
            left_b.append(i)
            j = i + 1
            while tp_str[j] != " ":
                j += 1
            tp_word = tp_str[i + 1: j]
            word_b.append(tp_word)
        elif char == ")":
            right_b.append(i)
        i += 1
    assert len(left_b) == len(right_b)


def check_redun_spans(tp_str):
    left_b = []
    word_b = []
    right_b = []
    length = len(tp_str)
    # print(len(tp_str), tp_str)
    i = 0
    while i < length:
        char = tp_str[i]
        if char == "(":
            left_b.append(i)
            j = i + 1
            while tp_str[j] != " ":
                j += 1
            tp_word = tp_str[ i +1: j]
            word_b.append(tp_word)
        elif char == ")":
            right_b.append(i)
        i += 1
    assert len(left_b) == len(right_b)
    parents = [-1] * len(left_b)
    child = [None] * len(left_b)
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
                par_index = left_b.index(left_pos[-1])
                parents[cur_pos] = par_index
                if child[par_index] is None:
                    child[par_index] = [cur_pos]
                else:
                    child[par_index].append(cur_pos)
    # print(child)
    # new_str = ""
    spans = []
    for i in range(len(left_b)):
        left_index = left_b[i]
        right_index = end_b[i] + 1
        spans.append(tp_str[left_index: right_index])
    # try:
    #     print(tp_str)
    # except:
    #     print(tp_str.encode('utf8'))
    new_str = deal_str(child, 1, spans)
    # try:
    #     print(new_str)
    # except:
    #     print(new_str.encode('utf8'))
    # try:
    #     check_bracket(new_str)
    # except:
    #     print(new_str)
    #     exit(-1)
    return new_str



def deal_str(child, index, spans):
    # print(index)
    tp_str = spans[index]
    if child[index] is None:
        return tp_str
    j = 0
    while tp_str[j] != " ":
        j += 1
    reserve_part = tp_str.split(" ")[0]
    if len(child[index]) == 1:
        k = child[index][0]
        if child[k] is None:
            reserve_part += " " + spans[k].split(" ")[1]
        else:
            reserve_part += " " + deal_str(child, k, spans) + ")"
    else:
        for k in child[index]:
            # print("middle", k)
            reserve_part += " " + deal_str(child, k, spans)
            # print(reserve_part)
        reserve_part += ")"
    try:
        check_bracket(reserve_part)
    except:
        print(index, child[index])
        print(reserve_part)
        exit(-1)
    return reserve_part


if __name__ == "__main__":
    # filename = "/mnt/DGX-1-Vol01/gaolehe/data/KBQA/Freebase/webqsp/train.con"
    # tp_str = "(ROOT (SBARQ (WHNP (WDT which) (NNS countries)) (SQ (VP (VBP border) (S (NP (DT the)) (NP (PRP us)))))))"
    # tp_str = "(ROOT (SBARQ (WHNP (WDT what) (NN business) (NNS titles)) (SQ (VBD was) " \
    #          "(NP (NP (DT the) (ADJP (RBS most) (JJ famous)) (NNS alumni)) (PP (IN of)" \
    #          " (NP (NN detroit) (NN business) (NN institute)))) (VP (ADVP (RB best)) (VBN known) (PP (IN for))))))"
    # read_const_tree(tp_str)
    filename = sys.argv[1]
    f = open(filename)
    i = 0
    # print(filename)
    for line in f:
        tp_obj = json.loads(line)
        check_redun_spans(tp_obj["con"])
        # str_remove_root(tp_obj["con"])
        i += 1
        # print("id", i)
    f.close()