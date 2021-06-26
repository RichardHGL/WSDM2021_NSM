import json
import os
from tqdm import tqdm
import re


def is_ent(tp_str):
    if len(tp_str) < 3:
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        print(tp_str)
        return True
    return False


def find_entity(sparql_str):
    str_lines = sparql_str.split("\n")
    # tp_list = re.finditer(r"ns:[a-z]\..* ", str_lines)
    ent_set = set()
    unend_flag = False
    unend_text = None
    for line in str_lines[1:]:
        if "ns:" not in line:
            continue
        # if unend_flag:
        #     line = unend_text + " " + line.lstrip()
        #     unend_flag = False
        #     unend_text = ""
        # print(line)
        # if ";" in line and len(line.split(";")) == 2:
        #     unend_text = line.split(";")[1].strip()
        #     unend_flag = True
        #     line = line.split(";")[0].strip()
        spline = line.strip().split(" ")
        for item in spline:
            ent_str = item[3:].replace("(", "")
            ent_str = ent_str.replace(")", "")
            if is_ent(ent_str):
                ent_set.add(ent_str)
        # print(line)
        # head = spline[0]
        # rel = spline[1]
        # tail = spline[2]
        # # print(spline)
        # if head.startswith("ns:"):
        #     # print(head)
        #     ent_set.add(head[3:])
        # if tail.startswith("ns:"):
        #     # print(tail)
        #     ent_set.add(tail[3:])
    return ent_set
# data can be downloaded from https://github.com/lanyunshi/KBQA-GST
data_folder = "/mnt/DGX-1-Vol01/gaolehe/tools/KBQA-GST/data/CWQ"
data_file = ["ComplexWebQuestions_train.json", "ComplexWebQuestions_test_wans.json", "ComplexWebQuestions_dev.json"]
# all_data = []
output_file = "CWQ_step0.json"
f_out = open(output_file, "w")
for file in data_file:
    filename = os.path.join(data_folder, file)
    with open(filename) as f_in:
        data = json.load(f_in)
        for q_obj in data:
            # question = q_obj['QuestionText']
            ID = q_obj["ID"]
            # print()
            # answer_list = q_obj["answers"]
            answer_list_new = []
            for answer_obj in q_obj["answers"]:
                new_obj = {}
                new_obj["kb_id"] = answer_obj["answer_id"]
                new_obj["text"] = answer_obj["answer"]
                answer_list_new.append(new_obj)
            question = q_obj["question"]
            sparql_str = q_obj["sparql"]
            # print(question)
            ent_set = find_entity(sparql_str)
            ent_list = [{"kb_id": ent, "text": ent} for ent in ent_set]
            new_obj = {
                "id": ID,
                "answers": answer_list_new,
                "question": question,
                "entities": ent_list,
            }
            f_out.write(json.dumps(new_obj) + "\n")
            # all_data.append(new_obj)
f_out.close()
