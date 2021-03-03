# WSDM2021_NSM (Neural State Machine for KBQA)

This is our Pytorch implementation for the paper:

> Gaole He, Yunshi Lan, Jing Jiang, Wayne Xin Zhao and Ji-Rong Wen (2021). Improving Multi-hop Knowledge Base Question Answering by Learning Intermediate Supervision Signals. [paper](https://arxiv.org/abs/2101.03737), [slides](https://github.com/RichardHGL/WSDM2021_NSM/blob/main/presentation/wsdm_slides_ver2.pptx), [poster](https://github.com/RichardHGL/WSDM2021_NSM/blob/main/presentation/wsdm-poster.pdf). In WSDM'2021.


## Introduction
Multi-hop Knowledge Base Question Answering (KBQA) aims to find the answer entities that are multiple hops away in the Knowledge Base (KB) from the entities in the question. A major challenge is the lack of supervision signals at intermediate steps. Therefore, multi-hop KBQA algorithms can only receive the feedback from the final answer, which makes the learning unstable or ineffective. To address this challenge, we propose a novel teacher-student approach for the multi-hop KBQA task. 

## Requirements:

- Python 3.6
- Pytorch >= 1.3

## Dataset
We provide three processed datasets in : WebQuestionsSP (webqsp), Complex WebQuestions 1.1 (CWQ), and MetaQA.
* We follow [GraftNet](https://github.com/OceanskySun/GraftNet) to preprocess the datasets and construct question-specific graph.

|Datasets | Train| Dev | Test | #entity| coverage |
|:---:|---:|---:|---:|---:|---:|
|MetaQA-1hop| 96,106 | 9,992 | 9,947 | 487.6 | 100%|
|MetaQA-2hop| 118,980 | 14,872 | 14,872 | 469.8 | 100%|
|MetaQA-3hop| 114,196 | 14,274 | 14,274 | 497.9| 99.0%|
|webqsp| 2,848 | 250 | 1,639 | 1,429.8 | 94.9%|
|CWQ| 27,639 | 3,519 | 3,531 | 1,305.8 | 79.3%|

Each dataset is organized with following structure:
- `data-name/`
  - `*.dep`: file contains question id, question text and dependency parsing (not used in our code);
  - `*_simple.json`: dataset file, every line describes a question and related question-specific graph;
  - `entities.txt`: file contains a list of entities;
  - `relations.txt`: file contains list of relations.
  - `vocab_new.txt`: vocab file.
  - `word_emb_300d.npy`: vocab related glove embeddings.

## Training Instruction
Download preprocessed datasets from [google drive](https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv?usp=sharing),
and unzip it into dataset folder, and use config --data_folder <data_path> to indicate it.
reported models for webqsp and CWQ dataset are available at [google drive](https://drive.google.com/file/d/15J02zSJTZUFyeBv-hk-2FII3qEoIVyr2/view?usp=sharing).
use following args to run the code

```
example commands: run_webqsp.sh, run_CWQ.sh, run_metaqa.sh
```

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
```
@inproceedings{He-WSDM-2021,
    title = "Improving Multi-hop Knowledge Base Question Answering by Learning Intermediate Supervision Signals",
    author = {Gaole He and
              Yunshi Lan and
              Jing Jiang and
              Wayne Xin Zhao and
              Ji{-}Rong Wen},
    booktitle = {{WSDM}},
    year = {2021},
}
```
Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:
* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.
