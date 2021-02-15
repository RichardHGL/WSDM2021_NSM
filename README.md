# WSDM2021_NSM (Neural State Machine for KBQA)

This is our Pytorch implementation for the paper:

> Gaole He, Yunshi Lan, Jing Jiang, Wayne Xin Zhao and Ji-Rong Wen (2021). Improving Multi-hop Knowledge Base Question Answering by Learning Intermediate Supervision Signals. [paper](https://arxiv.org/abs/2101.03737). In WSDM'2021.


## Introduction
In this paper, we take a new perspective that aims to leverage rich user-item interaction data (user interaction data for short) for improving the KGC task. Our work is inspired by the observation that many KG entities correspond to online items in application systems.

## Requirements:

- Python 3.6
- Pytorch >= 1.3

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
