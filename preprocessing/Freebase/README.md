# Preprocessing steps to obtain question-specific graph

In this page, I try to show how to preprocess Freebase related datasets (Take CWQ as example). For MetaQA preprocessing, you can refer to MetaQA folder.

Download Freebase dump with following command:

```
wget https://download.microsoft.com/download/A/E/4/AE428B7A-9EF9-446C-85CF-D8ED0C9B1F26/FastRDFStore-data.zip --no-check-certificate
```

unzip `FastRDFStore-data.zip` and keep `fb_en.txt` and `cvtnodes.bin` in `data/` folder.

## Step 0: filter Freebase 
```
python manual_filter_rel.py
# we can get manual_filter_fb.txt, it will be used in get_2hop_subgraph.py
```

## Step 1: prepare basic elements (question, answer, topic entities) for dataset
```
python preprocess_step0.py 
# get CWQ_step0.json
mkdir CWQ
mv CWQ_step0.json CWQ/

python get_seed_set.py CWQ/CWQ_step0.json CWQ/CWQ_seed.txt
# get all topic entities in CWQ_step0.json as seed set
```

## Step 2: extract 2-hop neighborhood for topic entities in questions
```
mkdir CWQ/subgraph/
python get_2hop_subgraph.py CWQ/CWQ_seed.txt CWQ/subgraph/CWQ_subgraph.txt
# usage: python get_2hop_subgraph.py <seed_file> <graph_file>
```

## Step 3: reserve important nodes with ppr and obtain question-specific graph
```
python preprocess_step1.py CWQ/subgraph/CWQ_subgraph.txt CWQ/CWQ_step0.json CWQ/CWQ_step1.json
# usage: python preprocess_step1.py <graph_file> <in_file> <out_file>
```

## Step 4: separate data split with id and get train/dev/test.json and put all files in CWQ/ folder

## Step 5: parse dependency tree & constituency tree (not necessary)
follow instructions in `preprocessing/parse/` folder
Actually, we can just use the function `tokenize_sent` in `preprocessing/parse/dep_parse.py` to tokenize sentences. 
Our repo can actually work without `.dep` files, just a small modification about question loading in `NSM/data/basic_dataset.py` is enough.
`tokenize_sent` function can be also found in that file.

## Step 6: map kb id, word id...
```
# usage: python build_vocab_from_dep.py <inpath> <outpath> <dataset>
python build_vocab_from_dep.py CWQ/ CWQ/ CWQ
# usage: python build_vocab_from_dep.py <inpath> <dataset>
python update_vocab_with_rel.py CWQ/ CWQ
# usage: map_kb_id.py <in_path> <out_path>
python map_kb_id.py CWQ/ CWQ/
# usage: load_emb_glove.py <data_folder>
python load_emb_glove.py CWQ/
```

## Step 7: simplify json file
```
python simplify_dataset.py CWQ/
# usage: python simplify_dataset.py <data_path>
```

Now, the dataset used in this repo is constructed.

