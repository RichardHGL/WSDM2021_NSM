# How to obtain preprocessed dataset for benchmarks used in our paper

# step 1: obtain subgraph with ppr and get raw dataset
python prepare_data.py <data_path> <max_ent>

# step 2: parse dependency tree & constituency tree
# follow instructions in parse/ folder

# step 3: map kb id, word id...
python build_vocab_from_dep.py <inpath> <outpath> <dataset>
python map_kb_id.py <kb_file> <out_path>
python load_emb_glove.py <data_folder>

# step 4: map raw dataset to id-dataset
python simplify_dataset.py <data_path>

