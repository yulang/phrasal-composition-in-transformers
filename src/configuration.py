from random import randint
from os import path

DATA_FOLDER = "/home-nfs/langyu/data_folder"

config = {
    # model_class only affect transformer_exp, option:
    # bert, roberta, distillbert, xlmroberta, xlnet
    "model_class": "bert",
    "normalize": False,
    "dump_every": 5,
    "embed_in_sent": True,
    "workload": "ppdb_exact", # ppdb, ppdb_exact, bird, stanford_sent, kintsch
    "downstream_task": "coe", # classification or coe
    "negative_sample_mode": "half_neg",  # None, "one_per_source", "half_neg", "all_neg", (ppdb_exact will always has helf neg)
    "dump_path": "/home-nfs/langyu/data_folder/bert-dump/",
    "BIRD_LOC": path.join(DATA_FOLDER, "BiRD/BiRD.txt"),
    "PPDB_LOC": path.join(DATA_FOLDER, "ppdb-2.0-tldr"),
    "KINTSCH_LOC": path.join(DATA_FOLDER, "kintsch.txt"),
    "STANFORD_LOC": path.join(DATA_FOLDER, "stanfordSentimentTreebank"),
    "TEXT_CORPUS": path.join(DATA_FOLDER, "enwiki/enwiki-unidecoded.txt"),
    "EMBEDDING_OUT_LOC": path.join(DATA_FOLDER, "bert-embeddings/"),
    "sample_size": 100,
    "overlap_threshold": 0.5,
    "rand_seed": str(randint(10000, 99999)),
    "correlation": True,
    "classification": True,
    "retrieve_seed": None,  # if skip_evaluation is true, use this seed to retrieve corresponding dumps
    "include_input_emb": True,
}

BERT_VARIANTS = ["roberta", "bert", "distillbert", "xlmroberta"]
