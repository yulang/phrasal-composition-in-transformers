# About
This repo contains datasets and code for EMNLP 2020 paper [*Assessing Phrasal Representation and Composition in Transformers*](https://www.aclweb.org/anthology/2020.emnlp-main.397.pdf), by Lang Yu and Allyson Ettinger.

Our follow-up paper [*On the Interplay Between Fine-tuning and Composition in Transformers*](https://arxiv.org/abs/2105.14668) is coming up at Findings of ACL 2021.

# Notes
I wrote two notes explaining the paper. They can be found [here](https://timur-yu.medium.com/analyzing-contextualized-embeddings-in-bert-and-other-transformer-models-214377e74a17).
# Dependencies
The code is implemented with Huggingface's transformer v2.5.1. There are multiple API changes in their recent releases. And you may need to adjust related code if you are using a different version.
# Repo structure
- `src/` contains source code to generate dataset and perform analysis
- `datasets` contains dataset for landmark test from (Kintsch 2001), and datasets we used to get the metrics reported in the paper.
# Dataset
- Similarity correlation dataset: As mentioned in the paper, the full dataset can be downloaded here: http://saifmohammad.com/WebPages/BiRD.html. Please refer to the original paper for details about the dataset.
- Paraphrase classification dataset: You can download ppdb-2.0-tldr from http://paraphrase.org.
- Landmark experiment dataset: included in the repo. Exact from the serie of papers by Kintsch.

You can use the datasets inclued in the dataset folder, or download the full dataset and run code to generate controlled dataset yourself. There are multiple configurations available to specify number of samples used, proportion of negative samples etc. in configuration.py.
# Code
- `main.py`: main entrance for correlation and classification experiment
- `workload_generator.py`: preprocessing logic. ppdb filtering and controlled dataset generation is implemented in this file.
- `configuration.py`: configuration
- `kintsch_exp.py`: main entrance for landmark experiment
- `analyzer`: helper class to perform analysis
# Usage
1. update `configuration.py` to specify: a) dataset location, b) experiment you want to run and c) model class you want to test
2. to run correlation or classification experiment, run `python main.py` (after you set proper configuration in configuration.py)
3. to run landmark experiment, run `python kintsch_exp.py`