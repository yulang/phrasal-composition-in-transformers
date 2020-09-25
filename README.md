This repo contains datasets and code for *Assessing Phrasal Representation and Composition in Transformers*, by Lang Yu and Allyson Ettinger.

# Dependencies
The code is implemented with Huggingface's transformer v2.5.1. There are multiple API changes in their recent releases. And you may need to adjust related code if you are using a different version.
# Repo structure
- `src/` contains source code to generate dataset and perform analysis
- `datasets` contains dataset for landmark test from (Kintsch 2001).
# Dataset
- Similarity correlation dataset: As mentioned in the paper, the full dataset can be downloaded here: http://saifmohammad.com/WebPages/BiRD.html. Please refer to the original paper for details about the dataset.
- Paraphrase classification dataset: You can download ppdb-2.0-tldr from http://paraphrase.org.
- Landmark experiment dataset: included in the repo. Exact from the serie of papers by Kintsch.
# Code
- `main.py`
- `workload_generator.py`
- `configuration.py`
- `kintsch_exp.py`
- `analyzer`
# Usage
1. update `configuration.py` to specify: a) dataset location, b) experiment you want to run and c) model class you want to test
2. to run correlation or classification experiment, run `python main.py` (after you set proper configuration in configuration.py)
3. to run landmark experiment, run `python kintsch_exp.py`