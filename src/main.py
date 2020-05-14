import torch
from torch.utils.data import DataLoader, TensorDataset
from numpy import load, save
import os
import logging
import pdb
from transformers import *

from workload_generator import bird_preprocess, ppdb_preprocess, ppdb_exact_preprocess, stanfordsent_preprocess, \
    trivial_score_to_label, nontrivial_score_to_label, generate_classifier_workloads, \
    generate_stanford_classifier_workloads, embed_phrase_transformer, embed_phrase_and_truncate
from kintsch_exp import kintsch_preprocess, generate_kintsch_embeddings_transformer, evaluate_kintsch_embeddings
from configuration import config, BERT_VARIANTS
from utilities import adjust_transformer_range, analyze_correlation_by_layer, print_stats_by_layer, \
    concact_hidden_states
from analyzer import TransformerAnalyzer
import argparse



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def init_model(model_name):
    if model_name == "roberta":
        pretrained_name = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_name)
        model = RobertaModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    elif model_name == "xlnet":
        pretrained_name = 'xlnet-base-cased'
        tokenizer = XLNetTokenizer.from_pretrained(pretrained_name)
        model = XLNetModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    elif model_name == "xlmroberta":
        pretrained_name = 'xlm-roberta-base'
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_name)
        model = XLMRobertaModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    elif model_name == "distillbert":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
        model = DistilBertModel.from_pretrained('distilbert-base-cased', output_hidden_states=True, output_attentions=True)
    elif model_name == "bert":
        pretrained_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        model = BertModel.from_pretrained(pretrained_name, output_hidden_states=True, output_attentions=True)
    else:
        logger.error("unsupported model: {}".format(model_name))

    return tokenizer, model


def encode_input(tokenizer, sequence_list):
    input_id_list = []
    for sequence in sequence_list:
        input_ids = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True)])
        input_id_list.append(input_ids)

    assert len(input_id_list) == len(sequence_list)
    print("encoded {} sequences".format(len(input_id_list)))
    return input_id_list


def encode_padded_input(tokenizer, sequence_list, model_name):
    input_id_list, attention_mask, sequence_length = [], [], []
    pad_id = tokenizer.pad_token_id
    max_length = 250  # to speed up the process
    for sequence in sequence_list:
        if model_name in BERT_VARIANTS:
            encoded_rst = tokenizer.encode_plus(sequence, pad_to_max_length=True, add_special_tokens=True,
                                            max_length=max_length, return_overflowing_tokens=True)
        elif model_name in ['transxl']:
            logger.error("nor support transxl")
            exit(1)
            encoded_rst = tokenizer.encode_plus(sequence, return_overflowing_tokens=True, max_length=max_length,
                                                add_space_before_punct_symbol=True, pad_to_max_length=True)
        elif model_name in ['xlnet']:
            # the padding is on the left
            encoded_rst = tokenizer.encode_plus(sequence, return_overflowing_tokens=True, max_length=max_length,
                                                pad_to_max_length=True)
        else:
            logger.error("unsupported model class: {}".format(model_name))
            exit(1)

        if 'overflowing_tokens' in encoded_rst:
            # max_length is not long enough
            logger.error("overflew tokens in encoder")

        input_id_list.append(encoded_rst["input_ids"])
        attention_mask.append(encoded_rst["attention_mask"])
        sequence_length.append(max_length - encoded_rst["input_ids"].count(pad_id))

    return torch.tensor(input_id_list), torch.tensor(attention_mask), sequence_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=False)
    parser.add_argument("--batch_size", default=None, type=int, required=False)

    # ------- preprocessing -------#
    args = parser.parse_args()
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = 12
    if args.model is not None:
        model_name = args.model
    else:
        model_name = config["model_class"]
    logger.info("loading model")

    tokenizer, model = init_model(model_name)
    model_config = model.config
    n_layers, n_heads = model_config.num_hidden_layers, model_config.num_attention_heads
    random_seed = config["rand_seed"]
    dump_filename = "{}-dump-{}.npy".format(model_name, random_seed)
    dump_path = os.path.join(config["dump_path"], dump_filename)


    logger.info("current random seed: {}".format(random_seed))

    if config["workload"] == "bird":
        input_filename, score_dic, score_range, phrase_pos, phrase_text = bird_preprocess(config["BIRD_LOC"],
                                                                                          random_seed,
                                                                                          config["sample_size"],
                                                                                          normalize=config["normalize"])
        phrase_dic = score_dic
    elif config["workload"] == "ppdb":
        input_filename, score_dic, score_range, phrase_pos, phrase_text, samples_dic = \
            ppdb_preprocess(config["PPDB_LOC"], random_seed, config["sample_size"],
                            negative_sampling_mode=config["negative_sample_mode"],
                            overlap_threshold=config["overlap_threshold"])
        phrase_dic = score_dic
    elif config["workload"] == "ppdb_exact":
        input_filename, exact_label_dic, phrase_pos, phrase_text = ppdb_exact_preprocess(config["PPDB_LOC"],
                                                                                         random_seed,
                                                                                         config["sample_size"])
        phrase_dic = exact_label_dic
    elif config["workload"] == "stanford_sent":
        input_filename, phrase_pos, phrase_text, phrase_labels, phrase_scores = stanfordsent_preprocess(random_seed,
                                                                                                        config[
                                                                                                            "sample_size"])
        # TODO embed in sents not support
        phrase_dic = None
    elif config["workload"] == "kintsch":
        input_filename, landmark_samples, inference_samples, phrase_pos, phrase_text = kintsch_preprocess(random_seed)
        # TODO embed in sents not support
        phrase_dic = None
    else:
        print("unsupport workload " + config["workload"])
        exit(1)

    logger.info("current configuration: {}".format(config))

    if config["embed_in_sent"]:
        logger.info("Embedding phrase in wiki text")
        if config["workload"] == "ppdb_exact":
            logger.info("Before truncating: {}".format(len(phrase_text)))
            sentence_texts, phrase_text, exact_label_dic = embed_phrase_and_truncate(phrase_dic, phrase_text, config["TEXT_CORPUS"])
            logger.info("After truncating: {}".format(len(sentence_texts)))
        else:
            sentence_texts = embed_phrase_transformer(phrase_dic, phrase_text, config["TEXT_CORPUS"])

        sents_loc = "./embedded_sents_" + random_seed + ".txt"
        sent_out = open(sents_loc, "w")
        for sentence in sentence_texts:
            sent_out.write(sentence)
        sent_out.close()

    logger.info("encoding inputs")
    if config["embed_in_sent"]:
        input_id_list, attention_mask_list, phrase_length_list = encode_padded_input(tokenizer, sentence_texts, model_name)
    else:
        input_id_list, attention_mask_list, phrase_length_list = encode_padded_input(tokenizer, phrase_text, model_name)
    eval_data = TensorDataset(input_id_list, attention_mask_list)
    data_loader = DataLoader(eval_data, batch_size=batch_size)

    logger.info("adjusting phrase position & genreating label dic")
    phrase_pos = adjust_transformer_range(phrase_text, input_id_list, tokenizer, model_name)

    if config["classification"] and (config["workload"] in ["bird", "ppdb"]):
        # generate label dic for classification task
        if config["negative_sample_mode"] is None:
            label_dic = nontrivial_score_to_label(score_dic, score_range)
        else:
            label_dic = trivial_score_to_label(score_dic)

    #----------------------------- evaluation -------------------------------#
    logger.info("evaluating model")
    model.eval()

    assert os.path.exists(dump_path) is False

    dump_write_handler = open(dump_path, "ab")
    accumulated_hidden_states = None
    cached_count = 0

    for input_ids, input_mask in data_loader:
        if model_name in ["roberta", "bert", "xlmroberta"]:
            last_hidden_state, pooler_output, hidden_states, attentions = model(input_ids, attention_mask=input_mask)
        elif model_name in ["distillbert"]:
            last_hidden_state, hidden_states, attentions = model(input_ids, attention_mask=input_mask)
        elif model_name in ['xlnet']:
            last_hidden_state, hidden_states, attentions = model(input_ids, attention_mask=input_mask)
        else:
            logger.error("unsupported model: {}".format(model_name))
            exit(1)

        if accumulated_hidden_states is None:
            accumulated_hidden_states = list(hidden_states)
        else:
            accumulated_hidden_states = concact_hidden_states(accumulated_hidden_states, hidden_states)
        cached_count += 1

        if cached_count == config["dump_every"]:
            save(dump_write_handler, accumulated_hidden_states)  # note: dump accumulated hidden states
            cached_count = 0
            accumulated_hidden_states = None

    if cached_count != 0:
        # dump remaining segments
        save(dump_write_handler, accumulated_hidden_states)
    logger.info("dumping segment size: {} samples per segment".format(batch_size * config["dump_every"]))

    dump_write_handler.close()

    logger.info("working on downstream task")
    analyzer = TransformerAnalyzer(dump_path, n_layers, phrase_text, phrase_text, phrase_length_list, model_name,
                                   config["include_input_emb"])
    if config["workload"] == "kintsch":
        logger.info("writing out kintsch embeddings")
        out_embedding_dir = os.path.join(config["EMBEDDING_OUT_LOC"], model_name)
        if os.path.exists(out_embedding_dir) is False:
            os.mkdir(out_embedding_dir)

        dump_read_handler = open(dump_path, "rb")
        generate_kintsch_embeddings_transformer(dump_read_handler, out_embedding_dir, phrase_pos, phrase_length_list,
                                                landmark_samples, inference_samples, n_layers,
                                                config["include_input_emb"])
        dump_read_handler.close()

        logger.info("evaluating kintsch embeddings")
        evaluate_kintsch_embeddings(os.path.join(out_embedding_dir, "kintsch"), landmark_samples, inference_samples,
                                    n_layers, config["include_input_emb"])
    elif config["workload"] in ["bird", "ppdb"]:
        if config["correlation"]:
            logger.info("analyzing correlation...")
            coe_by_layer, cos_sim_by_layer, target_score = analyze_correlation_by_layer(analyzer, score_dic, phrase_pos,
                                                                                        config["include_input_emb"])
            print_stats_by_layer(coe_by_layer, is_list=False, stat_type="cor")
            analyzer.reset_handler()

        if config["classification"]:
            logger.info("generating classification workloads...")
            generate_classifier_workloads(analyzer, config, random_seed, phrase_text, label_dic, phrase_pos,
                                          config["include_input_emb"])
    elif config["workload"] == "ppdb_exact":
        generate_classifier_workloads(analyzer, config, random_seed, phrase_text, exact_label_dic, phrase_pos,
                                      config["include_input_emb"])
    elif config["workload"] == "stanford_sent":
        generate_stanford_classifier_workloads(analyzer, config, random_seed, phrase_text, phrase_labels, phrase_pos,
                                               config["include_input_emb"])
    else:
        logger.error("unsupport task {}".format(config["workload"]))


if __name__ == "__main__":
    main()