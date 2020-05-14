from workload_generator import extract_related_embedding, init_handler_dic, write_out_embedding, \
    extract_related_embedding
import os
from numpy import load
import torch.nn as nn
import torch
import pdb
from configuration import config

LANDMARK_EXP_COUNT = 4 # four samples each


def load_kintsch():
    handler = open(config["KINTSCH_LOC"], "r")
    landmark_samples = []  # [[phrase1, phrase2, landmark1, landmark2], ...] and phrase1 should be closer to landmark1
    inference_samples = [] # [[sentence, sentence1, sentence2], ...]

    # loading landmark examples
    for _ in range(LANDMARK_EXP_COUNT):
        cur_sample = []
        for _ in range(4):
            line = handler.readline()
            line = line.rstrip()
            cur_sample.append(line)
        _ = handler.readline()  # skip source comment
        landmark_samples.append(cur_sample)

    while True:
        cur_sample = []
        source_sent = handler.readline().rstrip()
        if len(source_sent) == 0:
            break
        cur_sample.append(source_sent)
        pos_sent = handler.readline().rstrip().replace(" ##pos", "")
        neg_sent = handler.readline().rstrip().replace(" ##neg", "")
        cur_sample.append(pos_sent)
        cur_sample.append(neg_sent)
        inference_samples.append(cur_sample)
        _ = handler.readline() # skip source comment

    return landmark_samples, inference_samples


def kintsch_preprocess(random_seed):
    landmark_samples, inference_samples = load_kintsch()
    output_file_name = "./kintsch_" + random_seed + ".txt"
    phrase_text = []
    output_file = open(output_file_name, "w")

    for sample_group in landmark_samples:
        for sample in sample_group:
            output_file.write(sample + "\n")
            phrase_text.append(sample)

    for sentence_group in inference_samples:
        for sentence in sentence_group:
            output_file.write(sentence + "\n")
            phrase_text.append(sentence)

    phrase_pos = [1 for _ in range(len(phrase_text))]
    return output_file_name, landmark_samples, inference_samples, phrase_pos, phrase_text


def generate_kintsch_embeddings(analyzer, out_parent_dir, phrase_pos, landmark_samples, inference_samples):
    assert len(phrase_pos) == len(analyzer.features)
    assert len(landmark_samples) * 4 + len(inference_samples) * 3 == len(phrase_pos)
    assert LANDMARK_EXP_COUNT == len(landmark_samples)

    embedding_folder = os.path.join(out_parent_dir, "kintsch")
    n_layers = analyzer.n_layers
    output_handlers_by_layer = [{"cls": None, "head_token": None, "sep": None, "avg_all": None, "avg_phrase": []} for _
                                in range(n_layers)]

    os.mkdir(embedding_folder)
    init_handler_dic(n_layers, embedding_folder, output_handlers_by_layer)

    raw_input_handler = open(os.path.join(embedding_folder, "input.txt"), "w")
    for sample in landmark_samples:
        for phrase in sample:
            raw_input_handler.write(phrase + "\n")

    for sample in inference_samples:
        for sentence in sample:
            raw_input_handler.write(sentence + "\n")

    raw_input_handler.close()

    phrase_pos_handler = open(os.path.join(embedding_folder, "pos.txt"), "w")
    phrase_pos_handler.write(str(phrase_pos))
    phrase_pos_handler.close()

    for phrase_index, phrase_pos in enumerate(phrase_pos):
        phrase_tokens = analyzer.lookup_sent_text(phrase_index)
        start_pos, end_pos = phrase_pos
        for layer_id in range(n_layers):
            raw_embedding = analyzer.lookup_embedding(phrase_index, layer_id)
            related_embeddings = extract_related_embedding(raw_embedding, start_pos, end_pos, phrase_tokens)
            write_out_embedding(related_embeddings, layer_id, output_handlers_by_layer)


    # closing
    for handler_dic in output_handlers_by_layer:
        for _, handler in handler_dic.items():
            handler.close()


def generate_kintsch_embeddings_transformer(dump_read_handler, out_parent_dir, phrase_pos_list, phrase_length_list,
                                            landmark_samples, inference_samples, n_layers,
                                            include_input_emb=False):
    # copy from generate_kintsch_embeddings
    assert len(phrase_length_list) == len(phrase_pos_list)
    assert len(phrase_pos_list) == len(landmark_samples) * 4 + len(inference_samples) * 3
    assert LANDMARK_EXP_COUNT == len(landmark_samples)

    if include_input_emb:
        n_layers = n_layers + 1

    embedding_folder = os.path.join(out_parent_dir, "kintsch")

    output_handlers_by_layer = [{"cls": None, "head_token": None, "sep": None, "avg_all": None, "avg_phrase": []} for _
                                in range(n_layers)]

    if os.path.exists(embedding_folder) is False:
        os.mkdir(embedding_folder)

    init_handler_dic(n_layers, embedding_folder, output_handlers_by_layer)

    raw_input_handler = open(os.path.join(embedding_folder, "input.txt"), "w")
    for sample in landmark_samples:
        for phrase in sample:
            raw_input_handler.write(phrase + "\n")

    for sample in inference_samples:
        for sentence in sample:
            raw_input_handler.write(sentence + "\n")

    raw_input_handler.close()

    phrase_pos_handler = open(os.path.join(embedding_folder, "pos.txt"), "w")
    phrase_pos_handler.write(str(phrase_pos_list))
    phrase_pos_handler.close()
    # end copying

    batch_pt = None
    batch_size = None
    loaded_segment = None
    for phrase_index, phrase_pos in enumerate(phrase_pos_list):
        sequence_token_count = phrase_length_list[phrase_index]
        start_pos, end_pos = phrase_pos
        if (batch_pt is None) or (loaded_segment is None) or (batch_pt == batch_size):
            # load one more segment
            batch_pt = 0
            loaded_segment = load(dump_read_handler, allow_pickle=True)
            batch_size = loaded_segment[0].shape[0]
        for layer_id in range(n_layers):
            # hidden_embedding = loaded_segment[layer_id + 1][batch_pt]  # first element in the tuple is input embedding
            hidden_embedding = loaded_segment[layer_id][batch_pt]
            related_embeddings = extract_related_embedding(hidden_embedding, start_pos, end_pos,
                                                                       sequence_token_count)
            write_out_embedding(related_embeddings, layer_id, output_handlers_by_layer)
        batch_pt += 1

    # closing
    for handler_dic in output_handlers_by_layer:
        for _, handler in handler_dic.items():
            handler.close()


def evaluate_kintsch_embeddings(embedding_loc, landmark_samples, inference_samples, n_layers, include_input_emb=False):
    if include_input_emb:
        n_layers = n_layers + 1

    output_handlers_by_layer = [{"cls": None, "head_token": None, "sep": None, "avg_all": None, "avg_phrase": None} for _
                                in range(n_layers)]

    init_handler_dic(n_layers, embedding_loc, output_handlers_by_layer, "rb")
    cos_sim = nn.CosineSimilarity(dim=0)

    landmark_total_count = 0
    landmark_hit_by_layer = [{"cls": 0, "head_token": 0, "sep": 0, "avg_all": 0, "avg_phrase": 0} for _
                                in range(n_layers)]

    for samples in landmark_samples:
        landmark_total_count += 2  # two correct pairs per sample list
        for layer_id in range(n_layers):
            cur_handler_dic = output_handlers_by_layer[layer_id]
            for token, handler in cur_handler_dic.items():
                phrase1 = torch.from_numpy(load(handler, allow_pickle=True))
                phrase2 = torch.from_numpy(load(handler, allow_pickle=True))
                landmark1 = torch.from_numpy(load(handler, allow_pickle=True))
                landmark2 = torch.from_numpy(load(handler, allow_pickle=True))
                sim11 = cos_sim(landmark1, phrase1)
                sim12 = cos_sim(landmark1, phrase2)
                sim21 = cos_sim(landmark2, phrase1)
                sim22 = cos_sim(landmark2, phrase2)
                if sim11 > sim12:
                    landmark_hit_by_layer[layer_id][token] += 1
                if sim21 < sim22:
                    landmark_hit_by_layer[layer_id][token] += 1


    # note: for inference samples, only cls is tested
    inference_total_count = 0
    inference_cls_hit_by_layer = [0 for _ in range(n_layers)]
    for samples in inference_samples:
        inference_total_count += 1
        for layer_id in range(n_layers):
            cur_handler_dic = output_handlers_by_layer[layer_id]
            handler = cur_handler_dic["cls"]
            source_sent = torch.from_numpy(load(handler, allow_pickle=True))
            pos_sent = torch.from_numpy(load(handler, allow_pickle=True))
            neg_sent = torch.from_numpy(load(handler, allow_pickle=True))
            sim_pos = cos_sim(pos_sent, source_sent)
            sim_neg = cos_sim(neg_sent, source_sent)
            if sim_pos > sim_neg:
                inference_cls_hit_by_layer[layer_id] += 1

    # normalize hits
    for hit_dic in landmark_hit_by_layer:
        for token, hit in hit_dic.items():
            hit_dic[token] /= float(landmark_total_count)

    inference_cls_hit_by_layer = [float(x) / inference_total_count for x in inference_cls_hit_by_layer]

    # closing
    for handler_dic in output_handlers_by_layer:
        for _, handler in handler_dic.items():
            handler.close()

    # print out results
    tokens = ["cls", "head_token", "sep", "avg_all", "avg_phrase"]
    print("landmark experiments")
    for token in tokens:
        print("---{}---".format(token))
        for dic in landmark_hit_by_layer:
            print(dic[token])
        print("")

    print("inference experiments")
    for prec in inference_cls_hit_by_layer:
        print(prec)
    return landmark_hit_by_layer, inference_cls_hit_by_layer


def verbose_evaluate_kintsch_embeddings(embedding_loc, landmark_samples, inference_samples, n_layers, include_input_emb=False):
    if include_input_emb:
        n_layers = n_layers + 1
    output_handlers_by_layer = [{"cls": None, "head_token": None, "sep": None, "avg_all": None, "avg_phrase": []} for _
                                in range(n_layers)]

    init_handler_dic(n_layers, embedding_loc, output_handlers_by_layer, "rb")
    cos_sim = nn.CosineSimilarity(dim=0)

    print("\nlandmark experiments:")
    print("\tphrase 1\tphrase 2")
    format_str = "landmark 1\t{}\t{}\nlandmark 2\t{}\t{}"
    for samples in landmark_samples:
        print("current samples: {}".format(samples))
        for layer_id in range(n_layers):
            print("--- layer {} ---".format(layer_id))
            cur_handler_dic = output_handlers_by_layer[layer_id]
            for token, handler in cur_handler_dic.items():
                print("-- {} --".format(token))
                phrase1 = torch.from_numpy(load(handler, allow_pickle=True))
                phrase2 = torch.from_numpy(load(handler, allow_pickle=True))
                landmark1 = torch.from_numpy(load(handler, allow_pickle=True))
                landmark2 = torch.from_numpy(load(handler, allow_pickle=True))
                sim11 = cos_sim(landmark1, phrase1)
                sim12 = cos_sim(landmark1, phrase2)
                sim21 = cos_sim(landmark2, phrase1)
                sim22 = cos_sim(landmark2, phrase2)
                print(format_str.format(sim11, sim12, sim21, sim22))

    print("\ninference experiments:")
    print("\tpos sentence\tneg sentence")
    format_str = "source sentence\t{}\t{}"
    for samples in inference_samples:
        print("current samples: {}".format(samples))
        for layer_id in range(n_layers):
            print("--- layer {} ---".format(layer_id))
            cur_handler_dic = output_handlers_by_layer[layer_id]
            print("-- cls --")
            handler = cur_handler_dic["cls"]
            source_sent = torch.from_numpy(load(handler, allow_pickle=True))
            pos_sent = torch.from_numpy(load(handler, allow_pickle=True))
            neg_sent = torch.from_numpy(load(handler, allow_pickle=True))
            sim_pos = cos_sim(pos_sent, source_sent)
            sim_neg = cos_sim(neg_sent, source_sent)
            print(format_str.format(sim_pos, sim_neg))

    # closing
    for handler_dic in output_handlers_by_layer:
        for _, handler in handler_dic.items():
            handler.close()


def main():
    landmark_samples, inference_samples = load_kintsch()
    evaluate_kintsch_embeddings("/home-nfs/langyu/data_folder/bert-embeddings/kintsch", landmark_samples, inference_samples, 12)


if __name__ == "__main__":
    main()