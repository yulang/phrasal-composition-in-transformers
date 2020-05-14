import numpy as np
from scipy.stats.stats import pearsonr
from workload_generator import extract_embedding, extract_related_embedding
from configuration import BERT_VARIANTS
import torch.nn as nn
import torch


# concact hidden states to save to disc
def concact_hidden_states(acc_hidden, new_hidden):
    assert len(acc_hidden) == len(new_hidden)  # size should be n_layers + 1
    rst = []
    for layer_id in range(len(acc_hidden)):
        acc_tensor = acc_hidden[layer_id]  # of shape (batch_size, sequence_length, hidden_size)
        new_tensor = new_hidden[layer_id]
        acc_tensor = torch.cat((acc_tensor, new_tensor), 0)
        rst.append(acc_tensor)

    return rst


# helper func to validate dumping works
def validate_dumping_embeddings(analyzer):
    total_sent_no = len(analyzer.features)
    print("total number of sents = {}".format(total_sent_no))
    for sent_idx in range(total_sent_no):
        for layer_id in range(analyzer.n_layers):
            org = analyzer.lookup_embedding(sent_idx, layer_id).detach().numpy()
            restored = analyzer.lookup_embedding_with_dump(sent_idx, layer_id).detach().numpy()
            print(sent_idx, layer_id)
            np.testing.assert_array_equal(org, restored)
    print("pass validation!!")


def is_partial_word(token):
    # if token.startswith("##") or token.startswith("'") or token.startswith("-"):
    if token.startswith("##") or token.startswith("-"):
        return True
    return False


def is_partial_bigrams(token1, token2):
    if token1 == "'" and token2 == "s":
        return 1
    elif token1 == "," and token2 == "000":
        return 2
    else:
        return 0


def adjust_transformer_range(phrase_list, input_id_list, tokenizer, model_name):
    assert len(phrase_list) == len(input_id_list)
    phrase_pos = []

    for index, phrase_text in enumerate(phrase_list):
        sentence_ids = input_id_list[index]  # tensor of shape (batch_size, sequence_length)
        sentence_id_list = sentence_ids.squeeze().tolist()

        if model_name in BERT_VARIANTS:
            trial_encode = tokenizer.encode(phrase_text)
            phrase_id_str = " ".join(map(str, trial_encode[1:-1])) # remove cls and sep
        elif model_name in ['transxl']:
            trial_encode = tokenizer.encode(phrase_text, add_space_before_punct_symbol=True)
            phrase_id_str = " ".join(map(str, trial_encode)) # no cls and sep for xl like models
        elif model_name in ['xlnet']:
            trial_encode = tokenizer.encode(phrase_text)
            phrase_id_str = " ".join(map(str, trial_encode[:-2])) # remove cls and sep
        else:
            print("unsupported model {}".format(model_name))
            exit(1)
        phrase_id_count = len(phrase_id_str.split())
        sentence_id_str = " ".join(map(str, sentence_id_list))
        string_index = sentence_id_str.index(phrase_id_str)

        start_pos = len(sentence_id_str[:string_index].split())
        end_pos = start_pos + phrase_id_count
        phrase_pos.append((start_pos, end_pos))

    return phrase_pos


# to adjust range of phrase only input
def naive_adjust_phrase_range(phrase_pos, phrase_text, bert_features):
    assert len(phrase_pos) == len(bert_features)
    for sent_index in range(len(bert_features)):
        cur_tokens = bert_features[sent_index].tokens
        start_pos = phrase_pos[sent_index]
        size = len(cur_tokens)
        end_pos = size - 1
        phrase_pos[sent_index] = (start_pos, end_pos)
        reconstruct = " ".join(cur_tokens[start_pos: end_pos]).replace(" ##", "").replace(" ' ", "'").replace(" - ", "-")
        source_text = phrase_text[sent_index]
        if reconstruct != source_text:
            print(reconstruct)
            print(source_text)
    return phrase_pos


def adjust_phrase_range_v2(phrase_pos, phrase_text, bert_features):
    # handle tokenizer breaking the phrase and other words
    assert len(phrase_pos) == len(bert_features)
    assert len(phrase_text) == len(phrase_pos)
    for sent_index in range(len(bert_features)):
        cur_tokens = bert_features[sent_index].tokens
        start_pos = phrase_pos[sent_index]
        source_text = phrase_text[sent_index]
        phrase_word_counts = len(source_text.split())
        phrase_word_counts += source_text.count("'") + source_text.count("-")
        offset = 0

        # compute offset before start_pos
        for ind, token in enumerate(cur_tokens[:start_pos]):
            if is_partial_word(token):
                offset += 1
            elif is_partial_bigrams(token, cur_tokens[ind + 1]) != 0:
                offset += is_partial_bigrams(token, cur_tokens[ind + 1])

        # adjust start_pos
        while offset != 0:
            token = cur_tokens[start_pos]
            if not is_partial_word(token):
                offset -= 1
            start_pos += 1

        while is_partial_word(cur_tokens[start_pos]) or (is_partial_bigrams(cur_tokens[start_pos-1], cur_tokens[start_pos]) != 0) or (not source_text.startswith(cur_tokens[start_pos])):
            start_pos += 1

        end_pos = start_pos + phrase_word_counts
        offset = 0
        for token in cur_tokens[start_pos:end_pos]:
            if is_partial_word(token):
                offset += 1

        while offset != 0 or is_partial_word(cur_tokens[end_pos]):
            token = cur_tokens[end_pos]
            if not is_partial_word(token):
                offset -= 1
            end_pos += 1

        phrase_pos[sent_index] = (start_pos, end_pos)
        reconstruct = " ".join(cur_tokens[start_pos: end_pos]).replace(" ##", "").replace(" ' ", "'").replace(" - ", "-")
        if reconstruct != source_text:
            print(reconstruct)
            print(source_text)
    return phrase_pos


def adjust_phrase_range(phrase_pos, phrase_text, bert_features):
    # handle tokenizer breaking the phrase and other words
    assert len(phrase_pos) == len(bert_features)
    assert len(phrase_text) == len(phrase_pos)
    for sent_index in range(len(bert_features)):
        cur_tokens = bert_features[sent_index].tokens
        start_pos = phrase_pos[sent_index]
        source_text = phrase_text[sent_index]
        phrase_word_counts = len(source_text.split())

        if not source_text.startswith(cur_tokens[start_pos]):
            # adjust start position first
            start_pos += 1
            token = cur_tokens[start_pos]
            while not source_text.startswith(token):
                start_pos += 1
                token = cur_tokens[start_pos]

        # start position now is accurate
        end_pos = start_pos + phrase_word_counts
        reassemble_phrase = " ".join(cur_tokens[start_pos: end_pos])

        if reassemble_phrase != source_text:
            # what if '#' is not in the reassemble_phrase?
            reassemble_phrase = reassemble_phrase.replace(" ##", "")
            while (reassemble_phrase != source_text) and (len(reassemble_phrase) < len(source_text)):
                more_token = cur_tokens[end_pos]
                if more_token.startswith("##"):
                    more_token = more_token[2:]
                else:
                    more_token = " " + more_token
                end_pos += 1
                reassemble_phrase += more_token

        if reassemble_phrase != source_text:
            print(reassemble_phrase)
            print(source_text)
        phrase_pos[sent_index] = (start_pos, end_pos)

    return phrase_pos


def validate_embed_phrase(phrase_pos, phrase_text, bert_features):
    for sent_index in range(len(bert_features)):
        start, end = phrase_pos[sent_index]
        correct_phrase = phrase_text[sent_index]
        reassembled_phrase = " ".join(bert_features[sent_index].tokens[start:end])
        reassembled_phrase = reassembled_phrase.replace(" ##", "")
        if correct_phrase != reassembled_phrase:
            print(correct_phrase)
            print(reassembled_phrase)
    print("Pass sentence embedding validation")


def print_stats_by_layer(stats_by_layer, is_list=False, stat_type="cor"):
    # stat_type: either "cor" or "cosim"
    print("stats type: " + stat_type)
    title = "Layer\tMean\tDeviation"
    output_format = "{}\t{}\t{}"
    single_value_title = "Layer\tValue"
    single_value_output_format = "{}\t{}"

    if stat_type == "cor":
        out = open("coe.txt", "w")
        print("Correlation by token\n\n")
    elif stat_type == "cosim":
        out = open("cos_sim.txt", "w")
        print("\n\nCosine Similarity by token\n\n")
    else:
        exit(1)


    def print_one_token(token):
        print("token {}".format(token))
        out.write("token {}\n".format(token))
        if is_list:
            print(title)
            out.write(title+"\n")
            # list of coefficient. take average
            for layer_id, coe_dic in enumerate(stats_by_layer):
                coe_list = coe_dic[token]
                avg, std = np.mean(coe_list), np.std(coe_list)
                print(output_format.format(layer_id, avg, std))
                out.write(output_format.format(layer_id, avg, std) + "\n")
        else:
            print(single_value_title)
            out.write(single_value_title+"\n")
            for layer_id, coe_dic in enumerate(stats_by_layer):
                coe = coe_dic[token]
                print(single_value_output_format.format(layer_id, coe))
                out.write(single_value_output_format.format(layer_id, coe) + "\n")

    print_one_token("cls")
    print_one_token("pivot_token")
    print_one_token("avg_phrase")
    print_one_token("avg_all")
    print_one_token("sep")
    out.close()


def compute_cos_sim(src_emb, tgt_emb_list, normalized=True):
    cos_sim_list = []
    cos_sim = nn.CosineSimilarity(dim=0)
    for tgt_emb in tgt_emb_list:
        sim = cos_sim(src_emb, tgt_emb)
        if normalized:
            sim = (sim + 1) / 2.0
        cos_sim_list.append(sim.item())

    return cos_sim_list


def analyze_correlation_by_layer(analyzer, score_dic, phrase_pos, include_input_emb=False):
    if include_input_emb:
        n_layers = analyzer.n_layers + 1
    else:
        n_layers = analyzer.n_layers
    cos_sim_by_layer = [{"cls": [], "pivot_token": [], "sep": [], "avg_all": [], "avg_phrase": []} for _ in range(n_layers)]
    coe_by_layer = [{"cls": None, "pivot_token": None, "sep": None, "avg_all": None, "avg_phrase": None} for _ in range(n_layers)]
    target_score = []
    model_name = analyzer.model_name

    # assert len(phrase_pos) == len(analyzer.features)

    for source_phrase_index in score_dic:
        second_phrase_list = score_dic[source_phrase_index]
        # source_phrase_tokens = analyzer.lookup_sent_text(source_phrase_index)
        source_phrase_length = analyzer.lookup_sequence_length(source_phrase_index)
        src_start_pos, src_end_pos = phrase_pos[source_phrase_index]

        for layer_id in range(n_layers):
            source_embedding = analyzer.lookup_embedding(source_phrase_index, layer_id)
            out = extract_related_embedding(source_embedding, src_start_pos, src_end_pos, source_phrase_length,
                                            model_name)
            src_cls, src_pivot, src_sep, src_avg_all, src_avg_phrase = out
            # validate size of returned vectors
            assert src_avg_phrase.shape == src_cls.shape
            assert src_avg_all.shape == src_cls.shape

            trg_cls_list = []
            trg_pivot_list = []
            trg_sep_list = []
            trg_avg_phrase_list = []
            trg_avg_all_list = []

            for second_phrase_index, score in second_phrase_list:
                # target_phrase_tokens = analyzer.lookup_sent_text(second_phrase_index)
                target_phrase_length = analyzer.lookup_sequence_length(second_phrase_index)
                target_embedding = analyzer.lookup_embedding(second_phrase_index, layer_id)
                trg_start_pos, trg_end_pos = phrase_pos[second_phrase_index]
                # avoid adding same score 12 times
                if layer_id == n_layers - 1:
                    target_score += score,

                out = extract_related_embedding(target_embedding, trg_start_pos, trg_end_pos, target_phrase_length,
                                                model_name)
                trg_cls, trg_pivot, trg_sep, trg_avg_all, trg_avg_phrase = out

                assert trg_avg_phrase.shape == trg_cls.shape
                assert trg_avg_all.shape == trg_pivot.shape
                assert trg_sep.shape == trg_cls.shape

                trg_cls_list.append(trg_cls)
                trg_pivot_list.append(trg_pivot)
                trg_sep_list.append(trg_sep)
                trg_avg_phrase_list.append(trg_avg_phrase)
                trg_avg_all_list.append(trg_avg_all)

            cls_sim = compute_cos_sim(src_cls, trg_cls_list)
            pivot_sim = compute_cos_sim(src_pivot, trg_pivot_list)
            sep_sim = compute_cos_sim(src_sep, trg_sep_list)
            avg_phrase_sim = compute_cos_sim(src_avg_phrase, trg_avg_phrase_list)
            avg_all_sim = compute_cos_sim(src_avg_all, trg_avg_all_list)

            cos_sim_by_layer[layer_id]["cls"].extend(cls_sim)
            cos_sim_by_layer[layer_id]["pivot_token"].extend(pivot_sim)
            cos_sim_by_layer[layer_id]["sep"].extend(sep_sim)
            cos_sim_by_layer[layer_id]["avg_phrase"].extend(avg_phrase_sim)
            cos_sim_by_layer[layer_id]["avg_all"].extend(avg_all_sim)

    # compute correlation for each layer, each token
    for layer_id, cos_dic in enumerate(cos_sim_by_layer):
        assert len(cos_dic["cls"]) == len(target_score)
        cor, _ = pearsonr(cos_dic["cls"], target_score)
        coe_by_layer[layer_id]["cls"] = cor
        cor, _ = pearsonr(cos_dic["pivot_token"], target_score)
        coe_by_layer[layer_id]["pivot_token"] = cor
        cor, _ = pearsonr(cos_dic["sep"], target_score)
        coe_by_layer[layer_id]["sep"] = cor
        cor, _ = pearsonr(cos_dic["avg_phrase"], target_score)
        coe_by_layer[layer_id]["avg_phrase"] = cor
        cor, _ = pearsonr(cos_dic["avg_all"], target_score)
        coe_by_layer[layer_id]["avg_all"] = cor

    return coe_by_layer, cos_sim_by_layer, target_score


'''
    score_dic is generated from bird_preprocess. The dic is organized as
    {source_phrase_index_in_input_file : [(target_phrase_index, relatedness_score), ...]}
    analyze bird by source phrase group. generate average correlation across all source phrases.
'''
def analyze_bird(analyzer, score_dic, pivot_word_index=2):
    n_layers = analyzer.n_layers

    # in each of n_layers dic, value is a list of correlation coefficient (length equals to #source phrases)
    coe_by_layer = [{"cls": [], "pivot_token": [], "sep": []} for _ in range(n_layers)]

    for source_phrase_index in score_dic:
        second_phrase_list = score_dic[source_phrase_index]
        # include cls and sep. the length should be phrase length + 2
        source_phrase_tokens = analyzer.lookup_sent_text(source_phrase_index)

        relatedness_score_list = []
        source_embedding_by_layer = [{"cls": None, "pivot_token": None, "sep": None} for _ in range(n_layers)]
        target_embedding_by_layer = [{"cls": [], "pivot_token": [], "sep": []} for _ in range(n_layers)]

        for layer_id in range(n_layers):
            source_embedding_vec = analyzer.lookup_embedding(source_phrase_index, layer_id)
            assert source_embedding_vec.size(0) > 2
            source_embedding_by_layer[layer_id]["cls"] = source_embedding_vec[0]
            source_embedding_by_layer[layer_id]["pivot_token"] = source_embedding_vec[pivot_word_index]
            source_embedding_by_layer[layer_id]["sep"] = source_embedding_vec[len(source_phrase_tokens) - 1]

        for second_phrase_index, score in second_phrase_list:
            target_phrase_tokens = analyzer.lookup_sent_text(second_phrase_index)
            relatedness_score_list += score,

            for layer_id in range(n_layers):
                target_embedding_vec = analyzer.lookup_embedding(second_phrase_index, layer_id)
                target_embedding_by_layer[layer_id]["cls"].append(target_embedding_vec[0])
                target_embedding_by_layer[layer_id]["pivot_token"].append(target_embedding_vec[1])
                target_embedding_by_layer[layer_id]["sep"].append(target_embedding_vec[len(target_phrase_tokens) - 1])

        # compute correlation coefficient for current source phrase
        assert len(relatedness_score_list) == len(target_embedding_by_layer[0]["cls"])

        for layer_id in range(n_layers):
            # cls
            source_embedding_cls = source_embedding_by_layer[layer_id]["cls"]
            target_embeddings_cls = target_embedding_by_layer[layer_id]["cls"]
            cos_sim_cls = compute_cos_sim(source_embedding_cls, target_embeddings_cls)
            corr_coe_cls, _ = pearsonr(cos_sim_cls, relatedness_score_list)
            coe_by_layer[layer_id]["cls"].append(corr_coe_cls)
            # first_token
            source_embedding_ft = source_embedding_by_layer[layer_id]["pivot_token"]
            target_embeddings_ft = target_embedding_by_layer[layer_id]["pivot_token"]
            cos_sim_ft = compute_cos_sim(source_embedding_ft, target_embeddings_ft)
            corr_coe_ft, _ = pearsonr(cos_sim_ft, relatedness_score_list)
            coe_by_layer[layer_id]["pivot_token"].append(corr_coe_ft)
            # sep
            source_embedding_sep = source_embedding_by_layer[layer_id]["sep"]
            target_embeddings_sep = target_embedding_by_layer[layer_id]["sep"]
            cos_sim_sep = compute_cos_sim(source_embedding_sep, target_embeddings_sep)
            corr_coe_sep, _ = pearsonr(cos_sim_sep, relatedness_score_list)
            coe_by_layer[layer_id]["sep"].append(corr_coe_sep)

    # return correlation coefficient per layer
    return coe_by_layer