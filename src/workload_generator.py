import random
import os
import torch
import numpy as np
import re
import itertools
from sentiment_exp import construct_sentiment_dic
from configuration import BERT_VARIANTS
from collections import Counter, OrderedDict
from statistics import mean
from itertools import islice


# return true if current pair needs to be skipped
def ppdb_filterer(phrase1, phrase2, tag, overlap_threshold):
    target_len = 2
    target_tag = None
    # overlap_threshold = 0.5
    def is_trivial_word(word):
        skip_words = {"rather", "very", "hundred", "thousand", "million", "year", "years", "yes", "hundreds", "thousands"}
        if word.isnumeric():
            return True
        if word in skip_words:
            return True
        return False
    words1, words2 = phrase1.split(), phrase2.split()

    # print(overlap_ratio)
    if len(words1) != target_len:
        return True
    for word in words1 + words2:
        if is_trivial_word(word):
            return True
    if (target_tag is not None) and (tag == target_tag):
        return True
    if overlap_threshold is not None:
        word_overlap = set(words1).intersection(set(words2))
        overlap_ratio = float(len(word_overlap)) / min(len(words1), len(words2))
        if overlap_ratio >= overlap_threshold:
            return True
    return False


# filter out trivial phrases. return true if the phrase should not be included
def stanford_phrase_filter(phrase):
    if not bool(re.search('[a-zA-Z]', phrase)):
        # no alphabet characters
        return True
    if len(phrase) < 4:
        return True

    return False


# write out bert input. same interface as bird_preprocess, ppdb_preprocess
def stanfordsent_preprocess(random_seed, total_phrase_count=None):
    output_phrase_file_name = "./stanford_phrases_{}.txt".format(random_seed)
    output_file = open(output_phrase_file_name, "w")
    id2prhase, id2sent_score, id2sent_label, phrase2id = construct_sentiment_dic()
    phrase_text = []
    labels = []
    scores = []
    output_line_count = 0
    for id, phrase in id2prhase.items():
        if (total_phrase_count is not None) and (output_line_count >= total_phrase_count):
            break
        if stanford_phrase_filter(phrase):
            continue
        labels.append(id2sent_label[id])
        scores.append(id2sent_score[id])
        phrase_text.append(phrase)
        output_file.write(phrase + "\n")
        output_line_count += 1

    phrase_pos = [1 for _ in range(len(phrase_text))]

    assert len(phrase_text) == len(labels)
    assert len(labels) == len(scores)
    return output_phrase_file_name, phrase_pos, phrase_text, labels, scores


def truncate_samples_dic(samples_dic):
    max_target_phrase_count = 5  # limit max target phrases per source phrase to solve segment size issue.
    for source, dic in samples_dic.items():
        samples_dic[source] = dict(islice(dic.items(), max_target_phrase_count))


def is_ppdb_illegal_pair(phrase1, phrase2):
    special_illegal_words = ['enhancing implementation', 'give discharge',
                             'welcomes information', 'giving discharge',
                             'developing cross-straits',
                             'everybody ok', "'il go",
                             'grant discharge']
    special_illegal_tokens = ['-rrb-', '/', ':', '.', "!", "-"]

    if phrase1.startswith("<") or (phrase1 in special_illegal_words) or (phrase2 in special_illegal_words) or (
                any(x in phrase1 for x in special_illegal_tokens)) or (any(x in phrase2 for x in special_illegal_tokens)):
        return True
    return False


# write out bert input. same interface as bird_preprocess
# negative_sampling_mode option: one_per_source, half_neg
def ppdb_preprocess(path, random_seed, total_phrase_count=None, negative_sampling_mode=None, overlap_threshold=None):
    assert negative_sampling_mode in [None, "one_per_source", "half_neg", "all_neg"]
    samples_dic, source_phrase_by_tag, tag_set, phrase_set = ppdb_phrase_extract(path, total_phrase_count, ppdb_filterer, overlap_threshold)
    phrase_list = list(phrase_set)
    # output_phrase_file_name = "./ppdb_phrases.txt"
    output_phrase_file_name = "./ppdb_phrases_" + random_seed + ".txt"
    output_line_count = 0
    phrase_text = []
    # source_phrase_index in bert input file, {source_phrase_index: [(target_phrase_index1, score), (target_phrase_index2, score)]}
    score_dic = OrderedDict()
    score_range = [None, None]
    output_file = open(output_phrase_file_name, "w")

    truncate_samples_dic(samples_dic)

    if negative_sampling_mode == "all_neg":
        # compute pair count
        for phrase_index in range(total_phrase_count):
            phrase = random.choice(phrase_list)
            output_file.write(phrase + "\n")
            phrase_text.append(phrase)

            if phrase_index % 2 == 0:
                # source phrsae
                score_dic[phrase_index] = []
            else:
                # target phrase
                score_dic[phrase_index - 1].append((phrase_index, 0.0))
        output_line_count = total_phrase_count
        score_range = [0.0, 0.0]
    else:
        for source_phrase, dic in samples_dic.items():
            if (total_phrase_count is not None) and (output_line_count >= total_phrase_count):
                break
            # write out source phrase
            score_dic[output_line_count] = []
            cur_score_dic_list = score_dic[output_line_count]
            output_file.write(source_phrase + "\n")
            phrase_text.append(source_phrase)
            output_line_count += 1

            #write out target phrase
            # for (tag, target_phrase), score_list in dic.items():
            for target_phrase, score_list in dic.items():
                score = mean(score_list)

                # update score range
                if (score_range[0] is None) and (score_range[1] is None):
                    score_range[0] = score
                    score_range[1] = score
                elif score < score_range[0]:
                    score_range[0] = score
                elif score > score_range[1]:
                    score_range[1] = score

                output_file.write(target_phrase + "\n")
                phrase_text.append(target_phrase)
                cur_score_dic_list.append((output_line_count, score))
                output_line_count += 1
                if negative_sampling_mode == "half_neg":
                    neg_sample = random.choice(phrase_list)
                    output_file.write(neg_sample + "\n")
                    phrase_text.append(neg_sample)
                    cur_score_dic_list.append((output_line_count, 0.0))
                    output_line_count += 1

            if negative_sampling_mode == "one_per_source":
                neg_sample = random.choice(phrase_list)
                output_file.write(neg_sample + "\n")
                phrase_text.append(neg_sample)
                cur_score_dic_list.append((output_line_count, 0.0))
                output_line_count += 1

    phrase_pos = [1 for _ in range(output_line_count)]
    output_file.close()
    return output_phrase_file_name, score_dic, score_range, phrase_pos, phrase_text, samples_dic


def ppdb_phrase_extract(path, sample_size=None, filter=ppdb_filterer, overlap_threshold=None):
    sample_count = 0
    source_phrase_by_tag = {} # {tag: [phrase11, phrase12, ...]} where phrase11 etc are the keys in the samples dic
    samples_dic = {}  # organize as {phrase1: {(tag, phrase2): [score1, score2...]}, ...}}
    tag_set = set()
    phrase_set = set()
    with open(path, "r") as f:
        for line_count, line in enumerate(f):
            segments = line.strip().split(" ||| ")
            tag, phrase1, phrase2 = segments[0], segments[1], segments[2]

            if tag not in tag_set:
                source_phrase_by_tag[tag] = set()
                tag_set.add(tag)
            score = float(segments[3].split()[0].split("=")[1])
            if is_ppdb_illegal_pair(phrase1, phrase2):
                # url, skip
                continue

            if (filter is not None) and filter(phrase1, phrase2, tag, overlap_threshold) is True:
                # filter out phrases that are not of target length
                continue

            phrase_set.add(phrase1)
            phrase_set.add(phrase2)

            if phrase1 in samples_dic:
                # second_key = (tag, phrase2)
                second_key = phrase2  # for now, don't differentiate tag for the same phrase
                source_phrase_by_tag[tag].add(phrase1)
                if second_key in samples_dic[phrase1]:
                    samples_dic[phrase1][second_key].append(score)
                else:
                    samples_dic[phrase1][second_key] = [score]
            elif phrase2 in samples_dic:
                # reverse the source/target relation
                # second_key = (tag, phrase1)
                second_key = phrase1
                source_phrase_by_tag[tag].add(phrase2)
                if second_key in samples_dic[phrase2]:
                    samples_dic[phrase2][second_key].append(score)
                else:
                    samples_dic[phrase2][second_key] = [score]
            else:
                # never met this pair before
                samples_dic[phrase1] = {}
                # samples_dic[phrase1][(tag, phrase2)] = [score]
                samples_dic[phrase1][phrase2] = [score]
                source_phrase_by_tag[tag].add(phrase1)

            sample_count += 1
            if (sample_size is not None) and (sample_count >= sample_size):
                break

        return samples_dic, source_phrase_by_tag, tag_set, phrase_set


def ppdb_exact_preprocess(path, random_seed, sample_size=None):
    phrase_list = ppdb_overlap_control(path)
    output_phrase_file_name = "./ppdb_phrases_exact_" + random_seed + ".txt"
    output_line_count = 0
    phrase_text = []
    # source_phrase_index in bert input file,
    # {source_phrase_index: [(target_phrase_index1, label1), (target_phrase_index2, label2)]}
    label_dic = OrderedDict()
    output_file = open(output_phrase_file_name, "w")

    for sample_list in phrase_list:
        if (sample_size is not None) and (output_line_count >= sample_size):
            break
        source_item, rest_list = sample_list[0], sample_list[1:]
        source_phrase, label = source_item
        assert label == 'pos'
        label_dic[output_line_count] = []
        cur_label_dic_list = label_dic[output_line_count]
        output_file.write(source_phrase + "\n")
        phrase_text.append(source_phrase)
        output_line_count += 1

        for item in rest_list:
            phrase, label = item
            cur_label_dic_list.append((output_line_count, label))
            output_file.write(phrase + "\n")
            phrase_text.append(phrase)
            output_line_count += 1

    phrase_pos = [1 for _ in range(len(phrase_text))]
    return output_phrase_file_name, label_dic, phrase_pos, phrase_text


def test_valid_negative_sample(phrase1, phrase2, phrase_dic):
    # test each word in phrase whether it fits as a negative sample
    words1, words2 = phrase1.split(), phrase2.split()
    for word in words1 + words2:
        if word in phrase_dic:
            # landmark word is in the phrase_dic
            if len(phrase_dic[word]["neg"]) == 0 and (phrase1 not in phrase_dic[word]["pos"]) and (phrase2 not in phrase_dic[word]["pos"]):
                # only add phrase to negative samples if negative samples are empty
                if word in phrase1:
                    phrase_dic[word]["neg"].add(phrase1)
                if word in phrase2:
                    phrase_dic[word]["neg"].add(phrase2)


def process_phrase_dic(phrase_dic):
    result_list = []  # will be organized as [[(phrase, pos), ..., (phrase, neg)], []]
    for landmark_word, sample_dic in phrase_dic.items():
        if len(sample_dic["pos"]) > 0 and len(sample_dic["neg"]) > 0:
            # limit pos and neg samples so that downstream task can have more variant source phrases
            cur_element = []
            cur_element += [(x, "pos") for x in list(sample_dic["pos"])[:3]]
            cur_element += [(x, "neg") for x in list(sample_dic["neg"])[:3]]
            result_list.append(cur_element)
    return result_list


def ppdb_overlap_control(path):
    ppdb_handler = open(path, "r")
    phrase_dic = {}
    illegal_words = ["and", "he", "she", "it", "has", "have", "to", "of", "him", "her", "they", "them", "is", "are", "am"]

    def filter(words):
        for word in words:
            if word.isalpha() is False:
                return True
            if word in illegal_words:
                return True
        return False

    for line_count, line in enumerate(ppdb_handler):
        segments = line.strip().split(" ||| ")
        tag, phrase1, phrase2 = segments[0], segments[1], segments[2]
        # if phrase1.startswith("<"):
        if is_ppdb_illegal_pair(phrase1, phrase2):
            continue
        words1, words2 = phrase1.split(), phrase2.split()
        if len(words1) != 2 or len(words2) != 2:
            continue

        if filter(words1) or filter(words2):
            # contain illegal word. skip
            continue

        # construct positive pairs if current pairs share exactly 1 word
        overlapping = list(set(words1).intersection(set(words2)))
        if len(overlapping) == 1:
            # valid pairs
            landmark_word = overlapping[0]
            if landmark_word in phrase_dic:
                if ((phrase1 in phrase_dic[landmark_word]["pos"]) or (phrase2 in phrase_dic[landmark_word]["pos"])) and \
                        ((phrase1 not in phrase_dic[landmark_word]["neg"]) and (
                                phrase2 not in phrase_dic[landmark_word]["neg"])):
                    # should be positive samples
                    phrase_dic[landmark_word]["pos"].add(phrase1)
                    phrase_dic[landmark_word]["pos"].add(phrase2)
                elif ((phrase1 in phrase_dic[landmark_word]["neg"]) or (phrase2 in phrase_dic[landmark_word]["neg"])) and \
                        ((phrase1 not in phrase_dic[landmark_word]["pos"]) and (
                                phrase2 not in phrase_dic[landmark_word]["pos"])):
                    # should be negative samples
                    phrase_dic[landmark_word]["neg"].add(phrase1)
                    phrase_dic[landmark_word]["neg"].add(phrase2)
            else:
                phrase_dic[landmark_word] = {"pos": {phrase1, phrase2}, "neg": set()}

        # test if either of the current pairs can be used as negative samples
        # UPDATE: maybe should be more strict on adding negative samples?
        test_valid_negative_sample(phrase1, phrase2, phrase_dic)

    return process_phrase_dic(phrase_dic)


'''
input:
    - path to the BiRD file
    - number of __source__ term to be read 
        (one source term can have multiple target terms with varying relatedness scores). negative number will read the whole file
    
output:
    - path of the output sentences file (input file to bert) 
'''
def bird_preprocess(path, random_seed, sample_size=-1, normalize=False):
    # turns out normalization hurts the performance
    cur_source_phrase = None
    source_count = 0
    cur_source_index = 0
    output_line_count = 0
    phrase_text = []  # still need original phrase text to find the corresponding position in bert's token (could be splitted by tokenizer...)

    if normalize:
        print("Normalize input")
        # output_sentence_file_name = "./bird_phrases_normalized.txt"
        output_sentence_file_name = "./bird_phrases_normalized_" + random_seed + ".txt"
    else:
        # output_sentence_file_name = "./bird_phrases.txt"
        output_sentence_file_name = "./bird_phrases_" + random_seed + ".txt"
    output_file = open(output_sentence_file_name, "w")

    score_dic = OrderedDict()  # source_phrase_index (in bert input file)
    score_range = [None, None]
    bird_handler = open(path, "r")
    for line_no, line in enumerate(bird_handler):
        if line_no == 0:
            # skip header
            continue
        words = line.rstrip().split("\t")
        source_phrase = words[1]
        if source_phrase != cur_source_phrase:
            cur_source_phrase = source_phrase
            if source_count == sample_size:
                break
            source_count += 1
            cur_source_index = output_line_count
            if normalize:
                source_phrase = source_phrase.capitalize() + "."
            output_file.write(source_phrase + "\n")
            phrase_text.append(source_phrase)
            assert cur_source_index not in score_dic
            score_dic[cur_source_index] = []
            output_line_count += 1

        # write target phrase
        if normalize:
            output_file.write(words[2].capitalize() + ".\n")
        else:
            output_file.write(words[2]+"\n")
        phrase_text.append(words[2])
        score = float(words[-2])
        score_dic[cur_source_index].append((output_line_count, score))

        # update score range
        if (score_range[0] is None) and (score_range[1] is None):
            score_range[0] = score
            score_range[1] = score
        elif score < score_range[0]:
            score_range[0] = score
        elif score > score_range[1]:
            score_range[1] = score

        output_line_count += 1
    print("total #source phrase: {}".format(source_count))
    # construct phrase_pos list
    phrase_pos = [1 for _ in range(output_line_count)]
    return output_sentence_file_name, score_dic, score_range, phrase_pos, phrase_text


def embed_phrase(bird_file_path, source_sents_path, sample_size=-1):
    phrase_pos = []
    out_sents_file = "./bird_sents.txt"
    out_handler = open(out_sents_file, "w")
    source_sents = open(source_sents_path, "r")
    bird_handler = open(bird_file_path, "r")
    source_phrase_num = 0
    cur_source_phrase = None
    cur_source_sent = None
    # index of the phrase in cur_source_sent.
    # need this position to replace source phrase with all other target phrases
    cur_embed_token_index = None

    for line_no, line in enumerate(bird_handler):
        if line_no == 0:
            continue

        words = line.rstrip().split("\t")
        source_phrase, target_phrase = words[1], words[2]
        source_phrase = " " + source_phrase + " " # add space to mark the phrase as words
        target_phrase = " " + target_phrase + " "
        if source_phrase != cur_source_phrase:
            # get a new source phrase
            if source_phrase_num == sample_size:
                break
            source_phrase_num += 1
            cur_source_phrase = source_phrase
            cur_source_sent = source_sents.readline()
            cur_source_sent = cur_source_sent.replace("-", " ")  # preprocess!!
            assert cur_source_phrase in cur_source_sent
            cur_embed_index = cur_source_sent.index(source_phrase)
            pre_phrase_strings = cur_source_sent[:cur_embed_index]
            cur_embed_token_index = len(pre_phrase_strings.split()) + 1  # word count + CLS token
            out_handler.write(cur_source_sent)
            phrase_pos += cur_embed_token_index,

        # write target phrase
        embeded_sent = cur_source_sent.replace(cur_source_phrase, target_phrase)
        out_handler.write(embeded_sent)
        phrase_pos += cur_embed_token_index,

    out_handler.close()
    source_sents.close()
    return out_sents_file, phrase_pos


#v2 of embed_phrase_transformer
def embed_phrase_and_truncate(phrase_dic, phrase_text, sents_corpus_path):
    # delete the phrase if no context sentence can be found
    embedded_sents = []
    truncated_phrase_text = []
    truncated_phrase_dic = {}
    source_phrases =  set()
    for source_phrase_index, target_list in phrase_dic.items():
        source_phrases.add(phrase_text[source_phrase_index])

    embedded_dic = search_phrase_in_text(source_phrases, sents_corpus_path)
    output_line_count = 0
    skip_target_flag = False
    for phrase_index, phrase in enumerate(phrase_text):
        phrase_word = " " + phrase + " "
        if phrase_index in phrase_dic:
            # source phrase
            context_sent = embedded_dic[phrase]
            if context_sent is None:
                skip_target_flag = True
                continue
            else:
                skip_target_flag = False
                cur_context_sent = context_sent.lower()
                org_label_list = phrase_dic[phrase_index]
                assert phrase_word in cur_context_sent  # source phrase must show up in original context sent
                embedded_sents.append(cur_context_sent)
                truncated_phrase_text.append(phrase)
                cur_source_phrase = phrase_word
                truncated_phrase_dic[output_line_count] = []
                cur_label_dic_list = truncated_phrase_dic[output_line_count]
                output_line_count += 1
        elif not skip_target_flag:
            # target phrase
            cur_embeded_sent = cur_context_sent.replace(cur_source_phrase, phrase_word)
            embedded_sents.append(cur_embeded_sent)
            truncated_phrase_text.append(phrase)
            label = [x[1] for x in org_label_list if x[0] == phrase_index][0]
            cur_label_dic_list.append((output_line_count, label))
            output_line_count += 1

    return embedded_sents, truncated_phrase_text, truncated_phrase_dic


# NOTE: is not counterpart of embed_phrase!!!
# rewrite logic and integrate multiple functions
def embed_phrase_transformer(phrase_dic, phrase_text, sents_corpus_path):
    # phrase_dic should be either label_dic or score_dic
    # phrase_dic is required to extract source phrases from phrase_text
    embedded_sents = []
    source_phrases = set()
    for source_phrase_index, target_list in phrase_dic.items():
        source_phrases.add(phrase_text[source_phrase_index])

    # only source phrase should be embedded, target phrases should just use the same context
    embedded_dic = search_phrase_in_text(source_phrases, sents_corpus_path)

    for phrase_index, phrase in enumerate(phrase_text):
        phrase_word = " " + phrase + " "
        if phrase_index in phrase_dic:
            # source phrase
            cur_context_sent = embedded_dic[phrase].lower()
            assert phrase_word in cur_context_sent  # source phrase must show up in original context sent
            embedded_sents.append(cur_context_sent)
            cur_source_phrase = phrase_word
        else:
            # target phrase
            cur_embedded_sent = cur_context_sent.replace(cur_source_phrase, phrase_word)
            embedded_sents.append(cur_embedded_sent)

    return embedded_sents


def split_bird_by_pos(path):
    bird_handler = open(path, "r")
    pos_count = Counter()
    nn_out_name, an_out_name = "./nn_bird.txt", "./an_bird.txt"
    nn_out = open(nn_out_name, "w")
    an_out = open(an_out_name, "w")
    for line_no, line in enumerate(bird_handler):
        if line_no == 0:
            nn_out.write(line)
            an_out.write(line)
            continue
        terms = line.rstrip().split()
        pos = terms[-1]
        if pos == "n-n":
            nn_out.write(line)
        elif pos == "a-n":
            an_out.write(line)
        else:
            print("unknown pos {}".format(pos))
    print(pos_count)
    nn_out.close()
    an_out.close()


def split_bird_by_target_source(path, target_source):
    bird_handler = open(path, "r")
    phrase_source = Counter()
    out_name = "./" + target_source + ".txt"
    out = open(out_name, "w")
    for line_no, line in enumerate(bird_handler):
        if line_no == 0:
            # header
            out.write(line)
            continue
        terms = line.rstrip().split()
        source_relation = terms[-4]
        phrase_source[source_relation] += 1
        if source_relation == target_source:
            out.write(line)
    print(phrase_source)
    out.close()


def embed_bird_in_sent(bird_path, text_path):
    bird_handler = open(bird_path, "r")
    source_list = []
    cur_source_phrase = None

    for line_no, line in enumerate(bird_handler):
        if line_no == 0:
            continue

        words = line.rstrip().split("\t")
        source_phrase, target_phrase = words[1], words[2]
        if source_phrase != cur_source_phrase:
            cur_source_phrase = source_phrase
            if cur_source_phrase not in source_list:
                source_list.append(cur_source_phrase)

    result_dic = search_phrase_in_text(source_list, text_path)
    out = open("./embed_sents_v2.txt", "w")
    for phrase, text in result_dic.items():
        out.write(text)
    out.close()


# helper func for embed bird in sent
def search_phrase_in_text(phrase_list, text_path):
    large_corpus = open(text_path, "r")
    result_dic = OrderedDict()
    for phrase in phrase_list:
        result_dic[phrase] = None

    search_set = set(phrase_list)
    for line in large_corpus:
        if len(search_set) == 0:
            break
        found = set()
        for phrase in search_set:
            if " " + phrase.lower() + " " in line.lower():
                result_dic[phrase] = line
                found.add(phrase)
        search_set -= found
    large_corpus.close()
    if len(search_set) != 0:
        print("WARNING: can't find some of the source phrases in the corpus: {}".format(search_set))
        print("Total v.s. missed: {} v.s. {}".format(len(phrase_list), len(search_set)))

    return result_dic


'''
'''
def generate_sentences(samples_by_tag, sent_group_count=10):
    global objects, vp
    generated_sents = []
    target_range = []
    for _ in range(sent_group_count):
        obj = random.choice(objects)
        verb = random.choice(samples_by_tag["[VP]"])
        s_vp = random.choice(vp)
        left_ind = len(obj)
        right_ind2 = len(verb[0]) + left_ind
        right_ind3 = len(verb[1]) + left_ind
        sent1 = obj + " " + s_vp
        sent2 = obj + " " + " ".join(verb[0]) + " " + s_vp
        sent3 = obj + " " + " ".join(verb[1]) + " " + s_vp
        generated_sents.append((sent1, sent2, sent3))
        target_range.append((None, (left_ind, right_ind2), (left_ind, right_ind3)))

    return generated_sents, target_range


def label_converter(score, score_range=None):
    if score_range is None:
        if score >= 4.0:
            return "high"
        else:
            return "med"
    else:
        split = (score_range[0] + score_range[1]) / 2
        if score >= split:
            return "high"
        else:
            return "low"


def naive_label_converter(score):
    if score == 0:
        return "low"
    else:
        return "high"


def nontrivial_score_to_label(score_dic, score_range=None):
    label_dic = OrderedDict()
    for src_index, score_list in score_dic.items():
        label_list = [(index, label_converter(score, score_range)) for index, score in score_list]
        label_dic[src_index] = label_list
    return label_dic


# use for negative sampling case
def trivial_score_to_label(score_dic):
    label_dic = OrderedDict()
    for src_index, score_list in score_dic.items():
        label_list = [(index, naive_label_converter(score)) for index, score in score_list]
        label_dic[src_index] = label_list
    return label_dic


def extract_embedding(embedding_vector, target, start_loc=None, end=None):
    # explicit index if provided will be the return index
    # if only start_loc is provided, return single vector with that index.
    if target == "cls":
        return embedding_vector[0]
    elif target == "pivot":
        if start_loc is not None:
            return embedding_vector[start_loc]
        else:
            return embedding_vector[2]
    elif target == "sep":
        if start_loc is not None:
            return embedding_vector[start_loc]
        else:
            return embedding_vector[3]
    elif target == "avg_all":
        if end is not None:
            # return torch.mean(embedding_vector[:span], dim=0)
            return torch.mean(embedding_vector[: end], dim=0)
        else:
            return torch.mean(embedding_vector, dim=0)
    elif target == "avg_phrase":
        if start_loc is not None:
            return torch.mean(embedding_vector[start_loc: end], dim=0)
            # return torch.mean(embedding_vector[start_loc :start_loc + span], dim=0)
        else:
            return torch.mean(torch.stack([embedding_vector[1], embedding_vector[2]]), dim=0)
    elif target == "xlnet_special":
        # special adjustment for xlnet
        if (start_loc is not None) and (end is None):
            return embedding_vector[start_loc]
        elif (start_loc is not None) and (end is not None):
            return torch.mean(embedding_vector[start_loc: end], dim=0)
        else:
            print("incorrect use of extract embedding.")
            exit(1)
    else:
        print("unsupport target!!!")
        exit(1)


# helper function for classifier workload generation
# NOTE: tokens argument should be tokenized features from analyzer, not original phrase text
# end pos is exclusive. the phrase range is [start_pos, end_pos)
def depr_extract_related_embedding(embedding_vector, start_pos, end_pos, tokens):
    cls = extract_embedding(embedding_vector, "cls")
    head_token = extract_embedding(embedding_vector, "pivot", start_loc=end_pos - 1)
    sep = extract_embedding(embedding_vector, "sep", start_loc=len(tokens) - 1)
    avg_all = extract_embedding(embedding_vector, "avg_all", end=len(tokens))
    avg_phrase = extract_embedding(embedding_vector, "avg_phrase", start_loc=start_pos, end=end_pos)
    return cls, head_token, sep, avg_all, avg_phrase


# almost the same as the above func, but slightly change the input para
def extract_related_embedding(embedding_vector, start_pos, end_pos, sequence_length, model_name="bert"):
    if model_name in BERT_VARIANTS:
        cls = extract_embedding(embedding_vector, "cls")
        head_token = extract_embedding(embedding_vector, "pivot", start_loc=end_pos - 1)
        sep = extract_embedding(embedding_vector, "sep", start_loc=sequence_length - 1)
        avg_all = extract_embedding(embedding_vector, "avg_all", end=sequence_length)
        avg_phrase = extract_embedding(embedding_vector, "avg_phrase", start_loc=start_pos, end=end_pos)
    elif model_name == "xlnet":
        padded_length = len(embedding_vector)  # size includes the padding tokens
        cls = extract_embedding(embedding_vector, "xlnet_special", start_loc=padded_length - 1)  # last token is cls
        head_token = extract_embedding(embedding_vector, "pivot", start_loc=end_pos - 1)
        sep = extract_embedding(embedding_vector, "sep", start_loc=padded_length - 2)  # second to last is sep
        avg_all = extract_embedding(embedding_vector, "xlnet_special", start_loc=padded_length - sequence_length,
                                    end=padded_length)
        avg_phrase = extract_embedding(embedding_vector, "avg_phrase", start_loc=start_pos, end=end_pos)
    else:
        print("unsupported model: {}".format(model_name))
        exit(1)

    return cls, head_token, sep, avg_all, avg_phrase


# helper function for classifier workload generation
def init_handler_dic(n_layers, out_folder, handler_dic, mode="wb"):
    assert n_layers == len(handler_dic)

    for layer_id in range(n_layers):
        cls_file = os.path.join(out_folder, "cls-emb-" + str(layer_id) + ".pt")
        ht_file = os.path.join(out_folder, "ht-emb-" + str(layer_id) + ".pt")
        sep_file = os.path.join(out_folder, "sep-emb-" + str(layer_id) + ".pt")
        avg_all_file = os.path.join(out_folder, "avg-all-emb-" + str(layer_id) + ".pt")
        avg_phrase_file = os.path.join(out_folder, "avg-phrase-emb-" + str(layer_id) + ".pt")
        handler_dic[layer_id]["cls"] = open(cls_file, mode)
        handler_dic[layer_id]["head_token"] = open(ht_file, mode)
        handler_dic[layer_id]["sep"] = open(sep_file, mode)
        handler_dic[layer_id]["avg_all"] = open(avg_all_file, mode)
        handler_dic[layer_id]["avg_phrase"] = open(avg_phrase_file, mode)


def write_out_embedding(embeddings, layer_id, output_handlers_by_layer):
    cur_handler_dic = output_handlers_by_layer[layer_id]

    cls, head, sep, avg_all, avg_phrase = embeddings

    np.save(cur_handler_dic["cls"], cls.detach().numpy())
    np.save(cur_handler_dic["head_token"], head.detach().numpy())
    np.save(cur_handler_dic["sep"], sep.detach().numpy())
    np.save(cur_handler_dic["avg_all"], avg_all.detach().numpy())
    np.save(cur_handler_dic["avg_phrase"], avg_phrase.detach().numpy())


# use this function when embeddings are not in pairs (single embedding -> label. no src+trg phrases)
def generate_stanford_classifier_workloads(analyzer, config, rand_seed, phrase_text, phrase_labels, phrase_pos,
                                           include_input_emb=False):
    assert len(phrase_pos) == len(phrase_labels)
    assert len(phrase_pos) == len(phrase_text)

    if include_input_emb:
        n_layers = analyzer.n_layers + 1
    else:
        n_layers = analyzer.n_layers
    model_name = analyzer.model_name
    output_handlers_by_layer = [{"cls": None, "head_token": None, "sep": None, "avg_all": None, "avg_phrase": []} for _
                                in range(n_layers)]
    out_folder = config["EMBEDDING_OUT_LOC"]

    # init file handlers
    out_folder = os.path.join(out_folder, str(rand_seed))
    os.mkdir(out_folder)
    init_handler_dic(n_layers, out_folder, output_handlers_by_layer)
    label_raw_handler = open(os.path.join(out_folder, "label.txt"), "w")
    raw_input_handler = open(os.path.join(out_folder, "input.txt"), "w")
    config_handler = open(os.path.join(out_folder, "config.txt"), "w")
    config_handler.write(str(config) + "\n")
    config_handler.close()

    # write out labels
    # since no pair of phrases are involved, can write out label and embedding in serial, instead of tangled together
    for label in phrase_labels:
        label_raw_handler.write(label + "\n")
    label_raw_handler.close()

    for text in phrase_text:
        raw_input_handler.write(text + "\n")
    raw_input_handler.close()

    # write out embeddings
    for phrase_index, phrase_pos in enumerate(phrase_pos):
        # phrase_tokens = analyzer.lookup_sent_text(phrase_index)
        phrase_length = analyzer.lookup_sequence_length(phrase_index)
        start_pos, end_pos = phrase_pos
        for layer_id in range(n_layers):
            raw_embedding = analyzer.lookup_embedding(phrase_index, layer_id)
            # related_embeddings = extract_related_embedding(raw_embedding, start_pos, end_pos, phrase_tokens)
            related_embeddings = extract_related_embedding(raw_embedding, start_pos, end_pos, phrase_length, model_name)
            write_out_embedding(related_embeddings, layer_id, output_handlers_by_layer)

    # closing
    for handler_dic in output_handlers_by_layer:
        for _, handler in handler_dic.items():
            handler.close()

    print("write out {} embeddings & labels".format(len(phrase_labels)))


# use this function to generate embeddings in pair (src+trg for each token)
def generate_classifier_workloads(analyzer, config, rand_seed, phrase_text, label_dic, phrase_pos,
                                  include_input_emb=False):
    # assert len(phrase_pos) == len(analyzer.features)  # should all equal to number of sentences
    assert len(phrase_pos) == len(phrase_text)
    if include_input_emb:
        n_layers = analyzer.n_layers + 1
    else:
        n_layers = analyzer.n_layers
    model_name = analyzer.model_name
    output_handlers_by_layer = [{"cls": None, "head_token": None, "sep": None, "avg_all": None, "avg_phrase": []} for _
                                in range(n_layers)]
    out_folder = config["EMBEDDING_OUT_LOC"]
    pair_count = 0

    # init file handlers
    out_folder = os.path.join(out_folder, str(rand_seed))  # for safety reason. rand_seed should be string already
    os.mkdir(out_folder)

    label_handler = open(os.path.join(out_folder, "label.txt"), "w")
    raw_input_handler = open(os.path.join(out_folder, "raw_input.txt"), "w")  # it's just a copy of bert input file
    label_structure_handler = open(os.path.join(out_folder, "label-structure.pt"), "wb")
    np.save(label_structure_handler, label_dic)
    config_handler = open(os.path.join(out_folder, "config.txt"), "w")
    config_handler.write(str(config) + "\n")
    config_handler.close()

    for text in phrase_text:
        raw_input_handler.write(text + "\n")
    raw_input_handler.close()

    init_handler_dic(n_layers, out_folder, output_handlers_by_layer)


    def write_out_concact_embedding(source_embeddings, target_embeddings, layer_id):
        handler_dic = output_handlers_by_layer[layer_id]

        cls = torch.cat((source_embeddings[0], target_embeddings[0]), 0).detach().numpy()
        head = torch.cat((source_embeddings[1], target_embeddings[1]), 0).detach().numpy()
        sep = torch.cat((source_embeddings[2], target_embeddings[2]), 0).detach().numpy()
        avg_all = torch.cat((source_embeddings[3], target_embeddings[3]), 0).detach().numpy()
        avg_phrase = torch.cat((source_embeddings[4], target_embeddings[4]), 0).detach().numpy()

        np.save(handler_dic["cls"], cls)
        np.save(handler_dic["head_token"], head)
        np.save(handler_dic["sep"], sep)
        np.save(handler_dic["avg_all"], avg_all)
        np.save(handler_dic["avg_phrase"], avg_phrase)

    # write out embeddings
    for source_phrase_index in label_dic:
        second_phrase_list = label_dic[source_phrase_index]
        # source_phrase_tokens = analyzer.lookup_sent_text(source_phrase_index)
        source_phrase_length = analyzer.lookup_sequence_length(source_phrase_index)
        source_start_pos, source_end_pos = phrase_pos[source_phrase_index]

        for second_phrase_index, label in second_phrase_list:
            # target_phrase_tokens = analyzer.lookup_sent_text(second_phrase_index)
            target_phrase_length =  analyzer.lookup_sequence_length(second_phrase_index)
            target_start_pos, target_end_pos = phrase_pos[second_phrase_index]

            # write out current label
            label_handler.write(label + "\n")
            pair_count += 1

            for layer_id in range(n_layers):
                # extract embeddings of current layer
                source_embedding = analyzer.lookup_embedding(source_phrase_index, layer_id)
                target_embedding = analyzer.lookup_embedding(second_phrase_index, layer_id)
                src_embeddings = extract_related_embedding(source_embedding, source_start_pos, source_end_pos,
                                                           source_phrase_length, model_name)
                trg_embeddings = extract_related_embedding(target_embedding, target_start_pos, target_end_pos,
                                                           target_phrase_length, model_name)
                write_out_concact_embedding(src_embeddings, trg_embeddings, layer_id)

    # closing
    label_handler.close()
    label_structure_handler.close()
    for handler_dic in output_handlers_by_layer:
        for key, handler in handler_dic.items():
            handler.close()

    print("write out {} embeddings & labels".format(pair_count))


'''
input: 
    - generated_sents: list of strings. each string corresponds to a sentence
    - out: output path
'''
def write_sentences(generated_sents, out="./sents.txt"):
    out_f = open(out, "w")
    for sent_group in generated_sents:
        for sent in sent_group:
            out_f.write(sent+"\n")
    out_f.close()


def write_pairs(pairs, out="./pairs.txt"):
    out_f = open(out, "w")
    for phrase1, phrase2 in pairs:
        phrase1 = " ".join(phrase1)
        phrase2 = " ".join(phrase2)
        out_f.write(phrase1 + "\t" + phrase2 + "\n")
    out_f.close()
