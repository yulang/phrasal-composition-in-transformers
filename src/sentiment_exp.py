import os
from configuration import config



# possible labels: very neg, neg, neutral, pos, very pos
def score_to_label(score):
    if 0 <= score <= 0.2:
        return "very neg"
    elif 0.2 < score <= 0.4:
        return "neg"
    elif 0.4 < score <= 0.6:
        return "neutral"
    elif 0.6 < score <= 0.8:
        return "pos"
    elif 0.8 < score <= 1:
        return "very pos"
    else:
        print("illegal sentiment score")


def generate_workload(phrase_dic_path, sentiment_label_path):
    phrase_dic_handler = open(phrase_dic_path, "r")
    sentiment_label_handler = open(sentiment_label_path, "r")
    id_2phrase, id_2sentiment_score, id_2sentiment_label = {}, {}, {}
    phrase2id = {}  # for debug purpose

    # parse dictionary
    for line in phrase_dic_handler:
        line = line.strip()
        entries = line.split("|")
        phrase, phrase_id = entries[0], int(entries[1])
        id_2phrase[phrase_id] = phrase
        phrase2id[phrase] = phrase_id

    # parse sentiment
    for line_no, line in enumerate(sentiment_label_handler):
        if line_no == 0:
            # skip header
            continue
        line = line.strip()
        entries = line.split("|")
        phrase_id, sentiment_score = int(entries[0]), float(entries[1])
        id_2sentiment_score[phrase_id] = sentiment_score
        id_2sentiment_label[phrase_id] = score_to_label(sentiment_score)

    phrase_dic_handler.close()
    sentiment_label_handler.close()
    return id_2phrase, id_2sentiment_score, id_2sentiment_label, phrase2id


def construct_sentiment_dic():
    phrase_dic_path = os.path.join(config["STANFORD_LOC"], "dictionary.txt")
    sentiment_label_path = os.path.join(config["STANFORD_LOC"], "sentiment_labels.txt")
    id2phrase, id2sent_score, id2sent_label, phrase2id = generate_workload(phrase_dic_path, sentiment_label_path)
    return id2phrase, id2sent_score, id2sent_label, phrase2id
