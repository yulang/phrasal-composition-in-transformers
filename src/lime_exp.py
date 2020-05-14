from lime.lime_text import LimeTextExplainer
import numpy as np
from classifier import load_trained_classifier, generate_token2file, read_embeddings, load_labels
import multiprocessing as mp
from sklearn import preprocessing
import random
import argparse
import os
from configuration import config


all_tokens = ["cls", "ht", "sep", "avg_all", "avg_phrase"]
wrapped_text = " ".join(all_tokens)


def gather_embeddings(token2embedding, sample_index):
    # extract embeddings correspond to
    embedding_dic = {}
    for token, embedding_list in token2embedding.items():
        embedding_dic[token] = embedding_list[sample_index]
    return embedding_dic


def clf_wrapper(clf, token2embedding, sample_index):
    embedding_dic = gather_embeddings(token2embedding, sample_index)
    token_emb_dimension = len(embedding_dic["cls"])
    def wrapped_clf(texts):
        # text should be in form: "cls ht sep avg_all avg_phrase"
        rst = None
        for text in texts:
            words = text.split()
            embedding = []
            for token in all_tokens:
                if token in words:
                    cur_embedding = embedding_dic[token]
                else:
                    cur_embedding = np.zeros(token_emb_dimension)
                embedding = np.concatenate((embedding, cur_embedding))

            if rst is None:
                rst = clf.predict_proba([embedding])
            else:
                rst = np.vstack([rst, clf.predict_proba([embedding])[0]])
        return rst

    return wrapped_clf


def explain_classifier(clf, layer_id, labels, working_dir):
    # load embeddings of current layer
    embedding_files = generate_token2file(layer_id, working_dir)
    token2embedding = {}
    sample_size = len(labels)
    # print("total sample size {}".format(sample_size))

    for key, val in embedding_files.items():
        # key will be cls, ht, ..., etc
        handler = open(val, "rb")
        cur_token_embedding = read_embeddings(handler)  # cur_token_embedding is a list of embeddings
        handler.close()
        token2embedding[key] = cur_token_embedding
        assert len(cur_token_embedding) == sample_size

    sample_index = random.choice(range(sample_size))  # affects which embedding the wrapped_clf will load
    class_names = list(clf.classes_)

    explainer = LimeTextExplainer(class_names=class_names)
    wrapped_clf = clf_wrapper(clf, token2embedding, sample_index)
    # print("correct label {}".format(labels[sample_index]))
    prediction = wrapped_clf([wrapped_text])
    # print("model prediction {}".format(prediction))
    exp = explainer.explain_instance(wrapped_text, wrapped_clf, num_features=6)

    # generate class index for weight parsing
    if labels[sample_index] == class_names[0]:
        class_index = 0
    else:
        class_index = 1

    polarized_exp_list = weight_polarizer(exp, class_index)
    normalized_exp_list = weight_normalizer(exp, class_index)

    return polarized_exp_list, normalized_exp_list, class_index, prediction


def explain_by_layer(n_layers, labels, pool, num_samples, working_dir):
    trained_clfs = load_trained_classifier(n_layers, working_dir)
    for layer_id, trained_clf in enumerate(trained_clfs):
        print("explaining layer {}".format(layer_id))
        print("label order: {}".format(trained_clf.classes_))
        # repeat the approximation _num_samples_ times
        para_list = [(trained_clf, layer_id, labels, working_dir) for _ in range(num_samples)]
        # explain_classifier(trained_clf, layer_id, labels)
        results = pool.starmap(explain_classifier, para_list)
        acc_org, acc_norm = parse_multiprocess_results(results, num_samples)
        print("original weights:")
        print(acc_org)
        print("normalized weights")
        print(acc_norm)


def parse_multiprocess_results(results, num_samples):
    print("gathering data")
    # init accumulated weights
    acc_normalized_weights, acc_org_weights = {}, {}
    for token in all_tokens:
        acc_normalized_weights[token] = 0.0
        acc_org_weights[token] = 0.0

    for org_exp_list, normalized_exp_list, cur_label, prediction in results:
        for token, weight in org_exp_list:
            acc_org_weights[token] += weight
        for token, normalized_weight in normalized_exp_list:
            acc_normalized_weights[token] += normalized_weight

    # normalize weights by the number of random samples drawn
    for key in acc_org_weights:
        old_val = acc_org_weights[key]
        acc_org_weights[key] = old_val / num_samples
        old_val = acc_normalized_weights[key]
        acc_normalized_weights[key] = old_val / num_samples

    return acc_org_weights, acc_normalized_weights


def weight_normalizer(exp, label):
    # should call weight_polarizer first to make weights with correct signs
    # (positive pushing to right decision, negative pushing to wrong decision)
    # this funciton scales weights to be in range [0, 1]
    polarized_weights = weight_polarizer(exp, label)
    features = [x[0] for x in polarized_weights]
    weights = [x[1] for x in polarized_weights]
    normalized_weights = preprocessing.minmax_scale(weights)
    return list(zip(features, normalized_weights))


def weight_polarizer(exp, label):
    # if label = 0, negative weights push to right decision
    # label = 1, positive weights push to right decision
    exp_list = exp.as_list()
    features = [x[0] for x in exp_list]
    if label == 1:
        polarized_weights = [x[1] for x in exp_list]
    else:
        polarized_weights = [-x[1] for x in exp_list]
    return list(zip(features, polarized_weights))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--rand_seed", type=int, required=True)
    parser.add_argument("--n_layers", type=int, default=12, required=False)

    args = parser.parse_args()
    num_samples_per_layer = args.num_samples
    cur_rand_seed = args.rand_seed
    cur_working_dir = os.path.join(config["EMBEDDING_OUT_LOC"], str(cur_rand_seed))

    # load labels
    labels = load_labels(cur_working_dir)

    # create multiprocessing pool
    core_count = mp.cpu_count()
    pool = mp.Pool(core_count)
    print("Using {} cores".format(core_count))

    # start explaining
    print("Explaining...")
    explain_by_layer(args.n_layers, labels, pool, num_samples_per_layer, cur_working_dir)


if __name__ == "__main__":
    main()