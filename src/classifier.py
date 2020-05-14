from numpy import load
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import multiprocessing as mp
import os
import argparse
import logging
from configuration import config

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# NOTE: tokens should always be ordered as: cls, ht, sep, avg_all, avg_phrase
MAX_ITER = 250


def verify_embeddings(n_layers, working_dir):
    logger.info("validating embeddings...")
    count = None
    for layer_id in range(n_layers):
        token2file = generate_token2file(layer_id, working_dir)
        for file in token2file.values():
            handler = open(file, "rb")
            embeddings = read_embeddings(handler)
            if count is None:
                count = len(embeddings)
            else:
                assert count == len(embeddings)
            handler.close()
    logger.info("pass validation.")
    logger.info("#samples: {}".format(count))


def dump(clf_models, working_dir):
    # expect a list of (layer_id, model) tuple
    clf_filename = "trained-classifier-layer-{}.pt"
    for layer_id, model in enumerate(clf_models):
        cur_clf_filename = clf_filename.format(layer_id)
        handler = open(os.path.join(working_dir, cur_clf_filename), "wb")
        np.save(handler, model)
        handler.close()


# must be used in pair with dump function
def load_trained_classifier(n_layers, working_dir):
    filename = "trained-classifier-layer-{}.pt"
    trained_clf = []

    for layer_id in range(n_layers):
        cur_filename = filename.format(layer_id)
        handler = open(os.path.join(working_dir, cur_filename), "rb")
        model = np.load(handler, allow_pickle=True)
        trained_clf.append(model.item())
        handler.close()

    logger.info("loaded {} models".format(len(trained_clf)))
    return trained_clf


def load_labels(working_dic):
    label_handler = open(os.path.join(working_dic, "label.txt"), "r")
    logger.info("loading labels from {}".format(working_dic))
    labels = []
    for line in label_handler:
        line = line.strip()
        labels.append(line)
    label_handler.close()
    return labels


def classify_task(embedding_path, labels, test_size=0.25):
    # loading embeddings from file
    embedding_handler = open(embedding_path, "rb")
    embeddings = []
    try:
        while True:
            rst = load(embedding_handler, allow_pickle=True)
            embeddings.append(rst)
    except IOError:
        logger.info("Read {} embeddings from {}".format(len(embeddings), embedding_path))
        embedding_handler.close()

    # training classifier
    logger.info("Training classifier with {}".format(embedding_path))
    x = np.asarray(embeddings)
    y = np.asarray(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)  # TODO: need to fix the split
    clf = MLPClassifier(hidden_layer_sizes=256, activation='relu', max_iter=MAX_ITER)
    clf.fit(x_train, y_train)

    # testing
    logger.info("Testing classifier with {}".format(embedding_path))
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, acc


def generate_token2file(layer_id, working_dir):
    token2file = {"cls": os.path.join(working_dir, "cls-emb-" + str(layer_id) + ".pt"),
                  "ht": os.path.join(working_dir, "ht-emb-" + str(layer_id) + ".pt"),
                  "sep": os.path.join(working_dir, "sep-emb-" + str(layer_id) + ".pt"),
                  "avg_all": os.path.join(working_dir, "avg-all-emb-" + str(layer_id) + ".pt"),
                  "avg_phrase": os.path.join(working_dir, "avg-phrase-emb-" + str(layer_id) + ".pt")}
    return token2file


def read_embeddings(embedding_handler):
    embeddings = []
    try:
        while True:
            rst = load(embedding_handler, allow_pickle=True)
            embeddings.append(rst)
    except IOError:
        return embeddings


# should always use dispatch and collect function in pair. it's relying on the order of files
def dispatch(working_dir, layer_id, labels):
    file_list = []

    token2file = generate_token2file(layer_id, working_dir)

    file_list.append(token2file["cls"])
    file_list.append(token2file["ht"])
    file_list.append(token2file["sep"])
    file_list.append(token2file["avg_all"])
    file_list.append(token2file["avg_phrase"])

    para_list = [(file_name, labels) for file_name in file_list]
    return para_list


def collect(multi_process_results, layer_id, result_dic, model_dic):
    # throw away models for now. just record the accuracy number.
    # update: now return all the trained models as well, in order to pass to lime explanation
    cur_rst_dic = result_dic[layer_id]
    cur_model_dic = model_dic[layer_id]

    # update result dic
    cur_rst_dic["cls"] = multi_process_results[0][1]
    cur_rst_dic["head_token"] = multi_process_results[1][1]
    cur_rst_dic["sep"] = multi_process_results[2][1]
    cur_rst_dic["avg_all"] = multi_process_results[3][1]
    cur_rst_dic["avg_phrase"] = multi_process_results[4][1]

    # update model dic
    cur_model_dic["cls"] = multi_process_results[0][0]
    cur_model_dic["head_token"] = multi_process_results[1][0]
    cur_model_dic["sep"] = multi_process_results[2][0]
    cur_model_dic["avg_all"] = multi_process_results[3][0]
    cur_model_dic["avg_phrase"] = multi_process_results[4][0]


def classify_task_by_layer(layer_id, file_list, labels):
    # file_list is organized as cls, ht, sep, avg_all, avg_phrase
    logger.info("Classification task for layer {}".format(layer_id))

    # loading embeddings and init file handlers
    embeddings, handler_list = [], []
    for file in file_list:
        handler_list.append(open(file, "rb"))
    try:
        while True:
            cur_emb = []
            for handler in handler_list:
                emb = load(handler, allow_pickle=True)
                cur_emb = np.concatenate((cur_emb, emb))
            embeddings.append(cur_emb)
    except IOError:
        logger.info("Read {} embeddings for layer {}".format(len(embeddings), layer_id))
        for handler in handler_list:
            handler.close()

    # train classifier
    clf = MLPClassifier(hidden_layer_sizes=256, activation='relu', max_iter=MAX_ITER)
    x = np.asarray(embeddings)
    y = np.asarray(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    clf.fit(x_train, y_train)

    # testing
    logger.info("Testing classifier for layer {}".format(layer_id))
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return layer_id, clf, acc


# should always use dispatch and collect function in pair
def dispatch_by_layer(working_dir, n_layers, labels):
    para_list = []
    for layer_id in range(n_layers):
        token2file = generate_token2file(layer_id, working_dir)
        cur_file_list = [token2file["cls"], token2file["ht"], token2file["sep"], token2file["avg_all"], token2file["avg_phrase"]]
        para_list.append((layer_id, cur_file_list, labels))
    return para_list


def collect_by_layer(multi_process_results, result_list, model_list):
    for layer_id, model, acc in multi_process_results:
        logger.info("Processing results for layer {}".format(layer_id))
        result_list[layer_id] = acc
        model_list[layer_id] = model


def print_clf_rst(result_dic):
    output_format = "{}\t{}"
    title = "Layer\tAccuracy"
    logger.info("Classification Results")

    def print_token(token):
        logger.info("Token {}".format(token))
        logger.info(title)
        for layer_id, acc_dic in enumerate(result_dic):
            acc = acc_dic[token]
            logger.info(output_format.format(layer_id, acc))

    print_token("cls")
    print_token("head_token")
    print_token("avg_all")
    print_token("avg_phrase")
    print_token("sep")


def classify_by_token(n_layers, labels, pool, working_dir):
    clf_rst = [{"cls": None, "head_token": None, "sep": None, "avg_all": None, "avg_phrase": None} for _ in
               range(n_layers)]
    clf_models = [{"cls": None, "head_token": None, "sep": None, "avg_all": None, "avg_phrase": None} for _ in
               range(n_layers)]
    for layer_id in range(n_layers):
        para_list = dispatch(working_dir, layer_id, labels)
        results = pool.starmap(classify_task, para_list)
        collect(results, layer_id, clf_rst, clf_models)

    print_clf_rst(clf_rst)


def classify_by_layer(n_layers, labels, pool, working_dir):
    clf_models = [None for _ in range(n_layers)]
    clf_results = [None for _ in range(n_layers)]
    para_list = dispatch_by_layer(working_dir, n_layers, labels)
    results = pool.starmap(classify_task_by_layer, para_list)
    collect_by_layer(results, clf_results, clf_models)

    logger.info(clf_results)
    logger.info("dumping trained models...")
    dump(clf_models, working_dir)
    return clf_results, clf_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--by_layer", action='store_true')
    parser.add_argument("--n_layers", default=12, type=int, required=False)
    parser.add_argument("--rand_seed", type=int, required=True)
    args = parser.parse_args()

    # if false, run classification task by token instead
    # set true if downstream task is lime
    # false for per-token performance evaluation
    n_layers = args.n_layers
    by_layer = args.by_layer

    working_dir = os.path.join(config["EMBEDDING_OUT_LOC"], str(args.rand_seed))

    verify_embeddings(n_layers, working_dir)
    label_handler = open(os.path.join(working_dir, "label.txt"), "r")
    configure_handler = open(os.path.join(working_dir, "config.txt"), "r")
    labels = []
    core_count = mp.cpu_count()
    pool = mp.Pool(core_count)
    logger.info("Using {} cores".format(core_count))
    logger.info("Current configurations:")
    text = configure_handler.readlines()
    logger.info(text)

    for line in label_handler:
        line = line.strip()
        labels.append(line)

    if by_layer:
        classify_by_layer(n_layers, labels, pool, working_dir)
    else:
        classify_by_token(n_layers, labels, pool, working_dir)

    label_handler.close()
    configure_handler.close()


if __name__ == "__main__":
    main()