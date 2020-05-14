import torch
from numpy import load, save
import time


class TransformerAnalyzer:
    # wrapper for transformer dump
    def __init__(self, embedding_dump_loc, n_layers, phrase_list, sentence_list, sequence_length_list, model_name,
                 include_input_emb=False):
        assert len(phrase_list) == len(sentence_list)
        assert len(sentence_list) == len(sequence_length_list)
        timestamp = str(time.time())[-6:] + "-"
        self.rand_seed = timestamp
        self.dump_loc = embedding_dump_loc
        self.length_list = sequence_length_list
        self.embedding_handler = open(embedding_dump_loc, "rb")
        self.phrase_list = phrase_list
        self.sentence_list = sentence_list
        self.n_layers = n_layers
        self.include_input_emb = include_input_emb
        self.model_name = model_name

        # [fallback_lower_sentidx, cur_lower_sentidx) -> fallback segment
        # [cur_lower_sentidx, cur_upper_sentidx) -> cur_segment
        self.fallback_segment = None
        self.cur_segment = None
        self.cur_upper_sentidx = None
        self.cur_lower_sentidx = None
        self.fallback_lower_sentidx = None

    def lookup_sequence_length(self, sent_idx):
        return self.length_list[sent_idx]

    # if include_input_emb = true, layer_id can range from [0, 12], where layer 0 is the input embedding
    # include_input_emb = false, layer_id range from [0, 11], where layer 0 is the output from first layer
    def lookup_embedding(self, sent_idx, layer_id):
        if (self.cur_upper_sentidx is None) or (sent_idx >= self.cur_upper_sentidx):
            # load a new segment from disc
            self.fallback_segment = self.cur_segment
            self.cur_segment = self.restore_one_segment()
            segment_size = TransformerAnalyzer.get_segment_size(self.cur_segment)
            if self.cur_lower_sentidx is None:
                self.cur_lower_sentidx = 0
                self.cur_upper_sentidx = segment_size
            else:
                # update both cur boundary and fallback boundary
                self.fallback_lower_sentidx = self.cur_lower_sentidx
                self.cur_lower_sentidx = self.cur_upper_sentidx
                self.cur_upper_sentidx = self.cur_lower_sentidx + segment_size

        # finish loading. now lookup embedding
        if self.include_input_emb:
            layer_offset = layer_id
        else:
            layer_offset = layer_id + 1

        if sent_idx < self.cur_lower_sentidx:
            # lookup on fallback_segment
            if sent_idx < self.fallback_lower_sentidx:
                print("sent id {}".format(sent_idx))
                print(self.fallback_lower_sentidx, self.cur_lower_sentidx, self.cur_upper_sentidx)
                assert False
            offset_in_segment = sent_idx - self.fallback_lower_sentidx
            embedding_vec = self.fallback_segment[layer_offset][offset_in_segment]
        else:
            # normal lookup on cur_segment
            offset_in_segment = sent_idx - self.cur_lower_sentidx
            embedding_vec = self.cur_segment[layer_offset][offset_in_segment]

        # embedding_vec shape: (sequence_length, hidden_size)
        return embedding_vec

    def restore_one_segment(self):
        assert self.embedding_handler is not None
        segment = load(self.embedding_handler, allow_pickle=True)
        return segment

    def close_handler(self):
        self.embedding_handler.close()

    def reset_handler(self):
        self.embedding_handler.close()
        self.embedding_handler = open(self.dump_loc, "rb")
        self.fallback_segment = None
        self.cur_segment = None
        self.cur_upper_sentidx = None
        self.cur_lower_sentidx = None
        self.fallback_lower_sentidx = None

    @staticmethod
    def get_segment_size(segment):
        return segment[1].shape[0] # segment is of shape (#layers + 1, batch_size, sequence_length, hidden_size)
