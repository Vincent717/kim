# -*- coding: UTF-8 -*-


# date  : 2017.12.23

"""
data processing
"""

from tqdm import tqdm
from conf import universal_config
import json
import random
import collections
import re
import numpy as np
from mxnet import nd

import utils



PADDING = "<PAD>"
LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": -1
}

_ctx = utils.try_gpu()

def load_data(data_path, ctx=_ctx, snli=True, is_train=True, need_char=False, need_syntacic=False):
    dataset = load_nli_data(data_path, snli=snli)
    if need_char: # need character level data
    if is_train:
        i2w, w2i = sentences_to_padded_index_sequences([dataset], need_char, need_syntacic, ctx)
        dataset = get_pure_data(dataset, ctx)
        return dataset, i2w, w2i
    else:
        sentences_to_padded_sequences(dataset, need_char, need_syntacic, ctx)
        return dataset

def load_nli_data(path, snli=False, shuffle = True):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path) as f:
        for line in tqdm(f):
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    return data

def load_nli_data_genre(path, genre, snli=True, shuffle = True):
    """
    Load a specific genre's examples from MultiNLI, or load SNLI data and assign a "snli" genre to the examples.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. If set to true, it will overwrite the genre label for MultiNLI data.
    """
    data = []
    j = 0
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            if loaded_example["genre"] == genre:
                data.append(loaded_example)
        if shuffle:
            random.seed(1)
            random.shuffle(data)
    return data


# process data

def sentences_to_padded_index_sequences(datasets, need_char=False, need_syntacic=False, ctx=_ctx):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    # Extract vocabulary
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    word_counter = collections.Counter()
    char_counter = collections.Counter()
    # mgr = multiprocessing.Manager()
    # shared_content = mgr.dict()
    # process_num = config.num_process_prepro
    # process_num = 1
    maxlen = 0
    for i, dataset in enumerate(datasets):
        # if not shared_file_exist:
        #     num_per_share = len(dataset) / process_num + 1
        #     jobs = [ multiprocessing.Process(target=worker, args=(shared_content, dataset[i * num_per_share : (i + 1) * num_per_share] )) for i in range(process_num)]
        #     for j in jobs:
        #         j.start()
        #     for j in jobs:
        #         j.join()

        for example in tqdm(dataset):
            s1_tokenize = tokenize(example['sentence1_binary_parse'])
            s2_tokenize = tokenize(example['sentence2_binary_parse'])
            maxlen = max(len(s1_tokenize), len(s2_tokenize), maxlen)
            word_counter.update(s1_tokenize)
            word_counter.update(s2_tokenize)

            for i, word in enumerate(s1_tokenize):
                char_counter.update([c for c in word])
            for word in s2_tokenize:
                char_counter.update([c for c in word])

        # shared_content = {k:v for k, v in shared_content.items()}

    vocabulary = set([word for word in word_counter])
    vocabulary = list(vocabulary)
    if universal_config['embedding_replacing_rare_word_with_UNK']: 
        vocabulary = [PADDING, "<UNK>"] + vocabulary
    else:
        vocabulary = [PADDING] + vocabulary
    # print(char_counter)
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    indices_to_words = {v: k for k, v in word_indices.items()}

    # char level
    if need_char:
        char_vocab = set([char for char in char_counter])
        char_vocab = list(char_vocab)
        char_vocab = [PADDING] + char_vocab
        char_indices = dict(zip(char_vocab, range(len(char_vocab))))
        indices_to_char = {v: k for k, v in char_indices.items()}
        

    for i, dataset in enumerate(datasets):
        for example in tqdm(dataset):
            for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                example[sentence + '_index_sequence'] = np.zeros((universal_config["seq_length"]), dtype=np.int32)
                example[sentence + '_inverse_term_frequency'] = np.zeros((universal_config["seq_length"]), dtype=np.float32)

                token_sequence = tokenize(example[sentence])
                padding = universal_config["seq_length"] - len(token_sequence)
                      
                for i in range(universal_config["seq_length"]):
                    if i >= len(token_sequence):
                        index = word_indices[PADDING]
                        itf = 0
                    else:
                        if universal_config['embedding_replacing_rare_word_with_UNK']:
                            index = word_indices[token_sequence[i]] if word_counter[token_sequence[i]] >= universal_config['UNK_threshold'] else word_indices["<UNK>"]
                        else:
                            index = word_indices[token_sequence[i]]
                        itf = 1 / (word_counter[token_sequence[i]] + 1)
                    example[sentence + '_index_sequence'][i] = index
                    
                    example[sentence + '_inverse_term_frequency'][i] = itf
                
                # char level : seq_len * word_len
                if need_char:  
                    example[sentence + '_char_index'] = np.zeros((universal_config["seq_length"], universal_config["char_in_word_size"]), dtype=np.int32)
                    for i in range(universal_config["seq_length"]):
                        if i >= len(token_sequence):
                            continue
                        else:
                            chars = [c for c in token_sequence[i]]
                            for j in range(universal_config["char_in_word_size"]):
                                if j >= (len(chars)):
                                    break
                                else:
                                    index = char_indices[chars[j]]
                                example[sentence + '_char_index'][i,j] = index 
                if need_syntacic:
                    premise_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence1_parse'][:] for i in indices], premise_pad_crop_pair)
                    hypothesis_pos_vectors = generate_pos_feature_tensor([dataset[i]['sentence2_parse'][:] for i in indices], hypothesis_pad_crop_pair)

                    premise_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence1_token_exact_match_with_s2"][:] for i in range(len(indices))], premise_pad_crop_pair, 1)
                    hypothesis_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence2_token_exact_match_with_s1"][:] for i in range(len(indices))], hypothesis_pad_crop_pair, 1)
              
                    premise_exact_match = np.expand_dims(premise_exact_match, 2)
                    hypothesis_exact_match = np.expand_dims(hypothesis_exact_match, 2)


    print(maxlen, 'max len')
    if need_char:
        return indices_to_words, word_indices, char_indices, indices_to_char
    else:
        return indices_to_words, word_indices, None, None  


def sentences_to_padded_sequences(dataset, need_char=False, need_syntacic=False):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    # Extract vocabulary
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    def pad_list(s, fix_len):
        res = list(s)
        if len(res) > fix_len:
            return res[:fix_len]
        else:
            res += [''] * (fix_len - len(res))
            return res

    for example in tqdm(dataset):
        for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
            token_sequence = tokenize(example[sentence])[:universal_config["seq_length"]]
            example[sentence + '_sequence'] = token_sequence[:]
            padding = universal_config["seq_length"] - len(token_sequence)
            example[sentence + '_sequence'] += [PADDING] * padding
            if need_char:
                example[sentence + '_char'] = [pad_list(word, universal_config['char_in_word_size']) for word in token_sequence]
                example[sentence + '_char'] += [[''] * universal_config['char_in_word_size']] * padding

def get_pure_data(dataset, ctx):
    X = []
    y = []
    for doc in tqdm(dataset):
        ab = [doc.get('sentence1_binary_parse_index_sequence', [0] * universal_config['seq_length'])
             ,doc.get('sentence2_binary_parse_index_sequence', [0] * universal_config['seq_length'])
            ]
        X.append(ab)
        y.append(doc.get('label', -1))
    return nd.array(X, ctx=ctx), nd.array(y, ctx=ctx)

def map_to_index(dataset, w2i, ctx=_ctx):
    s2is = lambda s: [w2i.get(i,w2i.get(PADDING)) for i in s]

    X = []
    y = []
    for doc in tqdm(dataset):
        ab = [s2is(doc.get('sentence1_binary_parse_sequence', [])),
              s2is(doc.get('sentence2_binary_parse_sequence', [])),
            ]
        X.append(ab)
        y.append(doc.get('label', -1))
    return nd.array(X, ctx=ctx), nd.array(y, ctx=ctx)


if __name__ == '__main__':
    dpath = '../data/snli_1.0/snli_1.0_fake_train.jsonl'
    x = load_data(dpath)
    x1 = sentences_to_padded_index_sequences([x])