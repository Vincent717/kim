# -*- coding: UTF-8 -*-


# date  : 2017.12.23

"""
data processing
"""

import os
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
POS_Tagging = [PADDING, 'WP$', 'RBS', 'SYM', 'WRB', 'IN', 'VB', 'POS', 'TO', ':', '-RRB-', '$', 'MD', 'JJ', '#', 'CD', '``', 'JJR', 'NNP', "''", 'LS', 'VBP', 'VBD', 'FW', 'RBR', 'JJS', 'DT', 'VBG', 'RP', 'NNS', 'RB', 'PDT', 'PRP$', '.', 'XX', 'NNPS', 'UH', 'EX', 'NN', 'WDT', 'VBN', 'VBZ', 'CC', ',', '-LRB-', 'PRP', 'WP']
POS_dict = {pos:i for i, pos in enumerate(POS_Tagging)}


LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": -1
}

_ctx = utils.try_gpu()

def load_data(data_path, ctx=_ctx, snli=True, is_train=True, need_char=False, need_syntatic=False):
    dataset = load_nli_data(data_path, snli=snli)
    if is_train:
        i2w, w2i, i2c, c2i = sentences_to_padded_index_sequences([dataset], need_char, need_syntatic, ctx)
        dataset = get_pure_data(dataset, ctx)
        return dataset, i2w, w2i, i2c, c2i
    else:
        sentences_to_padded_sequences(dataset, need_char, need_syntatic, ctx)
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

def sentences_to_padded_index_sequences(datasets, need_char=False, need_syntatic=False, ctx=_ctx):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    # Extract vocabulary
    def tokenize(string):
        string = re.sub(r'\(|\)', '', string)
        return string.split()

    revs = lambda x: str(3-int(x))

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

    # if syntax information is required, should load shared json first
    if need_syntatic:
        shared_content = load_mnli_shared_content()

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
                    sentence_ = sentence[:9]  
                    example[sentence_ + '_char_index'] = np.zeros((universal_config["seq_length"], universal_config["char_in_word_size"]), dtype=np.int32)
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
                                example[sentence_ + '_char_index'][i,j] = index 
                if need_syntatic:
                    sentence_ = sentence[:9]
                    pad_crop_pair = (0,0)
                    example[sentence_ + '_pos_vector'] = generate_pos_feature_tensor(example[sentence_ +'_parse'][:], pad_crop_pair)

                    premise_exact_match = construct_one_hot_feature_tensor(shared_content[example['pairID']][sentence_ + "_token_exact_match_with_s" + revs(sentence_[-1])][:], pad_crop_pair, 1)
                    example[sentence_ + '_exact_match'] = np.expand_dims(premise_exact_match, 2)

    print(maxlen, 'max len')
    if need_char:
        return indices_to_words, word_indices, indices_to_char, char_indices 
    else:
        return indices_to_words, word_indices, None, None  


def sentences_to_padded_sequences(dataset, need_char=False, need_syntatic=False, ctx=_ctx):
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
                sentence_ = sentence[:9]
                example[sentence + '_char'] = [pad_list(word, universal_config['char_in_word_size']) for word in token_sequence]
                example[sentence + '_char'] += [[''] * universal_config['char_in_word_size']] * padding
            if need_syntatic:
                sentence_ = sentence[:9]
                pad_crop_pair = (0,0)
                example[sentence_ + '_pos_vector'] = generate_pos_feature_tensor(example[sentence_ +'_parse'][:], pad_crop_pair)
                premise_exact_match = construct_one_hot_feature_tensor(shared_content[example['pairIDs']][sentence_ + "token_exact_match_with_s" + revs(sentence_[-1])][:], pad_crop_pair, 1)
                example[sentence_ + '_exact_match'] = np.expand_dims(premise_exact_match, 2)


def get_pure_data(dataset, ctx):
    X = []
    X_char = []
    X_pos = []
    X_exact = []
    y = []
    for doc in tqdm(dataset):
        ab = [doc.get('sentence1_binary_parse_index_sequence', [0] * universal_config['seq_length'])
             ,doc.get('sentence2_binary_parse_index_sequence', [0] * universal_config['seq_length'])
            ]
        X.append(ab)
        if 'sentence1_char_index' in doc:   # need char level
            ab_char = [doc.get('sentence1_char_index')
                      ,doc.get('sentence2_char_index')]    # 2*30*8
            X_char.append(ab_char)
        if 'sentence1_pos_vector' in doc and 'sentence1_exact_match' in doc:    # need syntax info
            ab_pos = [doc.get('sentence1_pos_vector')       
                     ,doc.get('sentence2_pos_vector')]    # 2*30*47
            ab_exact = [doc.get('sentence1_exact_match')
                       ,doc.get('sentence2_exact_match')]   # 2*30*1
            X_pos.append(ab_pos)
            X_exact.append(ab_exact)
        y.append(doc.get('label', -1))
    X_output = [nd.array(X, ctx=ctx)]
    if X_char:
        X_output.append(nd.array(X_char, ctx=ctx))
    if X_pos:
        X_output.append(nd.array(X_pos, ctx=ctx))
    if X_exact:
        X_output.append(nd.array(X_exact, ctx=ctx))
    return tuple(X_output), nd.array(y, ctx=ctx)

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


def load_shared_content(fh, shared_content):
    for line in fh:
        row = line.rstrip().split("\t")
        key = row[0]
        value = json.loads(row[1])
        shared_content[key] = value

def load_mnli_shared_content():
    shared_file_exist = False
    # shared_path = config.datapath + "/shared_2D_EM.json"
    # shared_path = config.datapath + "/shared_anto.json"
    # shared_path = config.datapath + "/shared_NER.json"
    shared_path = universal_config['shared_json_path']
    # shared_path = "../shared.json"
    print(shared_path)
    if os.path.isfile(shared_path):
        shared_file_exist = True
    # shared_content = {}
    #assert shared_file_exist
    # if not shared_file_exist and config.use_exact_match_feature:
    #     with open(shared_path, 'w') as f:
    #         json.dump(dict(reconvert_shared_content), f)
    # elif config.use_exact_match_feature:
    with open(shared_path) as f:
        shared_content = {}
        load_shared_content(f, shared_content)
        # shared_content = json.load(f)
    return shared_content

def generate_pos_feature_tensor(parse, left_padding_and_cropping_pairs):
    pos = parsing_parse(parse)
    pos_vector = [(idx, POS_dict.get(tag, 0)) for idx, tag in enumerate(pos)]
    return construct_one_hot_feature_tensor(pos_vector, left_padding_and_cropping_pairs, 2, column_size=len(POS_Tagging))

def generate_quora_pos_feature_tensor(parses, left_padding_and_cropping_pairs):
    pos = parse.split()
    pos_vector = [(idx, POS_dict.get(tag, 0)) for idx, tag in enumerate(pos)]
    return construct_one_hot_feature_tensor(pos_vector, left_padding_and_cropping_pairs, 2, column_size=len(POS_Tagging))


def construct_one_hot_feature_tensor(sequence, pad_crop_pair, dim, column_size=None, dtype=np.int32):
    """
    sequences: [[(idx, val)... ()]...[]]
    left_padding_and_cropping_pairs: [[(0,0)...] ... []]
    """
    left_padding, left_cropping = pad_crop_pair
    if dim == 1:
        vec = np.zeros((universal_config['seq_length']))
        for num in sequence:
            if num + left_padding - left_cropping < universal_config['seq_length'] and num + left_padding - left_cropping >= 0:
                vec[num + left_padding - left_cropping] = 1
        return vec
    elif dim == 2:
        assert column_size
        mtrx = np.zeros((universal_config['seq_length'], column_size))
        for row, col in sequence:
            if row + left_padding - left_cropping < universal_config['seq_length'] and row + left_padding - left_cropping >= 0 and col < column_size:
                mtrx[row + left_padding - left_cropping, col] = 1
        return mtrx
    else:
        raise NotImplementedError

    #return vec

def parsing_parse(parse):
    base_parse = [s.rstrip(" ").rstrip(")") for s in parse.split("(") if ")" in s]
    pos = [pair.split(" ")[0] for pair in base_parse]
    return pos



if __name__ == '__main__':
    dpath = '../data/snli_1.0/snli_1.0_fake_train.jsonl'
    x = load_data(dpath)
    x1 = sentences_to_padded_index_sequences([x])