# -*- coding: UTF-8 -*-


# date  : 2017.12.23

"""
data processing
"""

from tqdm import tqdm
from kim_conf import universal_config


PADDING = "<PAD>"


def load_data(data_path, snli=True):
	return load_nli_data(data_path, snli=snli)


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

def sentences_to_padded_index_sequences(datasets):
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

    # char level, omit
    #char_vocab = set([char for char in char_counter])
    #char_vocab = list(char_vocab)
    #char_vocab = [PADDING] + char_vocab
    #char_indices = dict(zip(char_vocab, range(len(char_vocab))))
    #indices_to_char = {v: k for k, v in char_indices.items()}
    

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
                            index = word_indices[token_sequence[i]] if word_counter[token_sequence[i]] >= config.UNK_threshold else word_indices["<UNK>"]
                        else:
                            index = word_indices[token_sequence[i]]
                        itf = 1 / (word_counter[token_sequence[i]] + 1)
                    example[sentence + '_index_sequence'][i] = index
                    
                    example[sentence + '_inverse_term_frequency'][i] = itf
                
                # char level, omit
                #example[sentence + '_char_index'] = np.zeros((universal_config["seq_length"], config.char_in_word_size), dtype=np.int32)
                # for i in range(universal_config["seq_length"]):
                #     if i >= len(token_sequence):
                #         continue
                #     else:
                #         chars = [c for c in token_sequence[i]]
                #         for j in range(config.char_in_word_size):
                #             if j >= (len(chars)):
                #                 break
                #             else:
                #                 index = char_indices[chars[j]]
                #             example[sentence + '_char_index'][i,j] = index 
    

    return indices_to_words, word_indices  #, char_indices, indices_to_char



