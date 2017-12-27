# -*- coding: UTF-8 -*-


# date  : 2017.12.23

"""
conf
"""
from datetime import datetime

conf_params = dict()

universal_config = dict()

universal_config = {
	'embedding_replacing_rare_word_with_UNK': True,
	'UNK_threshold': 1,
	'seq_length': 30,
}


# stats of seq_length
# 0.98 26
# 0.97 24
# 0.985 28
# 0.9 19
# 0.95 22
# 0.99 30
# 0.995 34
# max: 82



# paper params
conf_params[0] = {
	'lr': 0.0004,
	'batch_size': 32,
	'dropout': 0.5,
	'epoch': 100,
	'train_data_path': '../data/snli_1.0/snli_1.0_5w_train.jsonl',
	'test_data_path': '../data/snli_1.0/snli_1.0_dev.jsonl',
	'word_vec_path': '../data/glove.dic',
	'save_model_path': '../model/kim_1_%s.params' % datetime.now().strftime('%y%m%d_%H%M'),
	'gpu': 0,
	'params': {
		'vocab_size': 6525,
		'embed_size': 300,
		'encode_hidden_size': 300,
		'encode_dropout': 0.5,
		'k_lambda': 0.5,  # 0.1 0.2 0.5 1 2 5 10 20 50
		'local_infer_dense_size': 300,
		'pool_size': 2,
		'strides': 1,
		'weight_pool_dense_size': 1,
		'composi_hidden_size': 300,
		'composi_dropout': 0.5,
	},
}

conf_params[1] = {
	'lr': 0.0004,
	'batch_size': 32,
	'dropout': 0.5,
	'epoch': 300,
	'train_data_path': '../data/snli_1.0/snli_1.0_train.jsonl',
	'test_data_path': '../data/snli_1.0/snli_1.0_dev.jsonl',
	'word_vec_path': '../data/glove.dic',
	'save_model_path': '../model/kim_1_%s.params' % datetime.now().strftime('%y%m%d_%H%M'),
	'gpu': 0,
	'params': {
		'vocab_size': 1000,
		'embed_size': 300,
		'encode_hidden_size': 300,
		'encode_dropout': 0.5,
		'k_lambda': 0.5,  # 0.1 0.2 0.5 1 2 5 10 20 50
		'local_infer_dense_size': 300,
		'pool_size': 2,
		'strides': 1,
		'weight_pool_dense_size': 1,
		'composi_hidden_size': 300,
		'composi_dropout': 0.5,
	},
}