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
	'seq_length': 25,
}

# paper params
conf_params[0] = {
	'lr': 0.0004,
	'batch_size': 32,
	'dropout': 0.5,
	'epoch': 5,
	'train_data_path': '../data/snli_1.0/snli_1.0_fake_train.jsonl',
	'test_data_path': '../data/snli_1.0/snli_1.0_fake_train.jsonl',
	'save_model_path': '../model/kim_1_%s.params' % datetime.now().strftime('%y%m%d_%H%M'),
	'params': {
		'vocab_size': 1000,
		'embed_size': 300,
		'encode_hidden_size': 300,
		'encode_dropout': 0.5,
		'k_lambda': 0.5,  # 0.1 0.2 0.5 1 2 5 10 20 50
		'local_infer_dense_size': 300,
		'pool_size': 2,
		'strides': 1,
		'weight_pool_dense_size': 300,
		'composi_hidden_size': 300,
		'composi_dropout': 0.5,
	},
}