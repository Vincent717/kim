# -*- coding: UTF-8 -*-


# date  : 2017.12.23

"""
conf
"""


conf_params = dict()

universal_config = dict()

universal_config = {
	'embedding_replacing_rare_word_with_UNK' = True,
	'seq_length': 20,
}

# paper params
conf_params[1] = {
	'lr': 0.0004,
	'batch_size': 32,
	'dropout': 0.5,
	'epoch': 500,
	'train_data_path': '../data/snli_1.0/snli_1.0/snli_1.0_fake_train.jsonl',
	'test_data_path': '../data/snli_1.0/snli_1.0/snli_1.0_fake_train.jsonl',
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
	},
}