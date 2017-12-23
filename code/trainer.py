# -*- coding: UTF-8 -*-


# date  : 2017.12.23

"""
training
"""

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, rnn

from kim import get_kim_model
from data_processer import load_data, sentences_to_padded_index_sequences
from kim_conf import conf_params
import utils


def train(data_path, save_model_path, params_index):
	# loda params
	train_data_path = conf_params[params_index]['train_data_path']
	test_data_path = conf_params[params_index]['test_data_path']
	lr = conf_params[params_index]['lr']
	batch_size = conf_params[params_index]['batch_size']
	epoches = conf_params[params_index]['epoch']
	params = conf_params['params_index']['params']

	# load data
	train_data = pretty_format(load_data(train_data_path))
	test_data = pretty_format(load_data(test_data_path))

	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
	net = get_kim_model()

	for epoch in range(epoches):
	    train_loss = 0.
	    train_acc = 0.
	    for data, label in train_data:
	        with autograd.record():
	            output = net(data)
	            loss = softmax_cross_entropy(output, label)
	        loss.backward()
	        trainer.step(batch_size)

	        train_loss += nd.mean(loss).asscalar()
	        train_acc += utils.accuracy(output, label)

	    test_acc = utils.evaluate_accuracy(test_data, net)
	    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
	        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))


def main(data_path, save_model_path, params_index):
	train(data_path, save_model_path, params_index)



if __name__ == '__main__':
	main()