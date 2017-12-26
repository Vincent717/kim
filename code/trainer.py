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
from data_processer import load_data, map_to_index
from kim_conf import conf_params
import utils
import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='kim_train.log',
                filemode='w')


def train(params_index=0):
    print('start training')
    # loda params
    train_data_path = conf_params[params_index]['train_data_path']
    test_data_path = conf_params[params_index]['test_data_path']
    save_model_path = conf_params[params_index]['save_model_path']
    lr = conf_params[params_index]['lr']
    batch_size = conf_params[params_index]['batch_size']
    epoches = conf_params[params_index]['epoch']
    params = conf_params[params_index]['params']

    # load data
    print('try to load data...')
    train_data, i2w, w2i = load_data(train_data_path)
    test_data = load_data(test_data_path, is_pure=False)
    test_data = map_to_index(test_data, w2i)
    #print(train_data,329)
    train_data = utils.DataLoader(train_data, batch_size)
    test_data = utils.DataLoader(test_data, batch_size)
    print('data loaded!', len(train_data), len(test_data))

    print('try to initialize model...')
    net = get_kim_model(params)
    ctx=utils.try_gpu()
    net.initialize(ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    print('model initialized!', net)

    print('start training...')
    for epoch in range(epoches):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:  # (32,2,20), (32,)
            word_sequences = utils.get_word_sequences(data, i2w)
            r= utils.find_wordnet_rel(word_sequences)
            with mx.autograd.record():
                output = net((data, r))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)

        test_acc = utils.evaluate_accuracy(test_data, net, i2w)
        line = "Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc)
        print(line)
        logging.log(line)
        # save model
        print('finish train')
        net.save_params(save_model_path + '.' + str(epoch))


def main(params_index):
    train(params_index)

def predictor(params_index):
    params = conf_params[params_index]['params']
    net = get_kim_model(params)
    ctx=utils.try_gpu()
    net.load_params('../model/kim_1_171226_1834.params', ctx=ctx)
    print(net)

if __name__ == '__main__':
    main(1)
    #predictor(0)