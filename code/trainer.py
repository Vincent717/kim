# -*- coding: UTF-8 -*-


# date  : 2017.12.23

"""
training
"""

import sys
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, rnn

from kim import get_kim_model
from data_processer import load_data, map_to_index
from conf import conf_params
import utils
import logging
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='kim_train_%s.log' % datetime.now().strftime('%y%m%d_%H%M'),
                filemode='w')


def train(model_type='kim', params_index=0):
    print('start training')
    # loda params
    train_data_path = conf_params[params_index]['train_data_path']
    test_data_path = conf_params[params_index]['test_data_path']
    word_vec_path = conf_params[params_index]['word_vec_path']
    save_model_path = conf_params[params_index]['save_model_path']
    lr = conf_params[params_index]['lr']
    grad_clip = conf_params[params_index]['grad_clip']
    optimizer = conf_params[params_index]['optimizer']
    batch_size = conf_params[params_index]['batch_size']
    epoches = conf_params[params_index]['epoch']
    gpu_index = conf_params[params_index]['gpu']
    params = conf_params[params_index]['params']
    ctx=utils.try_gpu(gpu_index)

    # load data
    print('try to load data...')
    train_data, i2w, w2i = load_data(train_data_path, ctx=ctx)
    test_data = load_data(test_data_path, ctx=ctx, is_train=False)
    test_data = map_to_index(test_data, w2i, ctx)
    print('vocab size ', len(w2i))
    train_data = utils.DataLoader(train_data, batch_size, ctx)
    test_data = utils.DataLoader(test_data, batch_size, ctx)
    print('data loaded!', len(train_data), len(test_data))
    word_vec = utils.load_word_vec(word_vec_path)

    print('try to initialize model...')
    if model_type == 'kim':
        net = get_kim_model(params, i2w=i2w, word_vec=word_vec, ctx=ctx)
        print('build kim model')
    else:
        net = get_diim_model(params, i2w=i2w, word_vec=word_vec, ctx=ctx)
        print('build diim model')
    net.initialize(ctx=ctx)
    #print(net.collect_params())

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), optimizer, {'learning_rate': lr, 'clip_gradient': grad_clip})
    print('model initialized!', net, ctx)
    print('start training...')
    for epoch in tqdm(range(epoches)):
        train_loss = 0.
        train_acc = 0.
        for data, label in tqdm(train_data):  # (32,2,20), (32,)
            word_sequences = utils.get_word_sequences(data, i2w)
            r = utils.find_wordnet_rel(word_sequences, ctx)
            #r= nd.ones([32, 30, 30, 5],ctx=ctx)
            with mx.autograd.record():
                output = net((data, r))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        print('train accuracy')
        test_acc = utils.evaluate_accuracy(test_data, net, i2w, ctx=[ctx])
        print('test accuracy')
        line = "Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc)
        print(line)
        logging.info(line)
        # save model
        net.save_params(save_model_path + '.' + str(epoch))
    print('finish train')



def main(params_index):
    train(params_index)

def predictor(params_index):
    params = conf_params[params_index]['params']
    net = get_kim_model(params)
    ctx=utils.try_gpu()
    net.load_params('../model/kim_1_171226_1834.params', ctx=ctx)
    print(net)

if __name__ == '__main__':
    index = int(sys.argv[1])
    main(index)
    #predictor(0)