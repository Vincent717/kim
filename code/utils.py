from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from mxnet import image
from mxnet.gluon import nn
import mxnet as mx
import numpy as np
from time import time
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import pickle

def try_gpu(i=0):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(i)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

_ctx = try_gpu()

class DataLoader(object):
    """similiar to gluon.data.DataLoader, but might be faster.

    The main difference this data loader tries to read more exmaples each
    time. But the limits are 1) all examples in dataset have the same shape, 2)
    data transfomer needs to process multiple examples at each time
    """
    def __init__(self, dataset, batch_size, ctx=_ctx, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx

    def __iter__(self):
        data = self.dataset[:]
        X = data[0] # X is a tuple
        y = nd.array(data[1], ctx=self.ctx)
        n = X[0].shape[0]
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = (nd.array(x_.asnumpy()[idx], ctx=self.ctx) for x_ in X)
            y = nd.array(y.asnumpy()[idx], ctx=self.ctx)

        for i in range(n//self.batch_size):
            yield ((x_[i*self.batch_size:(i+1)*self.batch_size] for x_ in X),
                   y[i*self.batch_size:(i+1)*self.batch_size])

    def __len__(self):
        return len(self.dataset[0][0])//self.batch_size

def load_data_fashion_mnist(batch_size, resize=None, root="~/.mxnet/datasets/fashion-mnist"):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        # transform a batch of examples
        if resize:
            n = data.shape[0]
            new_data = nd.zeros((n, resize, resize, data.shape[3]), ctx=_ctx)
            for i in range(n):
                new_data[i] = image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x weight x channel to batch x channel x height x weight
        return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')
    mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=transform_mnist)
    train_data = DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = DataLoader(mnist_test, batch_size, shuffle=False)
    return (train_data, test_data)



def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])

def evaluate_accuracy(data_iterator, net, i2w, ctx=[_ctx]):  #[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0], ctx=_ctx)
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            word_sequences = get_word_sequences(X, i2w)
            r = find_wordnet_rel(word_sequences, ctx=ctx[0])
            acc += nd.sum(net((X,r)).argmax(axis=1)==y) #.copyto(mx.cpu())
            n += y.size
        acc.wait_to_read() # don't push too many operators into backend
    return acc.asscalar() / n

def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    """Train a network"""
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0
        if isinstance(train_data, mx.io.MXDataIter):
            train_data.reset()
        start = time()
        for i, batch in enumerate(train_data):
            data, label, batch_size = _get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()
            train_acc += sum([(yhat.argmax(axis=1)==y).sum().asscalar()
                              for yhat, y in zip(outputs, label)])
            train_loss += sum([l.sum().asscalar() for l in losses])
            trainer.step(batch_size)
            n += batch_size
            m += sum([y.size for y in label])
            if print_batches and (i+1) % print_batches == 0:
                print("Batch %d. Loss: %f, Train acc %f" % (
                    n, train_loss/n, train_acc/m
                ))

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %.3f, Train acc %.2f, Test acc %.2f, Time %.1f sec" % (
            epoch, train_loss/n, train_acc/m, test_acc, time() - start
        ))




get_synsets = lambda x: set(wn.synsets(x))

def is_syn(x,y):
    return 1 if x & y else 0

def is_ant(x, y):
    for xi in x:
        for li in xi.lemmas():
            ants = [la.synset() for la in li.antonyms()]
            if set(ants) & y:
                return 1
    return 0

def is_hypernymy(x, y, n_type='avg'):
    diss = []
    for xi in x:
        xi_hyp_set = xi.hypernym_distances()
        for xih, dis in xi_hyp_set:
            if xih in y:
                diss.append(dis)
    if not diss:
        return 0
    else:
        if n_type == 'max':
            n = float(max(diss))
        elif n_type == 'min':
            n = float(min(diss))
        else:
            n = float(sum(diss)/len(diss))
        return 1 - n/8

def is_hyponymy(x, y, n_type='avg'):
    return is_hypernymy(y, x, n_type)

def is_same_hypernym(x, y):
    xh, yh = set(), set()
    for xi in x:
        xh.update(xi.hypernyms())
    for yi in y:
        yh.update(yi.hypernyms())
    return 1 if (is_syn(x,y)==0 and xh&yh) else 0

def find_wordnet_rel(word_seqs, ctx=_ctx):  #(batch_size, 2, seq_length)
    out = []
    #print(word_seqs, 88)
    for seqs in word_seqs:
        aout = []
        a, b = seqs[:]
        for ai in a:
            bout = []
            for bi in b:
                aw = get_synsets(ai)
                bw = get_synsets(bi)
                rel = [is_syn(aw, bw), is_ant(aw, bw), is_hypernymy(aw, bw),
                        is_hyponymy(aw, bw), is_same_hypernym(aw, bw)]
                bout.append(rel)    
            # bout.shape: (b_length, 5)
            aout.append(bout)
        # aout.shape: (a_length, b_length, 5)
        out.append(aout)
    # out.shape: (batch_size, a_length, b_length, 5)
    return nd.array(out, ctx=ctx)


def get_word_sequences(data, i2w):  #(batch_size, 2, seq_length)
    out = []
    for doc in data:
        stn_out = []
        for s in doc:
            stn_out.append([i2w[i.asscalar()] for i in s])
        out.append(stn_out)
    return out


def load_word_vec(wpath):
    with open(wpath, 'rb') as f:
        return pickle.loads(f.read())

def main():
    import sys
    aw = sys.argv[1]
    bw = sys.argv[2]
    a = get_synsets(aw)
    b = get_synsets(bw)
    print(aw, bw)
    print('syn: ', is_syn(a, b))
    print('ant: ', is_ant(a, b))
    print('hyper: ', is_hypernymy(a, b))
    print('hypon: ', is_hyponymy(a, b)) # just inverse of hyper
    print('same_hypernym: ', is_same_hypernym(a,b))


if __name__ == '__main__':
    main()