# -*- coding: UTF-8 -*-


# date  : 2017.12.22

"""
definition of KIM model
"""

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, rnn
from utils import find_wordnet_rel

# class KnowledgeEnrichedCoAttention(nn.Block):
#     def __init__(self, **kwargs):
#         super(KnowledgeEnrichedCoAttention, self).__init__(**kwargs)
#         self.kb = init_kb()
#         with self.name_scope():
#             self.attention_h = nn.soft

iszero = lambda x: sum(x != 0).asscalar() == 0


def F(m):
    """
    1
    m: (batch_size, seq_len, seq_len, 5)
    """
    out = nd.zeros(m.shape[:3])
    for ba in range(m.shape[0]):
        for i in range(m.shape[1]):
            for j in range(m.shape[2]):
                if not iszero(m[ba][i][j]):
                    out[ba][i][j] = 1
    return out

def get_co_attention(k_lambda):

    def _get_co_attention(as_, bs_, r, lamb=k_lambda):
        """
        as_, bs_: (batch_size, seq_len, embed_size)
        r: (batch_size, seq_len, seq_len, 5)
        """
        e = nd.batch_dot(as_, bs_, transpose_b=True) + lamb * F(r)   # (batch_size, seq_len, seq_len,)
        alpha = nd.softmax(e, axis=2)  # alpha_ij = exp(eij) / SUM_k(exp(eik))
        beta = nd.softmax(e, axis=1)   # beta_ij = exp(ij) / SUM_k(exp(ekj))
        ac = nd.batch_dot(alpha, bs_)               # 
        bc = nd.batch_dot(beta, as_, transpose_a=True)
        return ac, bc, alpha, beta
    return _get_co_attention


class InferenceComposition(nn.Block):
    def __init__(self, params, **kwargs):
        super(InferenceComposition, self).__init__(**kwargs)
        composi_hidden_size = params['composi_hidden_size']
        pool_size = params['pool_size']
        strides = params['strides']
        composi_dropout = params['composi_dropout']
        weight_pool_dense_size = params['weight_pool_dense_size']
        with self.name_scope():  
            self.inference_composition_a = nn.Sequential()
            self.inference_composition_b = nn.Sequential()
            self.inference_composition_a.add(rnn.LSTM(hidden_size=composi_hidden_size,
                dropout=composi_dropout, bidirectional=True))
            self.inference_composition_b.add(rnn.LSTM(hidden_size=composi_hidden_size,
                dropout=composi_dropout, bidirectional=True))
                # mean, max pooling
            self.mean_pooling_a = nn.AvgPool2D(pool_size=pool_size, strides=strides)
            self.mean_pooling_b = nn.AvgPool2D(pool_size=pool_size, strides=strides)
            self.max_pooling_a = nn.MaxPool2D(pool_size=pool_size, strides=strides)
            self.max_pooling_b = nn.MaxPool2D(pool_size=pool_size, strides=strides)
                # weight pooling
            self.weight_pooling_dense_a = nn.Dense(weight_pool_dense_size, activation='relu')
            self.weight_pooling_dense_b = nn.Dense(weight_pool_dense_size, activation='relu')
                # final MLP
            self.final_mlp = nn.Dense(3, activation='tanh')

    def forward(self, am, bm, alpha, beta, r):
        av = self.inference_composition_a(am)
        bv = self.inference_composition_b(bm)
        max_pool_a = self.max_pooling_a(av)
        max_pool_b = self.max_pooling_b(bv)
        mean_pool_a = self.mean_pooling_a(av)
        mean_pool_b = self.mean_pooling_b(bv)
        weight_pool_weight_a = nd.softmax(self.weight_pooling_dense_a(nd.dot(alpha, r)))
        weight_pool_weight_b = nd.softmax(self.weight_pooling_dense_b(nd.dot(beta, r)))
        aw = weight_pool_weight_a * av
        bw = weight_pool_weight_b * bv
        out = self.final_mlp(nd.concat(max_pool_a, mean_pool_aj, aw, max_pool_b, mean_pool_b, bw))
        return out


class Kim(nn.Block):
    def __init__(self, params, verbose=False, **kwargs):
        super(Kim, self).__init__(**kwargs)
        self.verbose = verbose

        vocab_size = params['vocab_size'] 
        embed_size = params['embed_size']
        encode_hidden_size = params['encode_hidden_size']
        encode_dropout  = params['encode_dropout']
        k_lambda = params['k_lambda']
        local_infer_dense_size = params['local_infer_dense_size']
        pool_size = params['pool_size']
        strides = params['strides']
        weight_pool_dense_size = params['weight_pool_dense_size']

        with self.name_scope():
            # first block: input_encoding
            self.input_encoding_layer_a = nn.Sequential()
            self.input_encoding_layer_b = nn.Sequential()
                # embedding
            self.input_encoding_layer_a.add(nn.Embedding(vocab_size, embed_size, 
                weight_initializer=None))  # should initialize with pre-trained word embedding
            self.input_encoding_layer_b.add(nn.Embedding(vocab_size, embed_size, 
                weight_initializer=None))
                # encoder: bilstm
            self.input_encoding_layer_a.add(rnn.LSTM(hidden_size=encode_hidden_size,
                dropout=encode_dropout, bidirectional=True))
            self.input_encoding_layer_b.add(rnn.LSTM(hidden_size=encode_hidden_size,
                dropout=encode_dropout, bidirectional=True))

            # second block: knowledge-enriched co-attention (a function used in forward)
            self.co_attention = get_co_attention(k_lambda)

            # third block: knowledge-enriched local inference collection
            self.local_inference_a = nn.Dense(local_infer_dense_size, activation='relu')
            self.local_inference_b = nn.Dense(local_infer_dense_size, activation='relu')

            #fourth block: knowledge-enriched inference composition
            self.inference_composition = InferenceComposition(params)

    def forward(self, data):
        data_, r = data
        a, b = data_[:,0], data_[:,1]   # (batch_size, seq_length)
        as_ = self.input_encoding_layer_a(a)
        bs_ = self.input_encoding_layer_b(b)
        ac, bc, alpha, beta = self.co_attention(as_, bs_, r)
        am = nd.concat(self.local_inference_a(nd.concat(as_, ac, as_-ac, as_*ac)), nd.batch_dot(alpha, r))
        bm = nd.concat(self.local_inference_b(nd.concat(bs_, bc, bs_-bc, bs_*bc)), nd.batch_dot(beta, r)) # maybe buggy
        out = slef.inference_composition(am, bm, alpha, beta, r)
        return out

def get_kim_model(params, **kwargs):
    return Kim(params, **kwargs)


