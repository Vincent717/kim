# -*- coding: UTF-8 -*-


# date  : 2017.12.22

"""
definition of KIM model
"""

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, rnn


# class KnowledgeEnrichedCoAttention(nn.Block):
#     def __init__(self, **kwargs):
#         super(KnowledgeEnrichedCoAttention, self).__init__(**kwargs)
#         self.kb = init_kb()
#         with self.name_scope():
#             self.attention_h = nn.soft

def get_co_attention(k_lambda):
    def 

class InferenceComposition(nn.Block):
    def __init__(self, **kwargs):
        super(InferenceComposition, self).__init__(**kwargs)
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
            self.inference_composition = InferenceComposition()

    def forward(self, data):
        a, b = data
        r = load_wordnet_rel(a, b)
        as_ = self.input_encoding_layer_a(a)
        bs_ = self.input_encoding_layer_b(b)
        ac, alpha = self.co_attention(as_, r)
        bc, beta = self.co_attention(bs_, r)
        am = nd.concat(self.local_inference_a(nd.concat(as_, ac, as_-ac,nd.dot(as_, ac))), alpha * r)
        bm = nd.concat(self.local_inference_b(nd.concat(bs_, bc, bs_-bc,nd.dot(bs_, bc))), beta * r)
        out = slef.inference_composition(am, bm, alpha, beta, r)
        return out

def get_kim_model(params, **kwargs):
    Kim(params, **kwargs)


