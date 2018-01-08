# -*- coding: UTF-8 -*-


# date  : 2018.01.05

"""
definition of DIIM model
"""

import mxnet as mx
from mxnet import nd, init
from mxnet import gluon
from mxnet.gluon import nn, rnn
from utils import find_wordnet_rel, try_gpu
from nn_utils import HighwayLayer, SelfAttentionLayer
import random


iszero = lambda x: sum(x != 0).asscalar() == 0

_ctx = try_gpu()

def F(m, ctx=_ctx):
    """
    1
    m: (batch_size, seq_len, seq_len, 5)
    """
    out = nd.zeros(m.shape[:3], ctx=ctx)
    for ba in range(m.shape[0]):
        for i in range(m.shape[1]):
            for j in range(m.shape[2]):
                if not iszero(m[ba][i][j]):
                    out[ba][i][j] = 1
    return out


class Diim(nn.Block):
    def __init__(self, params, ctx=_ctx, i2w='', word_vec='', verbose=False, **kwargs):
        super(Diim, self).__init__(**kwargs)
        self.verbose = verbose
        vocab_size = len(i2w) if i2w else params['vocab_size'] 
        highway_dense_size = params['highway_dense_size']
        highway_num = params['highway_num']
        keep_rate = params['keep_rate']
        # embed_size = len(word_vec['.'].split()) if word_vec else params['embed_size']
        # encode_hidden_size = params['encode_hidden_size']
        # encode_dropout  = params['encode_dropout']
        # k_lambda = params['k_lambda']
        # local_infer_dense_size = params['local_infer_dense_size']
        # pool_size = params['pool_size']
        # strides = params['strides']
        # weight_pool_dense_size = params['weight_pool_dense_size']


        with self.name_scope():
            # first block: (concat) embedding
                # word embedding (with dropout)
            self.word_embedding_layer_a = nn.Sequential()
            self.word_embedding_layer_a.add(nn.Embedding(vocab_size, embed_size, 
                weight_initializer=MyEmbedInit(i2w,word_vec,vocab_size,embed_size,ctx)))  # should initialize with pre-trained word embedding
            self.word_embedding_layer_a.add(nn.Dropout(keep_rate))
            self.word_embedding_layer_b = nn.Sequential()
            self.word_embedding_layer_b.add(nn.Embedding(vocab_size, embed_size, 
                weight_initializer=MyEmbedInit(i2w,word_vec,vocab_size,embed_size,ctx)))
            self.word_embedding_layer_b.add(nn.Dropout(keep_rate))
                # character embedding
            self.char_embedding_layer_a = nn.Sequential()
            self.char_embedding_layer_b = nn.Sequential()
            self.char_embedding_layer_a.add(nn.Embedding(char_vocab_size, embed_size))  # initializer?
            self.char_embedding_layer_b.add(nn.Embedding(char_vocab_size, embed_size))
            char_feature_conv = nn.Sequential()
            char_feature_conv.add(Conv1D(channel, kernel_size)) # filter_size, height
            char_feature_conv.add(nn.Maxpool1D)
            self.char_embedding_layer_a.add(char_feature_conv)
            self.char_embedding_layer_b.add(char_feature_conv)

            # second block: encoding layer
                # highway network
            self.highway_layer = nn.Sequential()
            for _ in range(highway_num):
                self.highway_layer.add(HighwayLayer(highway_dense_size))
                # self-attention (do not share, but should penalize)
            self.self_attention_layer_a = SelfAttentionLayer(need_fuse_gate=True)
            self.self_attention_layer_b = SelfAttentionLayer(need_fuse_gate=True)

            # third block: Interaction layer
            self.interaction_layer = cross_elem_wise_mp(keep_rate=keep_rate)

            #fourth block: feature extraction layer
            self.feature_extraction_layer = DenseNet(params)

            # fifth block: output layer
            self.output_layer = nn.Dense(3, activation='tanh')

    def forward(self, data):
        data_, r = data
        ab, ab_char, ab_pos, ab_exact = data_[:,0], data_[:,1], data_[:,2], data_[:,3]   # (batch_size, seq_length)
        
        # step 1: input embedding
        p_word = self.word_embedding_layer_a(ab[:,0])
        h_word = self.word_embedding_layer_b(ab[:,1])
        p_char = self.char_embedding_layer_a(ab_char[:,0]) 
        h_char = self.char_embedding_layer_b(ab_char[:,1])
        p_pos, h_pos = ab_pos[:,0], ab_pos[:,1]
        p_exact, h_exact = ab_exact[:,0], ab_exact[:,1]
        p = nd.concat(p_word, p_char, p_pos, p_exact, dim=3)
        h = nd.concat(h_word, h_char, h_pos, h_exact, dim=3)
        
        # step 2: input encoding
        p_hw = self.highway_layer(p)
        h_hw = self.highway_layer(h)
        p_enc, p_self_attend = self.self_attention_layer_a(p_hw)
        h_enc, h_self_attend = self.self_attention_layer_b(h_hw)

        # step 3: interaction
        I = self.interaction_layer(p_enc, h_enc)

        # step 4: feature extraction
        feature = self.feature_extraction_layer(I)

        # step 5: final mlp
        out = self.output_layer(feature)

        return out


class MyEmbedInit(init.Initializer):
    def __init__(self, i2w, word_vec, vocab_size, embed_size, ctx):
        super(MyEmbedInit, self).__init__()
        self._verbose = True
        self.i2w = i2w
        self.word_vec = word_vec
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.ctx = ctx

    def _init_weight(self, _, arr):
        # 初始化权重，使用out=arr后我们不需指定形状
        self.init_embedding()

    def init_embedding(self):
        if self.i2w and self.word_vec:
            out = []
            for i in range(len(self.i2w)):
                word = self.i2w[i]
                if word in self.word_vec:
                    out.append([float(i) for i in self.word_vec[word].split()])
                else:
                    out.append([random.uniform(0,1) for _ in range(self.embed_size)])
            return nd.array(out, ctx=self.ctx)
        else:
            return nd.random.uniform(shape=[self.vocab_size, self.embed_size], ctx=self.ctx)


def cross_elem_wise_mp(keep_rate=0.5):
    """
    p, h : (32, 40, 300)
    output: (32, 40, 40, 300)
    """
    def _cross_element_wise_mp(p, h):
        plen = p.shape[1]
        plen = h.shape[1]
        # order is important
        p_expand = nd.tile(nd.expand_dims(p, 2), [1,1,plen,1]) # (batch_size, seq_len, seq_len, embed_dim)
        h_expand = nd.tile(nd.expand_dims(p, 1), [1,hlen,1,1]) # (32, 40, 40, 300)
        out = p_expand * h_expand
        if interact_dropout != 1:
            out = nn.Dropout(keep_rate)(out)
        return out
    return _cross_element_wise_mp

def get_diim_model(params, **kwargs):
    return Diim(params, **kwargs)


