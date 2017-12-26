# -*- coding: UTF-8 -*-


# date  : 2017.12.22

"""
definition of KIM model
"""

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, rnn
from utils import find_wordnet_rel, try_gpu

# class KnowledgeEnrichedCoAttention(nn.Block):
#     def __init__(self, **kwargs):
#         super(KnowledgeEnrichedCoAttention, self).__init__(**kwargs)
#         self.kb = init_kb()
#         with self.name_scope():
#             self.attention_h = nn.soft

iszero = lambda x: sum(x != 0).asscalar() == 0

_ctx = try_gpu()

def F(m):
    """
    1
    m: (batch_size, seq_len, seq_len, 5)
    """
    out = nd.zeros(m.shape[:3], ctx=_ctx)
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
                # weight pooling
            self.weight_pooling_dense_a = nn.Dense(weight_pool_dense_size, activation='relu', flatten=False)
            self.weight_pooling_dense_b = nn.Dense(weight_pool_dense_size, activation='relu', flatten=False)
                # final MLP
            self.final_mlp = nn.Dense(3, activation='tanh')

    def forward(self, am, bm, alpha_r, beta_r):
        av = self.inference_composition_a(am)   # (batch_size, seq_len, hidden*2) (32,45,600)
        bv = self.inference_composition_b(bm)   
        max_pool_a = nd.max(av, axis=1)
        max_pool_b = nd.max(bv, axis=1)
        mean_pool_a = nd.mean(av, axis=1)
        mean_pool_b = nd.mean(bv, axis=1)
        weight_pool_weight_a = nd.softmax(self.weight_pooling_dense_a(alpha_r))
        weight_pool_weight_b = nd.softmax(self.weight_pooling_dense_b(beta_r))
        aw = nd.sum(weight_pool_weight_a * av, axis=1)
        bw = nd.sum(weight_pool_weight_b * bv, axis=1)
        out = self.final_mlp(nd.concat(max_pool_a, mean_pool_a, aw, max_pool_b, mean_pool_b, bw))
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
            self.local_inference_a = nn.Dense(local_infer_dense_size, activation='relu', flatten=False)
            self.local_inference_b = nn.Dense(local_infer_dense_size, activation='relu', flatten=False)

            #fourth block: knowledge-enriched inference composition
            self.inference_composition = InferenceComposition(params)

    def forward(self, data):
        data_, r = data
        a, b = data_[:,0], data_[:,1]   # (batch_size, seq_length)
            # first step: input encoding
        as_ = self.input_encoding_layer_a(a)
        bs_ = self.input_encoding_layer_b(b)
            
            # second stpe: co-attention, alpha_r and beta_r
            # alpha_r.shape = (batch_size, seq_len, 5)
        ac, bc, alpha, beta = self.co_attention(as_, bs_, r)
        alpha_r = nd.batch_dot(r[0], alpha[0].reshape(tuple(list(alpha.shape)[1:] + [1])), transpose_a=True)
        beta_r = nd.batch_dot(r[0], beta[0].reshape(tuple(list(beta.shape)[1:] + [1])), transpose_a=True)
        alpha_r = alpha_r.reshape(tuple([1] + list(alpha_r.shape)[:-1]))
        beta_r = beta_r.reshape(tuple([1] + list(beta_r.shape)[:-1]))
        for i in range(1, r.shape[0]):
            tmp = nd.batch_dot(r[i], alpha[i].reshape(tuple(list(alpha.shape)[1:] + [1])), transpose_a=True)
            tmp = tmp.reshape(tuple([1] + list(tmp.shape)[:-1]))
            alpha_r = nd.concat(alpha_r, tmp, dim=0)
        for i in range(1, r.shape[0]):
            tmp = nd.batch_dot(r[i], beta[i].reshape(tuple(list(beta.shape)[1:] + [1])), transpose_a=True)
            tmp = tmp.reshape(tuple([1] + list(tmp.shape)[:-1]))
            beta_r = nd.concat(beta_r, tmp, dim=0)
        #beta_r = nd.concat([nd.batch_dot(beta,r[:,:,:,i], transpose_a=True).reshape(tuple(list(beta.shape) + [1])) for i in range(r.shape[-1])] ,dim=3)

            # third step: local inference
            # am.shape: (batch_size, seq_len, hidden_size)
        am = self.local_inference_a(nd.concat(as_, ac, as_-ac, as_*ac, alpha_r, dim=2))
        bm = self.local_inference_b(nd.concat(bs_, bc, bs_-bc, bs_*bc, beta_r, dim=2))
        out = self.inference_composition(am, bm, alpha_r, beta_r)
        return out

def get_kim_model(params, **kwargs):
    return Kim(params, **kwargs)


