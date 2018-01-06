# -*- coding: UTF-8 -*-


# date  : 2018.01.0

"""
some network utils 
"""
from mxnet.gluon import nn

class HighwayLayer(nn.Block):
    def __init__(self, size, **kwargs):
        super(HighwayLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.high_fc = nn.Dense(size, activation='relu', flatten=False)
            self.high_trans_fc = nn.Dense(size, activation='sigmoid', flatten=False)

    def forward(self, data):
        trans = self.high_fc(data)
        gate = self.high_trans_fc(data)
        out = gate * trans + (1 - gate) * data
        return out


class SelfAttentionLayer(nn.Block):
    def __init__(self, embed_size, need_fuse_gate=False, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.need_fuse_gate = need_fuse_gate
        with self.name_scope():
            params = []
            W_itrAtt = nd.random_normal(shape=(embed_size*3,))
            params.append(W_itrAtt)
            
            if self.need_fuse_gate:
                W1 = nd.random_normal(shape=(embed_size*2, embed_size))
                W2 = nd.random_normal(shape=(embed_size*2, embed_size))
                W3 = nd.random_normal(shape=(embed_size*2, embed_size))
                b1 = nd.zeros(embed_size)
                b2 = nd.zeros(embed_size)
                b3 = nd.zeros(embed_size)
                params += [W1, b1, W2, b2, W3, b3]

            for param in params:
                param.attach_grad()


    def forward(self, data):
        trans = self.high_fc(data)
        gate = self.high_trans_fc(data)
        out = gate * trans + (1 - gate) * data
        return out

