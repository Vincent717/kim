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
            W_itrAtt = nd.random_normal(shape=(embed_size*3,))  # change to dense?
            params.append(W_itrAtt)
            for param in params:
                param.attach_grad()
            
            if self.need_fuse_gate:
                self.z_dense = nn.Dense(embed_size, activation='tanh', flatten=False)
                self.r_dense = nn.Dense(embed_size, activation='sigmoid', flatten=False)
                self.f_dense = nn.Dense(embed_size, activation='sigmoid', flatten=False)
                # W1 = nd.random_normal(shape=(embed_size*2, embed_size))
                # W2 = nd.random_normal(shape=(embed_size*2, embed_size))
                # W3 = nd.random_normal(shape=(embed_size*2, embed_size))
                # b1 = nd.zeros(embed_size)
                # b2 = nd.zeros(embed_size)
                # b3 = nd.zeros(embed_size)
                # params += [W1, b1, W2, b2, W3, b3]

    def forward(self, p):
        plen = p.shape[1]
        dim = p.shape[-1]
        p_aug_1 = nd.tile(nd.expand_dims(p, 2), [1,1,plen,1]) # (batch_size, seq_len, seq_len, embed_dim)
        p_aug_2 = nd.tile(nd.expand_dims(p, 1), [1,plen,1,1]) # (32, 40, 40, 300)
        p_elem_wise_mp = p_aug_1 * p_aug_2
        p_concat = nd.concat(p_aug_1, p_aug_2, p_elem_wise_mp, dim=3)
        self_attend = nd.dot(p_concat, W_itrAtt)
        self_attend_softmax = nd.softmax(self_attend, axis=0)
        p_itrAtt =  nd.batch_dot(self_attend_softmax, p, transpose_a=True)
        # fuse gate
        if self.need_fuse_gate:
            p_hw_itrAtt = nd.concat(p, p_itrAtt, dim=3)
            z = self.z_dense(p_hw_itrAtt)
            r = self.r_dense(p_hw_itrAtt)
            f = self.f_dense(p_hw_itrAtt)
            p_enc = r * p + f * z
            return p_enc, self_attend

        return p_itrAtt, self_attend



class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(self.conv_block(growth_rate))

    def conv_block(self, channels):
        out = nn.Sequential()
        out.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(channels, kernel_size=3, padding=1)
        )
        return out

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x


def transition_block(channels):
    out = nn.Sequential()
    out.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return out


def dense_net():
    net = nn.Sequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=7,
                      strides=2, padding=3),
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        )
        # dense blocks
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers)-1:
                net.add(transition_block(channels//2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes)
        )
    return net