from torch import nn
from torch.nn import Module, Linear, Sequential
from torch.nn.init import normal_, xavier_normal_, constant_
from numpy import eye as np_eye, array as np_array, power as np_power, isinf as np_isinf, diag as np_diag

class XavierLinear(Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''

    def __init__(self, in_features, out_features, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias)
        xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class MLP(Module):
    def __init__(self, dim_in_hid_out, act_fn='ReLU', last_act=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dim_in_hid_out) - 1):
            layers.append(XavierLinear(dim_in_hid_out[i], dim_in_hid_out[i + 1]))
            if i < len(dim_in_hid_out) - 2 or last_act:
                layers.append(getattr(nn, act_fn)())
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def layer_init(layer, init_para=0.1, normal=False, xavier=True):
    xavier = False if normal is True else True
    if normal is True:
        normal_(layer.weight, mean=0, std=init_para)
        if layer.bias is not None:
            constant_(layer.bias, 0)
        return
    elif xavier is True:
        xavier_normal_(layer.weight, gain=1.0)
        if layer.bias is not None:
            constant_(layer.bias, 0)
        return


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np_array(mx.sum(1))
    r_inv = np_power(rowsum, -1).flatten()
    r_inv[np_isinf(r_inv)] = 0.
    r_mat_inv = np_diag(r_inv)
    return r_mat_inv.dot(mx)


def adj_normalize(adj):
    return normalize(adj + np_eye(adj.shape[0]))
