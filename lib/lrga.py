from torch import ones as torch_ones, mm as torch_mm, sum as torch_sum, \
    cat as torch_cat, t as torch_t
from torch.nn import Module, ModuleList, Sequential, Linear, ReLU, \
    Dropout, BatchNorm1d
from torch.nn.init import xavier_normal_, constant_
from torch.nn.functional import relu as F_relu, dropout as F_dropout
# from torch_geometric.nn import GCNConv
from torch.cuda import current_device


CURRENT_DEVICE = current_device()


def joint_normalize2(U, V_T):
    # U and V_T are in block diagonal form
    tmp_ones = torch_ones((V_T.shape[1],1), device=CURRENT_DEVICE)
    norm_factor = torch_mm(U, torch_mm(V_T, tmp_ones))
    norm_factor = (torch_sum(norm_factor) / U.shape[0]) + 1e-6
    return 1/norm_factor


def weight_init(layer):
    # if type(layer) == Linear:
    if isinstance(layer, Linear):
        xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            constant_(layer.bias.data,0)
    return


class LowRankAttention(Module):
    def __init__(self, k, d, dropout):
        super(LowRankAttention, self).__init__()
        self.w = Sequential(Linear(d, 4*k), ReLU())
        self.activation = ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = Dropout(p=dropout)

    def forward(self, X):
        tmp = self.w(X)
        U = tmp[:, :self.k]
        V = tmp[:, self.k:2*self.k]
        Z = tmp[:, 2*self.k: 3*self.k]
        T = tmp[:, 3*self.k:]
        V_T = torch_t(V)
        # normalization
        D = joint_normalize2(U, V_T)
        res = torch_mm(U, torch_mm(V_T, Z))
        res = torch_cat((res*D,T),dim=1)
        return self.dropout(res)

#
# class GCNWithAttention(Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
#                  dropout, k):
#         super(GCNWithAttention, self).__init__()
#         self.k = k
#         two_k = 2*self.k
#         self.hidden = hidden_channels
#         self.num_layer = num_layers
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.convs = ModuleList()
#         self.convs.append(GCNConv(in_channels, hidden_channels))
#         self.attention = ModuleList()
#         self.dimension_reduce = ModuleList()
#         self.attention.append(LowRankAttention(self.k, in_channels, dropout))
#         self.dimension_reduce.append(Sequential(Linear(self.two_k + hidden_channels,\
#         hidden_channels), ReLU()))
#         self.bn = ModuleList([BatchNorm1d(hidden_channels) for _ in range(num_layers-1)])
#         for _ in range(num_layers - 1):
#             self.convs.append(GCNConv(hidden_channels, hidden_channels))
#             self.attention.append(LowRankAttention(self.k, hidden_channels, dropout))
#             self.dimension_reduce.append(Sequential(Linear(self.two_k + hidden_channels,\
#             hidden_channels)))
#         self.dimension_reduce[-1] = Sequential(Linear(self.two_k + hidden_channels,\
#             out_channels))
#         self.dropout = dropout
#
#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for glob_attention in self.attention:
#             glob_attention.apply(weight_init)
#         for dim_reduce in self.dimension_reduce:
#             dim_reduce.apply(weight_init)
#         for batch_norm in self.bn:
#             batch_norm.reset_parameters()
#
#     def forward(self, x, adj):
#         for i, conv in enumerate(self.convs[:-1]):
#             x_local = F_relu(conv(x, adj))
#             x_local = F_dropout(x_local, p=self.dropout, training=self.training)
#             x_global = self.attention[i](x)
#             x = self.dimension_reduce[i](torch_cat((x_global, x_local),dim=1))
#             x = F_relu(x)
#             x = self.bn[i](x)
#         x_local = F_relu(self.convs[-1](x, adj))
#         x_local = F_dropout(x_local, p=self.dropout, training=self.training)
#         x_global = self.attention[-1](x)
#         return self.dimension_reduce[-1](torch_cat((x_global, x_local),dim=1))
#

#
# class GCN(Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
#                  dropout):
#         super(GCN, self).__init__()
#
#         self.convs = ModuleList()
#         self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
#         for _ in range(num_layers - 2):
#             self.convs.append(
#                 GCNConv(hidden_channels, hidden_channels, cached=True))
#         self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
#
#         self.dropout = dropout
#
#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#
#     def forward(self, x, adj_t):
#         for conv in self.convs[:-1]:
#             x = conv(x, adj_t)
#             x = F_relu(x)
#             x = F_dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
#         return x
