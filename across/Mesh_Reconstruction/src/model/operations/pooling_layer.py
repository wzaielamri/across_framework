##############################################################################
# This file has been taken from https://github.com/pixelite1201/pytorch_coma #
##############################################################################
import torch
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.cheb_conv import ChebConv
from torch_geometric.utils import remove_self_loops
from torch.nn import Parameter

from across.Mesh_Reconstruction.src.model.operations.mesh_operations import normal

class ChebConv_Coma(ChebConv):
    def __init__(self, in_channels, out_channels, K, normalization=None, bias=True):
        super(ChebConv_Coma, self).__init__(in_channels, out_channels, K, normalization, bias)


    def reset_parameters(self):
        for lin in self.lins:
            normal(lin.weight, 0, 0.1)
        normal(self.bias, 0, 0.1)


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, -deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, norm, edge_weight=None):
        Tx_0 = x
        out = self.lins[0](Tx_0)

        #x = x.transpose(0,1)
        Tx_0 = x
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out



    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class Pool(MessagePassing):
    def __init__(self):
        super(Pool, self).__init__(flow='target_to_source')

    def forward(self, x, pool_mat,  dtype=None):
        #x = x.transpose(0,1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out#.transpose(0,1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


