"""
GCN implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
<https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .GCNs import GCNConv


@register.model_register
class Aug_GCN(GNNBasic):
    r"""
    The Graph Neural Network modified from the `"Mixup for Node and Graph Classification"
    <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper and `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = LISAGCNFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The Mixup-GCN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out


class LISAGCNFeatExtractor(GNNBasic):
    r"""
        GCN feature extractor using the :class:`~GCNEncoder` .

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(LISAGCNFeatExtractor, self).__init__(config)
        self.encoder = LISAGCNEncoder(config)
        self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GCN feature extractor using the :class:`~GCNEncoder` .

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        data = kwargs.get('data')
        ood_algorithm = kwargs.get('ood_algorithm')
        lam = ood_algorithm.lam
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        # for idx in torch.where(data.train_mask)[0]:

        out_readout = self.encoder(x, edge_index, edge_weight, batch)
        return out_readout


class LISAGCNEncoder(BasicEncoder):
    r"""
    The GCN encoder using the :class:`~GCNConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(LISAGCNEncoder, self).__init__(config)
        num_layer = config.model.model_layer

        self.conv1 = GCNConv(config.dataset.dim_node, config.model.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(config.model.dim_hidden, config.model.dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_weight, batch):
        r"""
        The GCN encoder.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_weight (Tensor): edge weights
            batch (Tensor): batch indicator

        Returns (Tensor):
            node feature representations
        """
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_weight))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_weight))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        out_readout = self.readout(post_conv, batch)
        return out_readout