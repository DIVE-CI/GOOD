r"""
Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism <https://arxiv.org/abs/2201.12987>`_.
"""
import copy

import munch
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected
from torch_geometric.data import Data
from torch_sparse import transpose

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor
from munch import munchify


@register.model_register
class GEIGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GEIGIN, self).__init__(config)
        '''
        Don't forget to modify vGIN part if modifying these definitions.
        '''
        self.sub_gnn = GINFeatExtractor(config)
        self.extractor = ExtractorMLP(config)

        self.sp_config = copy.deepcopy(config)
        self.sp_config.model.model_layer = config.model.model_layer - 1
        self.lc_gnn = GINFeatExtractor(config)
        self.la_gnn = GINFeatExtractor(config)
        self.ec_gnn = GINFeatExtractor(config)
        # self.ea_gnn = GINFeatExtractor(self.sp_config, without_embed=True)

        self.lc_classifier = Classifier(config)
        self.la_classifier = Classifier(config)
        self.ec_classifier = Classifier(munchify({'model': {'dim_hidden': config.model.dim_hidden},
                                                   'dataset': {'num_classes': config.dataset.num_envs}}))
        self.ea_classifier = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_ffn),
             nn.BatchNorm1d(2 * config.model.dim_ffn),
             nn.ReLU(),
             nn.Dropout(config.model.dropout_rate),
             nn.Linear(2 * config.model.dim_ffn, 2 * config.model.dim_ffn),
             nn.BatchNorm1d(2 * config.model.dim_ffn),
             nn.ReLU(),
             nn.Dropout(config.model.dropout_rate),
             nn.Linear(2 * config.model.dim_ffn, config.dataset.num_envs)]
        ))
        # Classifier(munchify({'model': {'dim_hidden': config.model.dim_hidden},
        #                                           'dataset': {'num_classes': config.dataset.num_envs}}))
        self.config = config

        self.learn_edge_att = True
        self.LA = config.ood.extra_param[0]
        self.EC = config.ood.extra_param[1]
        self.EA = config.ood.extra_param[2]


    def forward(self, *args, **kwargs):
        r"""
        The GEIGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        node_repr = self.sub_gnn.get_node_repr(*args, **kwargs)
        att_log_logits = self.extractor(node_repr, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, self.training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                nodesize = data.x.shape[0]
                edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)


        set_masks(edge_att, self.lc_gnn)
        lc_repr = self.lc_gnn(*args, **kwargs)
        lc_logits = self.lc_classifier(lc_repr)
        clear_masks(self)

        if self.LA:
            # --- !!! this LA is a relative value according to EA. ---: * LA * EA
            set_masks(1 - GradientReverseLayerF.apply(edge_att, self.LA * self.EA * self.config.train.alpha), self.la_gnn)
            la_logits = self.la_classifier(self.la_gnn(*args, **kwargs))
            clear_masks(self)
        else:
            la_logits = None

        if self.EC:
            set_masks(1 - edge_att, self.ec_gnn)
            ec_logits = self.ec_classifier(self.ec_gnn(*args, **kwargs))
            clear_masks(self)
        else:
            ec_logits = None

        if self.EA:
            EA_alpha = self.EA * self.config.train.alpha
            # set_masks(GradientReverseLayerF.apply(edge_att, EA_alpha), self.ea_gnn)
            ea_logits = self.ea_classifier(GradientReverseLayerF.apply(lc_repr, EA_alpha))
            # clear_masks(self)
        else:
            ea_logits = None



        return (lc_logits, la_logits, ec_logits, ea_logits), att, edge_att

    def sampling(self, att_log_logits, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        # if training:
        if True:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


@register.model_register
class GEIvGIN(GEIGIN):
    r"""
    The GIN virtual node version of GSAT.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GEIvGIN, self).__init__(config)
        self.sub_gnn = vGINFeatExtractor(config)
        self.lc_gnn = vGINFeatExtractor(config)
        self.la_gnn = vGINFeatExtractor(config)
        self.ec_gnn = vGINFeatExtractor(config)
        # self.ea_gnn = vGINFeatExtractor(self.sp_config, without_embed=True)


class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = True
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                # m.append(nn.BatchNorm1d(channels[i], track_running_stats=True))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

class GradientReverseLayerF(Function):
    r"""
    Gradient reverse layer for DANN algorithm.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        r"""
        gradient forward propagation

        Args:
            ctx (object): object of the GradientReverseLayerF class
            x (Tensor): feature representations
            alpha (float): the GRL learning rate

        Returns (Tensor):
            feature representations

        """
        ctx.alpha = alpha
        return x.view_as(x)  # * alpha

    @staticmethod
    def backward(ctx, grad_output):
        r"""
        gradient backpropagation step

        Args:
            ctx (object): object of the GradientReverseLayerF class
            grad_output (Tensor): raw backpropagation gradient

        Returns (Tensor):
            backpropagation gradient

        """
        output = grad_output.neg() * ctx.alpha
        return output, None


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._apply_sigmoid = False
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None
