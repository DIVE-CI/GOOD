
import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from torch_geometric.utils import to_undirected
from GOOD.utils.metric import binary_cross_entropy_with_logits

@register.ood_alg_register
class G_splice_gen_diff(BaseOODAlg):
    r"""
    Implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
    <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.model.model_level`, :obj:`config.metric.loss_func()`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(G_splice_gen_diff, self).__init__(config)
        self.lam = None
        self.data_perm = None
        self.id_a2b: Tensor
        self.id: Tensor
        self.id_a2b_2: Tensor
        self.bridge: Tensor

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:

        # batch_size = data.batch[-1] + 1

        return data, targets, mask, node_norm


    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; get domain classifier predictions

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        self.attr_loss = model_output[1]
        self.num_loss = model_output[2]
        # self.kl_divergence = model_output[2]
        # self.edge_attr_loss = model_output[3]

        return model_output[0]


    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss based on Mixup algorithm

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func()}
                                   })


        Returns (Tensor):
            loss based on Mixup algorithm

        """

        # loss = binary_cross_entropy_with_logits(raw_pred.view(-1), self.bridge.view(-1), reduction='none')
        # kl_divergence = 0.5 / A_pred.size(0) * (
        #             1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()


        # loss_b = config.metric.loss_func(raw_pred, targets[self.id_a2b], reduction='none') * mask
        # loss_b = config.metric.loss_func(raw_pred, targets[id_1], reduction='none') * mask
        # loss_2 = config.metric.loss_func(raw_pred, targets[id_2], reduction='none') * mask
        # loss_3 = config.metric.loss_func(raw_pred, targets[id_3], reduction='none') * mask
        # loss_4 = config.metric.loss_func(raw_pred, targets[id_4], reduction='none') * mask
        # loss_5 = config.metric.loss_func(raw_pred, targets[id_5], reduction='none') * mask
        # if config.model.model_level == 'node':
        #     loss_a = loss_a * node_norm * mask.sum()
        #     loss_b = loss_b * node_norm[self.id_a2b] * mask.sum()
        # loss = self.lam * loss_a + (1 - self.lam) * loss_b
        # loss = (loss_5 + loss_4 + loss_3 + loss_2 + loss_b + loss_a)/6

        return raw_pred


    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:

        # loss = loss.sum()/loss.shape[0]
        self.mean_loss = loss
        self.spec_loss = self.num_loss
        if config.dataset.dataset_type == 'mol':
            self.spec_loss = self.spec_loss + self.attr_loss
        # if config.dataset.dataset_type == 'mol':
        #     self.spec_loss = self.spec_loss + self.edge_attr_loss/50
            loss = self.mean_loss + self.spec_loss
        else:
            loss = self.mean_loss
        return loss
