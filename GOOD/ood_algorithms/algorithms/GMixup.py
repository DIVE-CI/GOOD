"""
Implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
<https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper
"""
import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg




@register.ood_alg_register
class GMixup(BaseOODAlg):
    r"""
    Implementation of the G-Mixup algorithm from `"G-Mixup"
    <XXXX>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.model.model_level`, :obj:`config.metric.loss_func()`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GMixup, self).__init__(config)
        self.lam = None

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set input data format and preparations

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        if training:
            targets = torch.stack([data.y1, data.y2], dim=0)
            self.lam = data.lam
        return data, targets, mask, node_norm

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss based on G-Mixup algorithm

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
        loss_a = config.metric.loss_func(raw_pred, targets[0], reduction='none') * mask
        loss_b = config.metric.loss_func(raw_pred, targets[1], reduction='none') * mask
        if config.model.model_level == 'node':
            raise Exception("G-Mixup only serves for graph level tasks.")
        loss = self.lam * loss_a + (1 - self.lam) * loss_b
        return loss
