
import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg



@register.ood_alg_register
class LISA(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(LISA, self).__init__(config)
        self.lam = None
        self.data_perm = None
        self.id_a2b: Tensor
        self.id: Tensor
        self.id_a2b_2: Tensor

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:

        if training and config.model.model_level != 'node':
            targets = targets.float()
            alpha = config.ood.ood_param  # 2,4
            self.lam = np.random.beta(alpha, alpha)
            batch_size = int((data.batch[-1] + 1)/2)
            # targets = targets[:batch_size]
            mask = mask[:batch_size] & mask[batch_size:]
            mask = torch.cat((mask, mask))
            self.id_a2b = torch.cat((torch.arange(batch_size)+batch_size, torch.arange(batch_size)))
        elif training:
            alpha = config.ood.ood_param
            num_class = torch.unique(data.y).shape[0]
            self.lam = np.random.beta(alpha, alpha)
            for i in range(num_class):
                label_idx = torch.logical_and((data.y == i).squeeze(), data.train_mask).clone().detach()
                if torch.unique(data.env_id[label_idx]).shape[0] < 2:
                    continue
                else:
                    in_num = label_idx.nonzero().shape[0]
                    for j in range(in_num):
                        idx_a = label_idx.nonzero()[j]
                        valid_b = torch.logical_and(data.env_id != data.env_id[idx_a], label_idx != 0).nonzero().view(
                            -1)
                        choice = torch.multinomial(valid_b.float(), 1)
                        idx_b = valid_b[choice]
                        data.x[idx_a] = self.lam * data.x[idx_a] + (1-self.lam) * data.x[idx_b]

        else:
            self.lam = 1
            self.id_a2b = torch.arange(data.num_nodes, device=config.device)

        return data, targets, mask, node_norm


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

        # loss_a = config.metric.loss_func(raw_pred.chunk(2)[0], targets, reduction='none') * mask
        # loss_b = config.metric.loss_func(raw_pred.chunk(2)[1], targets, reduction='none') * mask
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        if config.model.model_level == 'node':
            loss = loss * node_norm * mask.sum()
            # loss_a = loss_a * node_norm.chunk(2)[0] * mask.sum()
            # loss_b = loss_b * node_norm.chunk(2)[1] * mask.sum()
        # loss = self.lam * loss_a + (1 - self.lam) * loss_b
        return loss
