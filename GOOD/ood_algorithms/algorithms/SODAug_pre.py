
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
class SODAug_pre(BaseOODAlg):
    r"""
    Implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
    <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.model.model_level`, :obj:`config.metric.loss_func()`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SODAug_pre, self).__init__(config)
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

        # self.lam = 0.5
        # batch_size = data.batch[-1] + 1
        # # if training:
        # new_batch = []
        # for i in range(batch_size):
        #
        #     data_a = data[i]
        #     frag_1 = int(data_a.x.shape[0] / 2)
        #     frag_2 = data_a.x.shape[0] - frag_1
        #     # bi_edge_index = to_undirected(data_a.edge_index)
        #     edge_mask = ((data_a.edge_index[0] < frag_1) & (data_a.edge_index[1] >= frag_1)) | ((data_a.edge_index[0] >= frag_1) & (data_a.edge_index[1] < frag_1))
        #
        #     edge_idx = data_a.edge_index[:, ~edge_mask]
        #     if config.dataset.dataset_type == 'mol':
        #         edge_attr = data_a.edge_attr[~edge_mask]
        #
        #     edge_mask = (data_a.edge_index[0] < frag_1) & (data_a.edge_index[1] >= frag_1)
        #     y = torch.zeros((frag_1, frag_2), device=config.device)
        #     m = data_a.edge_index[:, edge_mask]
        #     y[m[0], m[1]-frag_1] = 1
        #
        #     if config.dataset.dataset_type == 'mol':
        #         new_batch.append(
        #             Data(x=data_a.x, edge_index=edge_idx, edge_attr=edge_attr, bridge=y.view(-1), bridge_num=torch.tensor(m.shape[1]).to(config.device), frag_1=frag_1, frag_2=frag_2, y=data_a.y))
        #         # org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, edge_attr=data_a.edge_attr, y=data_a.y))
        #     else:
        #         new_batch.append(Data(x=data_a.x, edge_index=edge_idx, bridge=y.view(-1), bridge_num=torch.tensor(m.shape[1]).to(config.device), frag_1=frag_1, frag_2=frag_2, y=data_a.y))
        #         # new_batch_2.append(Data(x=x_2, edge_index=edge_idx_2, y=data_a.y))
        #         # org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, y=data_a.y))
        #
        # data = Batch.from_data_list(new_batch)
        # data = Batch.from_data_list(new_batch)
        # targets = torch.cat((targets, targets, targets))
        # mask_mix = mask & mask[self.id_a2b]
        # mask_mix_2 = mask & mask[self.id_a2b] & mask[self.id_a2b_2]
        # # mask = mask_mix
        # mask = torch.cat((mask, mask_mix, mask_mix_2))
        # self.id = torch.arange(batch_size)
        # self.id_a2b = torch.cat((torch.arange(batch_size), self.id_a2b))
        # self.id_a2b = torch.arange(data.num_nodes, device=config.device)

        if training:
            mask = mask.chunk(2)[0] & mask.chunk(2)[1]
            targets = targets.chunk(2)[0]

        return data, targets, mask, node_norm


    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; get domain classifier predictions

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        self.bridge = model_output[1]
        self.kl_divergence = model_output[2]
        self.edge_attr_loss = model_output[3]
        self.num_loss = model_output[5]
        self.pl_pred = model_output[7]

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

        loss = binary_cross_entropy_with_logits(raw_pred.view(-1), self.bridge.view(-1), reduction='none')

        pl_loss = config.metric.loss_func(self.pl_pred, targets, reduction='none') * mask
        pl_loss = pl_loss * node_norm * mask.sum() if config.model.model_level == 'node' else pl_loss
        self.pl_loss = pl_loss.sum() / mask.sum()
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

        return loss


    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:

        loss = loss.sum()/loss.shape[0]
        self.mean_loss = loss
        self.spec_loss = 0 - config.ood.extra_param[0] * self.kl_divergence + config.ood.extra_param[1]*self.num_loss
        if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
            self.spec_loss = self.spec_loss + config.ood.extra_param[2]*self.edge_attr_loss
        self.spec_loss = self.spec_loss + self.pl_loss
        loss = self.mean_loss + self.spec_loss
        return loss