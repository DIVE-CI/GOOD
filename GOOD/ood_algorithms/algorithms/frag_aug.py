
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
class frag_aug(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(frag_aug, self).__init__(config)
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

        self.lam = 0.5
        batch_size = data.batch[-1] + 1
        if training:  # and config.train.epoch > 40:

            # if training:
            new_batch = []
            org_batch = []
            for i in range(batch_size):

                data_a = data[i]
                size_a = torch.randint(int(0.5 * data_a.x.shape[0]), int(0.8 * data_a.x.shape[0])+1, (1, 1), device=config.device)
                # frag_1 = int(data_a.x.shape[0] / 2)
                # frag_2 = data_a.x.shape[0] - frag_1
                # bi_edge_index = to_undirected(data_a.edge_index)
                edge_mask = (data_a.edge_index[0] < size_a) & (data_a.edge_index[1] < size_a)

                edge_idx = data_a.edge_index[:, edge_mask.squeeze()]
                if config.dataset.dataset_type == 'mol':
                    edge_attr = data_a.edge_attr[edge_mask.squeeze()]

                if config.dataset.dataset_type == 'mol':
                    new_batch.append(Data(x=data_a.x[:size_a], edge_index=edge_idx, edge_attr=edge_attr, y=data_a.y, env_id=data_a.env_id))
                    org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, edge_attr=data_a.edge_attr, y=data_a.y, env_id=data_a.env_id))
                else:
                    new_batch.append(Data(x=data_a.x[:size_a], edge_index=edge_idx, y=data_a.y, env_id=data_a.env_id))
                    org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, y=data_a.y, env_id=data_a.env_id))

            data = Batch.from_data_list(org_batch+new_batch)
            targets = torch.cat((targets, targets))
            mask = torch.cat((mask, mask))
        # else:
        #     self.lam = 1
        #     self.id_a2b = torch.arange(batch_size, device=config.device)

        return data, targets, mask, node_norm


    # def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
    #                    config: Union[CommonArgs, Munch]) -> Tensor:
    #     r"""
    #     Calculate loss based on Mixup algorithm
    #
    #     Args:
    #         raw_pred (Tensor): model predictions
    #         targets (Tensor): input labels
    #         mask (Tensor): NAN masks for data formats
    #         node_norm (Tensor): node weights for normalization (for node prediction only)
    #         config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)
    #
    #     .. code-block:: python
    #
    #         config = munchify({model: {model_level: str('graph')},
    #                                metric: {loss_func()}
    #                                })
    #
    #
    #     Returns (Tensor):
    #         loss based on Mixup algorithm
    #
    #     """
    #     # id_1 = torch.cat((self.id, self.id, self.id_a2b))
    #     # id_2 = torch.cat((self.id, self.id, self.id_a2b_2))
    #     # id_3 = torch.cat((self.id, self.id_a2b, self.id))
    #     # id_4 = torch.cat((self.id, self.id_a2b, self.id_a2b))
    #     # id_5 = torch.cat((self.id, self.id_a2b, self.id_a2b_2))
    #     loss_a = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
    #     loss_b = config.metric.loss_func(raw_pred, targets[self.id_a2b], reduction='none') * mask
    #     # loss_b = config.metric.loss_func(raw_pred, targets[id_1], reduction='none') * mask
    #     # loss_2 = config.metric.loss_func(raw_pred, targets[id_2], reduction='none') * mask
    #     # loss_3 = config.metric.loss_func(raw_pred, targets[id_3], reduction='none') * mask
    #     # loss_4 = config.metric.loss_func(raw_pred, targets[id_4], reduction='none') * mask
    #     # loss_5 = config.metric.loss_func(raw_pred, targets[id_5], reduction='none') * mask
    #     if config.model.model_level == 'node':
    #         loss_a = loss_a * node_norm * mask.sum()
    #         loss_b = loss_b * node_norm[self.id_a2b] * mask.sum()
    #     loss = self.lam * loss_a + (1 - self.lam) * loss_b
    #     # loss = (loss_5 + loss_4 + loss_3 + loss_2 + loss_b + loss_a)/6
    #     return loss


    # def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
    #     r"""
    #     Process loss based on VREx algorithm
    #
    #     Args:
    #         loss (Tensor): base loss between model predictions and input labels
    #         data (Batch): input data
    #         mask (Tensor): NAN masks for data formats
    #         config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    #
    #     .. code-block:: python
    #
    #         config = munchify({device: torch.device('cuda'),
    #                                dataset: {num_envs: int(10)},
    #                                ood: {ood_param: float(0.1)}
    #                                })
    #
    #
    #     Returns (Tensor):
    #         loss based on VREx algorithm
    #
    #     """
    #     loss_list = []
    #     for i in range(config.dataset.num_envs):
    #         env_idx = data.env_id == i
    #         if loss[env_idx].shape[0] > 0:
    #             loss_list.append(loss[env_idx].sum() / mask[env_idx].sum())
    #     spec_loss = config.ood.ood_param * torch.var(torch.tensor(loss_list, device=config.device))
    #     if torch.isnan(spec_loss):
    #         spec_loss = 0
    #     mean_loss = loss.sum() / mask.sum()
    #     loss = spec_loss + mean_loss
    #     self.mean_loss = mean_loss
    #     self.spec_loss = spec_loss
    #     return loss