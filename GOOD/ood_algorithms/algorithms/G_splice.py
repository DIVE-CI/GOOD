
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
class G_splice(BaseOODAlg):
    r"""
    Implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
    <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.model.model_level`, :obj:`config.metric.loss_func()`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(G_splice, self).__init__(config)
        self.lam = None
        self.data_perm = None
        self.id_a2b: Tensor
        self.id: Tensor
        self.id_a2b_2: Tensor

    def input_generation(self,
                         gen_model: torch.nn.Module,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:

        self.lam = 0.5
        num_bridge = 2
        if config.dataset.dataset_name == 'GOODSST2':
            num_bridge = 1
        batch_size = data.batch[-1] + 1
        if training and config.train.epoch > 20:
            new_batch = []
            # new_batch_2 = []
            org_batch = []
            if config.dataset.dataloader_name == 'PairDataLoader':
                self.id_a2b = torch.arange(int(batch_size/2), batch_size)
            else:
                self.id_a2b = torch.randperm(batch_size)
            # self.id_a2b_2 = torch.randperm(batch_size)
            for idx_a, idx_b in enumerate(self.id_a2b):
                data_a = data[idx_a]
                data_b = data[idx_b]

                x = torch.cat((data_a.x, data_b.x), dim=0)

                # num_bridge = 1
                if config.dataset.dataset_type == 'mol':
                    # bridge_attr_idx = torch.randint(0, data_a.edge_attr.shape[0], (1, num_bridge), device=config.device)
                    # bridge_attr = torch.squeeze(copy.deepcopy(data_a.edge_attr[bridge_attr_idx]), dim=0)
                    # bridge_attr = torch.zeros((num_bridge, data_a.edge_attr.shape[1]), device=config.device).long()
                    edge_attr = torch.cat((data_a.edge_attr, data_b.edge_attr), dim=0)
                # bridge_a = torch.randint(0, data_a.x.shape[0], (1, num_bridge), device=config.device)
                # bridge_b = torch.randint(0, data_b.x.shape[0], (1, num_bridge), device=config.device) + data_a.x.shape[0]
                #
                # bridge = torch.cat((bridge_a, bridge_b), dim=0)

                edge_idx = torch.cat((data_a.edge_index, data_b.edge_index + data_a.x.shape[0]), dim=1)
                y = 0

                if config.dataset.dataset_type == 'mol':
                    new_batch.append(Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, bridge=y, frag_1=data_a.x.shape[0], frag_2=data_b.x.shape[0], y=data_a.y))
                    org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, edge_attr=data_a.edge_attr, y=data_a.y, env_id=data_a.env_id))
                else:
                    new_batch.append(Data(x=x, edge_index=edge_idx, bridge=y, frag_1=data_a.x.shape[0], frag_2=data_b.x.shape[0], y=data_a.y))
                    org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, y=data_a.y, env_id=data_a.env_id))

            new_data = Batch.from_data_list(new_batch)
            model_output = gen_model(data=new_data, edge_weight=None, ood_algorithm=config.ood.ood_alg)
            bridge_score = model_output[0].detach()


            batch_size = new_data.batch[-1] + 1
            final_batch = []
            # config.dataset.num_envs = config.dataset.num_envs + 1
            env_new = copy.deepcopy(data[0].env_id) - data[0].env_id + config.dataset.num_envs
            s = 0
            for idx_ in range(batch_size):
                data_a = new_data[idx_]
                score = bridge_score[s:s+data_a.frag_1 * data_a.frag_2]
                v, indices = torch.topk(score.squeeze(), num_bridge)
                indices = indices.cpu()
                bridge_a = (indices/data_a.frag_2).long().to(config.device)
                bridge_b = (indices % data_a.frag_2 + data_a.frag_1).to(config.device)
                bridge = torch.cat((bridge_a.unsqueeze(0), bridge_b.unsqueeze(0)), dim=0)
                undirected_bridge = torch.cat((bridge_b.unsqueeze(0), bridge_a.unsqueeze(0)), dim=0)
                edge_idx = torch.cat((data_a.edge_index, bridge, undirected_bridge), dim=1)

                if config.dataset.dataset_type == 'mol':
                    bridge_attr_pred = model_output[4].detach()
                    # bridge_attr_idx = torch.randint(0, data_a.edge_attr.shape[0], (1, num_bridge), device=config.device)
                    # bridge_attr = torch.squeeze(copy.deepcopy(data_a.edge_attr[bridge_attr_idx]), dim=0)
                    bridge_attr = bridge_attr_pred[s:s+data_a.frag_1 * data_a.frag_2][indices]
                    bridge_attr_idx = torch.cat((torch.norm(data.edge_attr - bridge_attr[0], dim=1).argmin().unsqueeze(0), torch.norm(data.edge_attr - bridge_attr[1], dim=1).argmin().unsqueeze(0)))
                    bridge_attr = data.edge_attr[bridge_attr_idx]
                    edge_attr = torch.cat((data_a.edge_attr, bridge_attr.type(data_a.edge_attr.dtype), bridge_attr.type(data_a.edge_attr.dtype)), dim=0)

                s = s + data_a.frag_1 * data_a.frag_2

                if config.dataset.dataset_type == 'mol':
                    final_batch.append(Data(x=data_a.x, edge_index=edge_idx, edge_attr=edge_attr, y=data_a.y, env_id=env_new))
                else:
                    final_batch.append(Data(x=data_a.x, edge_index=edge_idx, y=data_a.y, env_id=env_new))


            data = Batch.from_data_list(org_batch + final_batch)
            # data = Batch.from_data_list(new_batch)
            targets = torch.cat((targets[:batch_size], targets[:batch_size]))
            mask_mix = mask[:batch_size] & mask[self.id_a2b]

            # mask = mask_mix
            mask = torch.cat((mask[:batch_size], mask_mix))
            # self.id = torch.arange(batch_size)
            self.id_a2b = torch.cat((torch.arange(batch_size), self.id_a2b))
        else:
            self.lam = 1
            self.id_a2b = torch.arange(batch_size, device=config.device)

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
        # id_1 = torch.cat((self.id, self.id, self.id_a2b))
        # id_2 = torch.cat((self.id, self.id, self.id_a2b_2))
        # id_3 = torch.cat((self.id, self.id_a2b, self.id))
        # id_4 = torch.cat((self.id, self.id_a2b, self.id_a2b))
        # id_5 = torch.cat((self.id, self.id_a2b, self.id_a2b_2))
        loss_a = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss_b = config.metric.loss_func(raw_pred, targets[self.id_a2b], reduction='none') * mask
        # loss_b = config.metric.loss_func(raw_pred, targets[id_1], reduction='none') * mask
        # loss_2 = config.metric.loss_func(raw_pred, targets[id_2], reduction='none') * mask
        # loss_3 = config.metric.loss_func(raw_pred, targets[id_3], reduction='none') * mask
        # loss_4 = config.metric.loss_func(raw_pred, targets[id_4], reduction='none') * mask
        # loss_5 = config.metric.loss_func(raw_pred, targets[id_5], reduction='none') * mask
        if config.model.model_level == 'node':
            loss_a = loss_a * node_norm * mask.sum()
            loss_b = loss_b * node_norm[self.id_a2b] * mask.sum()
        loss = self.lam * loss_a + (1 - self.lam) * loss_b
        # loss = (loss_5 + loss_4 + loss_3 + loss_2 + loss_b + loss_a)/6
        return loss


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