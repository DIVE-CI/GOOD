
import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from torch_geometric.utils import subgraph


@register.ood_alg_register
class SODAugFeat(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SODAugFeat, self).__init__(config)
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
        uniq_feats = data.x.unique(dim=0)
        if training:  # and config.train.epoch > 20:
            new_batch = []
            org_batch = []
            frag_batch = []
            recom_batch = []
            env_list = []
            env_recom_list = []
            env_new = copy.deepcopy(data[0].env_id) - data[0].env_id + config.dataset.num_envs
            env_frag = copy.deepcopy(env_new) + 1
            # config.dataset.num_envs = config.dataset.num_envs + 2
            # if config.dataset.dataloader_name == 'PairDataLoader':
            #     self.id_a2b = torch.arange(int(batch_size/2), batch_size)
            # else:
            #     self.id_a2b = torch.randperm(batch_size)
            frag_out = gen_model.frag_sample(data=data)
            batch = data.batch
            self.id_a2b = torch.randperm(int(batch_size/2))
            for idx_a, idx_b in enumerate(self.id_a2b):
                data_a = data[idx_a + int(batch_size / 2)]
                data_b = data[idx_b]
                if idx_a+1 > config.ood.extra_param[10]*int(batch_size/2):
                    if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                        org_batch.append(
                            Data(x=feataug(data_a.x, uniq_feats), edge_index=data_a.edge_index, edge_attr=data_a.edge_attr, y=data_a.y,
                                 env_id=data_a.env_id))
                    else:
                        org_batch.append(
                            Data(x=feataug(data_a.x, uniq_feats), edge_index=data_a.edge_index, y=data_a.y, env_id=data_a.env_id))
                    continue

                out_a = frag_out[batch == (idx_a)]
                out_b = frag_out[batch == (idx_a+int(batch_size/2))]
                if out_a.shape[0] > out_b.shape[0]:
                    out_a, out_b = out_b, out_a
                    data_s = data[idx_a + int(batch_size / 2)]
                    data_l = data[idx_a]
                else:
                    data_l = data[idx_a + int(batch_size / 2)]
                    data_s = data[idx_a]

                cos_sim = gen_model.PLNN.sim(out_a, out_b)
                max_a, pick_b = cos_sim.max(1)
                size_a = torch.randint(max(int(0.5 * cos_sim.shape[0]), 1), max(int(0.8 * cos_sim.shape[0]), 2),
                                       ()).item()
                k = min(size_a, config.ood.extra_param[3])
                pick_a = torch.topk(max_a, k, dim=0).indices
                pick_b = pick_b[pick_a]
                x_f = data_s.x[pick_a]
                if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                    edge_idx_f, edge_attr_f = subgraph(pick_a, data_s.edge_index, data_s.edge_attr, relabel_nodes=True, num_nodes=data_s.x.shape[0])
                else:
                    edge_idx_f = subgraph(pick_a, data_s.edge_index, relabel_nodes=True, num_nodes=data_s.x.shape[0])[0]

                if config.ood.extra_param[9] > 0.5:
                    combined = torch.cat((torch.arange(0, data_l.x.shape[0]).to(config.device), pick_b))
                    uniques, counts = combined.unique(return_counts=True)
                    pick_b_spu = uniques[counts == 1]
                    # pick_b_spu = pick_b[torch.topk(max_a, k, dim=0, largest=False).indices]
                    x_spu = data_l.x[pick_b_spu]
                    if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                        edge_idx_spu, edge_attr_spu = subgraph(pick_b_spu, data_l.edge_index, data_l.edge_attr,
                                                               relabel_nodes=True, num_nodes=data_l.x.shape[0])
                    else:
                        edge_idx_spu = subgraph(pick_b_spu, data_l.edge_index, relabel_nodes=True, num_nodes=data_l.x.shape[0])[0]

                    x_recom = torch.cat((x_f, x_spu), dim=0)

                    if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                        edge_attr_recom = torch.cat((edge_attr_f, edge_attr_spu), dim=0)

                    edge_idx_recom = torch.cat((edge_idx_f, edge_idx_spu + x_f.shape[0]), dim=1)

                x = torch.cat((data_a.x, data_b.x), dim=0)

                if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                    edge_attr = torch.cat((data_a.edge_attr, data_b.edge_attr), dim=0)

                edge_idx = torch.cat((data_a.edge_index, data_b.edge_index + data_a.x.shape[0]), dim=1)
                y = 0
                env_list.append(env_new)
                env_recom_list.append(data_l.env_id)

                if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                    new_batch.append(Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, bridge=y, bridge_num=torch.tensor(0).to(config.device), frag_1=data_a.x.shape[0], frag_2=data_b.x.shape[0], y=data_a.y))
                    org_batch.append(Data(x=feataug(data_a.x, uniq_feats), edge_index=data_a.edge_index, edge_attr=data_a.edge_attr, y=data_a.y, env_id=data_a.env_id))
                    frag_batch.append(Data(x=feataug(x_f, uniq_feats), edge_index=edge_idx_f, edge_attr=edge_attr_f, y=data_a.y,env_id=env_frag))
                    if config.ood.extra_param[9] > 0.5:
                        recom_batch.append(
                            Data(x=x_recom, edge_index=edge_idx_recom, edge_attr=edge_attr_recom, bridge=y,
                                 bridge_num=torch.tensor(0).to(config.device), frag_1=x_f.shape[0],
                                 frag_2=x_spu.shape[0], y=data_s.y))
                else:
                    new_batch.append(Data(x=x, edge_index=edge_idx, bridge=y, bridge_num=torch.tensor(0).to(config.device), frag_1=data_a.x.shape[0], frag_2=data_b.x.shape[0], y=data_a.y))
                    org_batch.append(Data(x=feataug(data_a.x, uniq_feats), edge_index=data_a.edge_index, y=data_a.y, env_id=data_a.env_id))
                    frag_batch.append(Data(x=feataug(x_f, uniq_feats), edge_index=edge_idx_f, y=data_a.y, env_id=env_frag))
                    if config.ood.extra_param[9] > 0.5:
                        recom_batch.append(Data(x=x_recom, edge_index=edge_idx_recom, bridge=y,
                                                bridge_num=torch.tensor(0).to(config.device), frag_1=x_f.shape[0],
                                                frag_2=x_spu.shape[0], y=data_s.y))

            if config.ood.extra_param[9] > 0.5:
                new_data = Batch.from_data_list(new_batch+recom_batch)
                env_list = torch.cat((torch.cat(env_list[:]).squeeze(), torch.cat(env_recom_list[:]).squeeze()))
            else:
                new_data = Batch.from_data_list(new_batch)
                env_list = torch.cat(env_list[:]).squeeze()
            gen_model.eval()
            model_output = gen_model.bridge_sample(data=new_data, edge_weight=None, ood_algorithm=config.ood.ood_alg)
            bridge_score = model_output[0].detach()
            num_bridge_all = torch.round(model_output[6]).long()

            batch_size = new_data.batch[-1] + 1
            final_batch = []

            s = 0
            for idx_ in range(batch_size):
                data_a = new_data[idx_]
                num_bridge = num_bridge_all[idx_].item()
                if num_bridge>0 and num_bridge<=int(data_a.frag_1*data_a.frag_2):
                    score = bridge_score[s:s + data_a.frag_1 * data_a.frag_2]
                    v, indices = torch.topk(score.squeeze(), num_bridge)
                    indices = indices.cpu()
                    bridge_a = (indices / data_a.frag_2).long().to(config.device)
                    bridge_b = (indices % data_a.frag_2 + data_a.frag_1).to(config.device)
                    bridge = torch.cat((bridge_a.unsqueeze(0), bridge_b.unsqueeze(0)), dim=0)
                    undirected_bridge = torch.cat((bridge_b.unsqueeze(0), bridge_a.unsqueeze(0)), dim=0)
                    edge_idx = torch.cat((data_a.edge_index, bridge, undirected_bridge), dim=1)
                else:
                    edge_idx = data_a.edge_index

                if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                    if num_bridge > 0:
                        bridge_attr_pred = model_output[4].detach()
                        # bridge_attr_idx = torch.randint(0, data_a.edge_attr.shape[0], (1, num_bridge), device=config.device)
                        # bridge_attr = torch.squeeze(copy.deepcopy(data_a.edge_attr[bridge_attr_idx]), dim=0)
                        bridge_attr = bridge_attr_pred[s:s + data_a.frag_1 * data_a.frag_2][indices]
                        # bridge_attr_idx = torch.cat((torch.norm(data.edge_attr - bridge_attr[0], dim=1).argmin().unsqueeze(0), torch.norm(data.edge_attr - bridge_attr[1], dim=1).argmin().unsqueeze(0)))
                        for co in range(bridge_attr.shape[0]):
                            temp = torch.norm(data.edge_attr - bridge_attr[co], dim=1).argmin().unsqueeze(0)
                            if co == 0:
                                bridge_attr_idx = temp
                            else:
                                bridge_attr_idx = torch.cat((bridge_attr_idx, temp))
                        bridge_attr = data.edge_attr[bridge_attr_idx]
                        edge_attr = torch.cat((data_a.edge_attr, bridge_attr.type(data_a.edge_attr.dtype),
                                               bridge_attr.type(data_a.edge_attr.dtype)), dim=0)
                    else:
                        edge_attr = data_a.edge_attr

                s = s + data_a.frag_1 * data_a.frag_2

                if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                    final_batch.append(Data(x=feataug(data_a.x, uniq_feats), edge_index=edge_idx, edge_attr=edge_attr, y=data_a.y, env_id=env_list[idx_][None]))
                else:
                    final_batch.append(Data(x=feataug(data_a.x, uniq_feats), edge_index=edge_idx, y=data_a.y, env_id=env_list[idx_][None]))

            if config.ood.extra_param[9] > 0.5:
                batch_size = int(batch_size / 2)
                if config.ood.extra_param[8] > 0.5:
                    data = Batch.from_data_list(org_batch + final_batch)
                    self.id_a2b = torch.cat(
                        (torch.arange(mask.chunk(2)[0].shape[0]), self.id_a2b[:batch_size], torch.arange(batch_size)))
                    targets = torch.cat((targets.chunk(2)[0], targets[:batch_size], targets[:batch_size]))
                    mask_mix = mask.chunk(2)[1][:batch_size] & mask[self.id_a2b][:batch_size]
                    mask_recom = mask.chunk(2)[0][:batch_size] & mask.chunk(2)[1][:batch_size]
                    mask = torch.cat((mask.chunk(2)[0], mask_mix, mask_recom))
                    # self.id = torch.arange(batch_size)
                else:
                    data = Batch.from_data_list(org_batch + final_batch + frag_batch)
                    self.id_a2b = torch.cat(
                        (torch.arange(mask.chunk(2)[0].shape[0]), self.id_a2b[:batch_size], torch.arange(batch_size),
                         torch.arange(batch_size)))
                    targets = torch.cat(
                        (targets.chunk(2)[0], targets[:batch_size], targets[:batch_size], targets[:batch_size]))
                    mask_mix = mask.chunk(2)[1][:batch_size] & mask[self.id_a2b][:batch_size]
                    mask_recom = mask.chunk(2)[0][:batch_size] & mask.chunk(2)[1][:batch_size]
                    mask_frag = mask_recom
                    mask = torch.cat((mask.chunk(2)[0], mask_mix, mask_recom, mask_frag))
            else:
                if config.ood.extra_param[8] > 0.5:
                    data = Batch.from_data_list(org_batch + final_batch)
                    self.id_a2b = torch.cat(
                        (torch.arange(mask.chunk(2)[0].shape[0]), self.id_a2b[:batch_size]))
                    targets = torch.cat((targets.chunk(2)[0], targets[:batch_size]))
                    mask_mix = mask.chunk(2)[1][:batch_size] & mask[self.id_a2b][:batch_size]
                    # mask_recom = mask.chunk(2)[0][:batch_size] & mask.chunk(2)[1][:batch_size]
                    mask = torch.cat((mask.chunk(2)[0], mask_mix))
                    # self.id = torch.arange(batch_size)
                else:
                    data = Batch.from_data_list(org_batch + final_batch + frag_batch)
                    self.id_a2b = torch.cat(
                        (torch.arange(mask.chunk(2)[0].shape[0]), self.id_a2b[:batch_size],
                         torch.arange(batch_size)))
                    targets = torch.cat(
                        (targets.chunk(2)[0], targets[:batch_size], targets[:batch_size]))
                    mask_mix = mask.chunk(2)[1][:batch_size] & mask[self.id_a2b][:batch_size]
                    mask_recom = mask.chunk(2)[0][:batch_size] & mask.chunk(2)[1][:batch_size]
                    mask_frag = mask_recom
                    mask = torch.cat((mask.chunk(2)[0], mask_mix, mask_frag))



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


    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on VREx algorithm

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   dataset: {num_envs: int(10)},
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns (Tensor):
            loss based on VREx algorithm

        """
        loss_list = []
        spec_loss = torch.tensor(0)
        if config.ood.extra_param[5] > 0:
            for i in range(config.dataset.num_envs+2):
                env_idx = data.env_id == i
                if loss[env_idx].shape[0] > 0:
                    loss_list.append(loss[env_idx].sum() / mask[env_idx].sum())
            spec_loss = config.ood.extra_param[5] * torch.var(torch.tensor(loss_list, device=config.device))
        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = loss.sum() / mask.sum()
        loss = spec_loss + mean_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss
        return loss


def feataug(feat, uniq_feats):
    chance = torch.randint(2, (1,)).item()
    if chance == 0:
        aug_feat = feat
    else:
        uniq = torch.randint(uniq_feats.shape[0], (1,)).item()
        aug = torch.randint(3, (1,)).item()
        if aug == 0:
            aug_feat = (feat + uniq_feats[uniq]) / 2
        elif aug == 1:
            aug_feat = 2 * feat - uniq_feats[uniq]
        else:
            aug_feat = 2 * uniq_feats[uniq] - feat
    return aug_feat
