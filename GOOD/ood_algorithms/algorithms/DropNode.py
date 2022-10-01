
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
class DropNode(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DropNode, self).__init__(config)
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

        if training:
            batch = data.batch
            i_mark = batch - torch.roll(batch, 1)
            i_mark[0] = 1
            idx = torch.empty(data.x.size(0), dtype=torch.float32, device=config.device).uniform_(0, 1)
            # node_mask = ((idx) > config.ood.ood_param)
            node_idx = torch.where(torch.logical_or((idx > config.ood.ood_param), (i_mark == 1)))[0]
            if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                edge_idx_f, edge_attr_f = subgraph(node_idx, data.edge_index, data.edge_attr,
                                                   relabel_nodes=True, num_nodes=data.x.size(0))
                data.edge_index = edge_idx_f
                data.edge_attr = edge_attr_f
            else:
                edge_idx_f = subgraph(node_idx, data.edge_index, relabel_nodes=True, num_nodes=data.x.size(0))[0]
                data.edge_index = edge_idx_f
            data.x = data.x[node_idx]
            data.batch = data.batch[node_idx]

        return data, targets, mask, node_norm

