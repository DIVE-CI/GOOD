
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
class DropEdge(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DropEdge, self).__init__(config)
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
            idx = torch.empty(data.edge_index.size(1), dtype=torch.float32).uniform_(0, 1)
            edge_mask = ((idx) >= config.ood.ood_param)
            data.edge_index = data.edge_index[:, edge_mask]
            if hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None:
                data.edge_attr = data.edge_attr[edge_mask]
            if hasattr(data, 'edge_norm') and getattr(data, 'edge_norm') is not None:
                # if data.edge_norm is not None:
                data.edge_norm = data.edge_norm[edge_mask]


        return data, targets, mask, node_norm
