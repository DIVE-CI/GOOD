
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
class Maskfeat(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Maskfeat, self).__init__(config)
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
            idx = torch.empty(data.x.size(1), dtype=torch.float32).uniform_(0, 1)
            feat_idx = torch.where(idx < config.ood.ood_param)[0]
            if config.model.model_level == 'node':
                data.x[:, feat_idx][data.train_mask] = 0
            else:
                data.x[:, feat_idx] = 0

        return data, targets, mask, node_norm

