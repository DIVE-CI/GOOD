
import copy
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from transformers import BertTokenizer


@register.ood_alg_register
class Momu(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Momu, self).__init__(config)
        self.lam = None
        self.data_perm = None
        self.id_a2b: Tensor
        self.id: Tensor
        self.id_a2b_2: Tensor
        self.tokenizer = BertTokenizer.from_pretrained('/data/xinerli/MoMu/Pretrain/bert_pretrained/')

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:

        batch_size = data.batch[-1] + 1
        def tokenizer_func(text):
            sentence_token = self.tokenizer(text=text,
                                            truncation=True,
                                            padding='max_length',
                                            add_special_tokens=False,
                                            max_length=batch_size,
                                            return_tensors='pt',
                                            return_attention_mask=True)
            input_ids = sentence_token['input_ids']  # [176,398,1007,0,0,0]
            attention_mask = sentence_token['attention_mask']  # [1,1,1,0,0,0]
            return input_ids, attention_mask

        final_batch = []
        for idx in range(batch_size):
            smiles_tokens, smiles_mask = tokenizer_func(data[idx].smiles)
            if training:
                final_batch.append(Data(x=data[idx].x, edge_index=data[idx].edge_index, edge_attr=data[idx].edge_attr, y=data[idx].y, env_id=data[idx].env_id, smiles=data[idx].smiles, smiles_tokens=smiles_tokens.to(config.device), smiles_mask=smiles_mask.to(config.device)))
            else:
                final_batch.append(
                    Data(x=data[idx].x, edge_index=data[idx].edge_index, edge_attr=data[idx].edge_attr, smiles=data[idx].smiles,
                         smiles_tokens=smiles_tokens.to(config.device), smiles_mask=smiles_mask.to(config.device)))

        data = Batch.from_data_list(final_batch)
        return data, targets, mask, node_norm

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; get domain classifier predictions

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        self.text_pred = model_output[1]
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

        # text_loss: torch.Tensor = config.metric.cross_entropy_with_logit(self.text_pred, data.env_id, reduction='none')
        # else:
        # dc_loss: torch.Tensor = binary_cross_entropy_with_logits(dc_pred, torch.nn.functional.one_hot(data.env_id % config.dataset.num_envs, num_classes=config.dataset.num_envs).float(), reduction='none') * mask
        mean_loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss = mean_loss
        # self.mean_loss = mean_loss
        if config.ood.ood_param > 0:
            spec_loss = config.ood.ood_param * config.metric.loss_func(self.text_pred, targets,
                                                                       reduction='none') * mask  # text_loss
            self.spec_loss = spec_loss
        return loss

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns (Tensor):
            processed loss

        """
        if config.ood.extra_param[0] > 0:
            loss_list = []
            text_loss_list = []
            for i in range(config.dataset.num_envs):
                env_idx = data.env_id == i
                if loss[env_idx].shape[0] > 0:
                    loss_list.append(loss[env_idx].sum() / mask[env_idx].sum())
                if config.ood.ood_param > 0:
                    if self.spec_loss[env_idx].shape[0] > 0:
                        text_loss_list.append(self.spec_loss[env_idx].sum() / mask[env_idx].sum())
            graph_spec_loss = config.ood.extra_param[0] * torch.var(torch.stack(loss_list))
            if torch.isnan(graph_spec_loss):
                graph_spec_loss = 0
            if config.ood.ood_param > 0:
                text_spec_loss = config.ood.extra_param[0] * torch.var(torch.stack(text_loss_list))
                if torch.isnan(text_spec_loss):
                    text_spec_loss = 0
            mean_loss = loss.sum() / mask.sum()
            if config.ood.ood_param > 0:
                self.spec_loss = self.spec_loss.sum() / mask.sum()
            if config.ood.ood_param > 0:
                loss = graph_spec_loss + mean_loss + self.spec_loss + text_spec_loss
            else:
                loss = graph_spec_loss + mean_loss
            self.mean_loss = graph_spec_loss + mean_loss
            if config.ood.ood_param > 0:
                self.spec_loss = self.spec_loss + text_spec_loss
        else:
            self.mean_loss = loss.sum() / mask.sum()
            if config.ood.ood_param > 0:
                self.spec_loss = self.spec_loss.sum() / mask.sum()
                loss = self.mean_loss + self.spec_loss
            else:
                loss = self.mean_loss
        return loss
