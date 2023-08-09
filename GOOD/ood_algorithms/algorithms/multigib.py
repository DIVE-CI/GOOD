import torch
from torch.autograd import grad
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage

from itertools import chain
import torch.nn.functional as F
# from torch import tensor
# from torch.optim import Adam
# from sklearn.model_selection import StratifiedKFold
# from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn


@register.ood_alg_register
class multigib(BaseOODAlg):
    r"""
    Implementation of the IRM algorithm from `"Invariant Risk Minimization"
    <https://arxiv.org/abs/1907.02893>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(multigib, self).__init__(config)
        # self.perturb = None
        # self.allow_reset = True
        # self.m = int(config.ood.ood_param)
        # self.step_size = config.ood.extra_param[0]
        self.config = config

    # def stage_control(self, config):
    #     r"""
    #     Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
    #     settings.
    #
    #     Args:
    #         config: munchified dictionary of args.
    #
    #     """
    #     if self.stage == 0 and at_stage(1, config):
    #         reset_random_seed(config)
    #         config.train_helper.model.feat_encoder.encoder.ood_algorithm = self
    #         self.stage = 1

    # def reset_perturb(self, shape):
    #     if self.allow_reset:
    #         self.perturb = torch.zeros(shape, device=self.config.device, dtype=torch.float).uniform_(-self.step_size, self.step_size)
    #         self.perturb.requires_grad_()
    #         self.allow_reset = False
    #     return self.perturb

    def set_up(self, model: torch.nn.Module, config: Union[CommonArgs, Munch]):
        r"""
        Training setup of optimizer and scheduler

        Args:
            model (torch.nn.Module): model for setup
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.train.lr`, :obj:`config.metric`, :obj:`config.train.mile_stones`)

        Returns:
            None

        """
        self.model: torch.nn.Module = model
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.train.mile_stones, gamma=0.1)
        self.model.main_model.reset_parameters()
        self.optimizer_model = torch.optim.Adam(self.model.main_model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
        generators_params = []
        for generator in self.model.generators:
            generator.reset_parameters()
            generators_params.append(generator.parameters())
        # initialize optimizer
        self.optimizer_generator = torch.optim.AdamW(chain.from_iterable(generators_params), lr=config.train.lr)

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; get domain classifier predictions

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        # self.dc_pred = model_output[1]
        return model_output[0]

    def backward(self, loss: Tensor, data, targets, mask):
        config = self.config
        # loss /= self.m
        # node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
        # edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None
        #
        # for _ in range(self.m - 1):
        #     loss.backward()
        #     perturb_data = self.perturb.detach() + self.step_size * torch.sign(self.perturb.grad.detach())
        #     self.perturb.data = perturb_data.data
        #     self.perturb.grad[:] = 0
        #
        #     model_output = config.train_helper.model(data=data, edge_weight=edge_weight, ood_algorithm=self)
        #     raw_pred = self.output_postprocess(model_output)
        #     loss = self.loss_calculate(raw_pred, targets, mask, node_norm, config)
        #     loss = self.loss_postprocess(loss, data, mask, config)
        #     loss /= self.m
        #
        # loss.backward()
        # self.allow_reset = True
        inner_loop = 20
        num_generators = int(config.ood.ood_param)
        joint = int(config.ood.extra_param[0])
        dist_weight = 0.1
        kld_weight = 0.1

        for j in range(0, inner_loop):

            self.optimizer_generator.zero_grad()
            loss_array = []
            sqrt_loss_array = []
            kld_array = []
            out_og, out_embs = self.model.main_model(data)
            loss_og = config.metric.loss_func(out_og, targets, reduction='none') * mask
            # loss_og = F.nll_loss(out_og, data.y.view(-1).long())
            loss_array.append(loss_og.view(-1))

            for k in range(0, num_generators):
                generator = self.model.generators[k]
                if joint:
                    kld_loss, node_mask, edge_mask = generator(out_embs, data.edge_index, data.batch)
                else:
                    kld_loss, edge_mask = generator(out_embs, data.edge_index)
                kld_array.append(kld_loss.view(-1))
                set_masks(edge_mask, self.model.main_model)
                if joint:
                    out_local, _ = self.model.main_model(data, mask=node_mask)
                    clear_masks(self.model.main_model)
                    loss_local = config.metric.loss_func(out_local, targets, reduction='none') * mask
                    loss_array.append(loss_local.view(-1))
                    sqrt_loss_array.append(torch.sqrt(loss_local).view(-1))
                else:
                    out_local, _ = self.model.main_model(data)
                    loss_local = config.metric.loss_func(out_local, targets, reduction='none') * mask
                    loss_array.append(loss_local.view(-1))
                    sqrt_loss_array.append(torch.sqrt(loss_local).view(-1))
                    clear_masks(self.model.main_model)


            Loss = torch.cat(loss_array, dim=0)
            sqrt_loss = torch.cat(sqrt_loss_array, dim=0)
            _, Mean = torch.var_mean(Loss)
            Var, _ = torch.var_mean(sqrt_loss)
            kld_loss = torch.cat(kld_array, dim=0)
            _, kld_loss = torch.var_mean(kld_loss)

            self.optimizer_generator.zero_grad()
            loss_generator = Mean + kld_weight * kld_loss - dist_weight * Var
            loss_generator.backward()
            self.optimizer_generator.step()

        loss_array = []
        out_og, out_embs = self.model.main_model(data)
        loss_og = config.metric.loss_func(out_og, targets, reduction='none') * mask
        loss_array.append(loss_og.view(-1))

        for k in range(0, num_generators):
            generator = self.model.generators[k]
            if joint:
                kld_loss, node_mask, edge_mask = generator(out_embs, data.edge_index, data.batch)
                set_masks(edge_mask, self.model.main_model)
                out_local, _ = self.model.main_model(data, node_mask)
                clear_masks(self.model.main_model)
                loss_local = config.metric.loss_func(out_local, targets, reduction='none') * mask
                loss_array.append(loss_local.view(-1))
            else:
                kld_loss, edge_mask = generator(out_embs, data.edge_index)
                set_masks(edge_mask, self.model.main_model)
                out_local, _ = self.model.main_model(data)
                loss_local = config.metric.loss_func(out_local, targets, reduction='none') * mask
                loss_array.append(loss_local.view(-1))
                clear_masks(self.model.main_model)

        Loss = torch.cat(loss_array, dim=0)
        Var, Mean = torch.var_mean(Loss)
        self.optimizer_model.zero_grad()
        loss_classifier = Mean + dist_weight * Var
        # print(Mean, Var)
        loss_classifier.backward()
        self.optimizer_model.step()


def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None