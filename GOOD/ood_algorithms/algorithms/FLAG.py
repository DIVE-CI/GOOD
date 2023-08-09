"""
Implementation of the IRM algorithm from `"Invariant Risk Minimization"
<https://arxiv.org/abs/1907.02893>`_ paper
"""
import torch
from torch.autograd import grad
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage


@register.ood_alg_register
class FLAG(BaseOODAlg):
    r"""
    Implementation of the IRM algorithm from `"Invariant Risk Minimization"
    <https://arxiv.org/abs/1907.02893>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(FLAG, self).__init__(config)
        self.perturb = None
        self.allow_reset = True
        self.m = int(config.ood.ood_param)
        self.step_size = config.ood.extra_param[0]
        self.config = config

    def stage_control(self, config):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            config.train_helper.model.feat_encoder.encoder.ood_algorithm = self
            self.stage = 1

    def reset_perturb(self, shape):
        if self.allow_reset:
            self.perturb = torch.zeros(shape, device=self.config.device, dtype=torch.float).uniform_(-self.step_size, self.step_size)
            self.perturb.requires_grad_()
            self.allow_reset = False
        return self.perturb

    def backward(self, loss: Tensor, data, targets, mask):
        config = self.config
        loss /= self.m
        node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
        edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None

        for _ in range(self.m - 1):
            loss.backward()
            perturb_data = self.perturb.detach() + self.step_size * torch.sign(self.perturb.grad.detach())
            self.perturb.data = perturb_data.data
            self.perturb.grad[:] = 0

            model_output = config.train_helper.model(data=data, edge_weight=edge_weight, ood_algorithm=self)
            raw_pred = self.output_postprocess(model_output)
            loss = self.loss_calculate(raw_pred, targets, mask, node_norm, config)
            loss = self.loss_postprocess(loss, data, mask, config)
            loss /= self.m

        loss.backward()
        self.allow_reset = True
        self.optimizer.step()
