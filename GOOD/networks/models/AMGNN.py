"""
GIN and GIN-virtual implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
<https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper
"""
import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor


@register.model_register
class AM_vGIN(GNNBasic):
    r"""
        The Graph Neural Network modified from the `"Mixup for Node and Graph Classification"
        <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper and `"Neural Message Passing for Quantum Chemistry"
        <https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(AM_vGIN, self).__init__(config)
        self.encoder = vGINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None
        self.config = config

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The Mixup-vGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): (1) dictionary of OOD args (:obj:`kwargs.ood_algorithm`) (2) key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        ood_algorithm = kwargs.get('ood_algorithm')
        out_readout = self.encoder(*args, **kwargs)

        if self.training:
            # data = kwargs.get('data')
            # assert data is not None
            targets = kwargs.get('targets')
            assert targets is not None
            mask = kwargs.get('mask')
            assert mask is not None

            lam = ood_algorithm.lam

            raw_pred = self.classifier(out_readout)
            loss = self.config.metric.loss_func(raw_pred, targets, reduction='none') * mask
            # loss = loss * node_norm * mask.sum() if self.config.model.model_level == 'node' else loss
            # Calculate the loss
            # loss = F.nll_loss(out, target)
            # Zero all existing gradients
            # self.encoder.zero_grad()
            # self.classifier.zero_grad()

            # Calculate gradients of model in backward pass
            # loss.backward()

            # Collect ``datagrad``
            meanloss = loss.sum() / mask.sum()
            data_grad = torch.autograd.grad(meanloss, out_readout, retain_graph=True)[0]
            # data_grad = out_readout.grad.data

            # Restore the data to its original scale
            # data_denorm = denorm(data)
            # Call FGSM Attack
            perturbed_data = fgsm_attack(out_readout, ood_algorithm.epsilon, data_grad)

            # Reapply normalization
            # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
            out_readout = lam * perturbed_data + (1 - lam) * out_readout[ood_algorithm.id_a2b]

            # Re-classify the perturbed image
            # output = model(perturbed_data_normalized)
            # Check for success
            # final_pred = output.max(1, keepdim=True)[1]

        out = self.classifier(out_readout)
        return out


@register.model_register
class AM_GIN(GNNBasic):
    r"""
    The Graph Neural Network modified from the `"Mixup for Node and Graph Classification"
    <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper and `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(AM_GIN, self).__init__(config)
        self.encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The Mixup-GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): (1) dictionary of OOD args (:obj:`kwargs.ood_algorithm`) (2) key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        ood_algorithm = kwargs.get('ood_algorithm')
        out_readout = self.encoder(*args, **kwargs)

        if self.training:
            # data = kwargs.get('data')
            # assert data is not None
            targets = kwargs.get('targets')
            assert targets is not None
            mask = kwargs.get('mask')
            assert mask is not None

            lam = ood_algorithm.lam

            raw_pred = self.classifier(out_readout)
            loss = self.config.metric.loss_func(raw_pred, targets, reduction='none') * mask
            # loss = loss * node_norm * mask.sum() if self.config.model.model_level == 'node' else loss
            # Calculate the loss
            # loss = F.nll_loss(out, target)
            # Zero all existing gradients
            # self.encoder.zero_grad()
            # self.classifier.zero_grad()

            # Calculate gradients of model in backward pass
            # loss.backward()

            # Collect ``datagrad``
            meanloss = loss.sum() / mask.sum()
            data_grad = torch.autograd.grad(meanloss, out_readout, retain_graph=True)[0]
            # data_grad = out_readout.grad.data

            # Restore the data to its original scale
            # data_denorm = denorm(data)
            # Call FGSM Attack
            perturbed_data = fgsm_attack(out_readout, ood_algorithm.epsilon, data_grad)

            # Reapply normalization
            # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
            out_readout = lam * perturbed_data + (1 - lam) * out_readout[ood_algorithm.id_a2b]

            # Re-classify the perturbed image
            # output = model(perturbed_data_normalized)
            # Check for success
            # final_pred = output.max(1, keepdim=True)[1]

        out = self.classifier(out_readout)
        return out


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image