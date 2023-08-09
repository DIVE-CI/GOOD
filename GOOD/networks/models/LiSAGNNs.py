import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .MolEncoders import AtomEncoder, BondEncoder
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor
from typing import Tuple
from torch_geometric.nn.conv import MessagePassing

# import torch
import torch.nn.functional as F
# import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, JumpingKnowledge
# from torch_geometric.utils import to_dense_adj, to_dense_batch, add_self_loops
# import torch_scatter as tscatter

@register.model_register
class LiSA_vGIN(GNNBasic):
    r"""
        The Graph Neural Network modified from the `"Domain-Adversarial Training of Neural Networks"
        <https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper and `"Neural Message Passing for Quantum Chemistry"
        <https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_envs`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        # self.encoder = vGINFeatExtractor(config)
        # self.classifier = Classifier(config)
        #
        # self.dc = nn.Linear(config.model.dim_hidden, config.dataset.num_envs)
        #
        # self.dropout = nn.Dropout(config.model.dropout_rate)
        # self.graph_repr = None
        self.config = config
        self.joint = int(config.ood.extra_param[0])
        # if config.dataset.num_classes == 1:
        #     self.main_model = MultiGCN(config.dataset.dim_node, config.dataset.num_classes + 1, config.model.model_layer, config.model.dim_hidden, self.joint, config).to(config.device)
        # else:
        self.main_model = MultiGCN(config.dataset.dim_node, config.dataset.num_classes, config.model.model_layer, config.model.dim_hidden, self.joint, config).to(config.device)
        self.generators = []
        for i in range(int(config.ood.ood_param)):
            if self.joint:
                self.generators.append(JointGenerator(hidden_size=config.model.dim_hidden, device=config.device).to(config.device))
            else:
                self.generators.append(EdgeGenerator(hidden_size=config.model.dim_hidden, device=config.device).to(config.device))


    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        The DANN-vGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            [label predictions, domain predictions]

        """
        data = kwargs.get('data')
        assert data is not None

        out_og, out_embs = self.main_model(data)
        # out_readout = self.encoder(*args, **kwargs)
        # self.graph_repr = out_readout
        #
        # dc_out = GradientReverseLayerF.apply(out_readout, self.config.train.alpha)
        # dc_out = self.dc(dc_out)
        #
        # out = self.classifier(out_readout)
        return out_og, out_embs


class MultiGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden, joint, config: Union[CommonArgs, Munch]):
        super(MultiGCN, self).__init__()

        self.ismol = False
        if config.dataset.dataset_type == 'mol':
            self.ismol = True
            self.atom_encoder = AtomEncoder(config.model.dim_hidden, config)
            self.conv1 = GINConv(Sequential(
                Linear(hidden, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        else:
            self.conv1 = GINConv(Sequential(
                Linear(num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GINConv(Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.joint = joint

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data, mask = None):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.ismol:
            x = self.atom_encoder(x)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        if mask is not None:
            embeds = mask * x
        else:
            embeds = x
        x = global_mean_pool(embeds, batch)
        x = self.lin1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        out = torch.clamp(x, -10, 10)
        # return F.log_softmax(out, dim=-1), embeds
        return out, embeds

    def __repr__(self):
        return self.__class__.__name__


#edge
class EdgeGenerator(torch.nn.Module):
    def __init__(self, hidden_size, device):
        super(EdgeGenerator, self).__init__()

        self.input_size = 2 * hidden_size
        self.device = device
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.5):
        bias = bias + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + sampling_weights) / temperature
        graph = torch.sigmoid(gate_inputs)

        return graph

    def _kld(self, mask):
        pos = mask
        neg = 1 - mask
        kld_loss = torch.mean(pos * torch.log(pos/0.5 + 0.00000001) + neg * torch.log(neg/0.5 + 0.000000001))

        return kld_loss

    def _create_explainer_input(self, pair, embeds):
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl

    def forward(self, x_embeds, edges):
        input_embs = self._create_explainer_input(edges, x_embeds)
        pre = self.relu(self.lin1(input_embs))
        pre = self.lin2(pre)
        mask = self._sample_graph(pre)
        kld_loss = self._kld(mask)

        return kld_loss, mask

class JointGenerator(torch.nn.Module):
    def __init__(self, hidden_size, device):
        super(JointGenerator, self).__init__()

        self.input_size = hidden_size
        self.device = device
        self.hidden_size = hidden_size
        self.lin1 = torch.nn.Linear(self.input_size,self.hidden_size)
        self.lin2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu = torch.nn.ReLU()
        self.rate = 0.7
        self.epsilon = 0.00000001

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def _kld(self, mask):
        pos = mask
        neg = 1 - mask
        kld_loss = torch.mean(pos * torch.log(pos/0.5 + self.epsilon) + neg * torch.log(neg/0.5 + self.epsilon))

        return kld_loss

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.5):
        bias = bias + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + sampling_weights) / temperature
        graph = torch.sigmoid(gate_inputs)

        return graph

    def edge_sample(self, node_mask, edge_idx):
        src_val = node_mask[edge_idx[0]]
        dst_val = node_mask[edge_idx[1]]
        edge_val = 0.5 * (src_val + dst_val)

        return edge_val


    def forward(self, x_embeds, edges, batch):
        input_embs = x_embeds
        pre = self.relu(self.lin1(input_embs))
        pre = self.lin2(pre)
        pre = torch.clamp(pre, min = -10, max = 10)
        node_mask = self._sample_graph(pre)
        kld_loss = self._kld(node_mask)
        edge_mask = self.edge_sample(node_mask, edges)

        return kld_loss, node_mask, edge_mask

