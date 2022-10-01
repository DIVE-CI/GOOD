from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch_geometric.nn

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from torch.nn.functional import l1_loss
from .BaseGNN import GNNBasic
from .GINs import GINFeatExtractor, GINEConv
from .GINvirtualnode import vGINFeatExtractor
from .GCNs import GCNConv
import torch_geometric.nn as gnn
from .MolEncoders import AtomEncoder, BondEncoder
from torch.nn import Identity
from torch import Tensor
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
from torch_geometric.data import Batch, Data
from .PLNN import PL_GIN

@register.model_register
class SODAugNN(nn.Module):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(SODAugNN, self).__init__()
        self.VGAE = VGAE(config)
        self.PLNN = PL_GIN(config)
        # self.hidden1_dim = config.model.dim_hidden
        # self.hidden2_dim = config.model.dim_hidden
        self.device = config.device
        # self.classifier = Classifier(config)
        # self.num_predictor = Num_predictor(config)
        # self.Edge_attr_predictor = Edge_attr_predictor(config)
        # self.attr = Classifier(config)
        # self.feat_encoder = GCNFeatExtractor(config)


        if config.dataset.dataset_type == 'mol':
            self.edge_feat = True

        else:
            self.edge_feat = False

        if config.dataset.dataset_type == 'mol' and config.dataset.dataset_name != 'DD' and config.dataset.dataset_name != 'NCI1':
            self.edge_at = True

        else:
            self.edge_at = False


    def forward(self, *args, **kwargs):

        data = kwargs.get('data')
        assert data is not None

        batch_size = data.batch[-1] + 1
        # if training:
        new_batch = []
        for i in range(int(batch_size/2)):

            data_a = data[i]
            frag_1 = int(data_a.x.shape[0] / 2)
            frag_2 = data_a.x.shape[0] - frag_1
            # bi_edge_index = to_undirected(data_a.edge_index)
            edge_mask = ((data_a.edge_index[0] < frag_1) & (data_a.edge_index[1] >= frag_1)) | (
                        (data_a.edge_index[0] >= frag_1) & (data_a.edge_index[1] < frag_1))

            edge_idx = data_a.edge_index[:, ~edge_mask]
            if self.edge_at:
                edge_attr = data_a.edge_attr[~edge_mask]

            edge_mask = (data_a.edge_index[0] < frag_1) & (data_a.edge_index[1] >= frag_1)
            y = torch.zeros((frag_1, frag_2), device=self.device)
            m = data_a.edge_index[:, edge_mask]
            y[m[0], m[1] - frag_1] = 1

            if self.edge_at:
                new_batch.append(
                    Data(x=data_a.x, edge_index=edge_idx, edge_attr=edge_attr, bridge=y.view(-1),
                         bridge_num=torch.tensor(m.shape[1]).to(self.device), frag_1=frag_1, frag_2=frag_2,
                         y=data_a.y))
                # org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, edge_attr=data_a.edge_attr, y=data_a.y))
            else:
                new_batch.append(Data(x=data_a.x, edge_index=edge_idx, bridge=y.view(-1),
                                      bridge_num=torch.tensor(m.shape[1]).to(self.device), frag_1=frag_1,
                                      frag_2=frag_2, y=data_a.y))
                # new_batch_2.append(Data(x=x_2, edge_index=edge_idx_2, y=data_a.y))
                # org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, y=data_a.y))

        frag_data = Batch.from_data_list(new_batch)

        A_pred, bridge, kl_divergence, edge_attr_loss, bridge_attr, num_loss, num_pred = self.VGAE(data=frag_data)
        PL_pred = self.PLNN(data=data)

        return A_pred, bridge, kl_divergence, edge_attr_loss, bridge_attr, num_loss, num_pred, PL_pred

    @torch.no_grad()
    def bridge_sample(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        data = kwargs.get('data')
        assert data is not None
        A_pred, bridge, kl_divergence, edge_attr_loss, bridge_attr, num_loss, num_pred = self.VGAE(data=data)
        return A_pred, bridge, kl_divergence, edge_attr_loss, bridge_attr, num_loss, num_pred

    @torch.no_grad()
    def frag_sample(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        out_readout = self.PLNN.frag_sample(*args, **kwargs)
        return out_readout



@register.model_register
class VGAE(nn.Module):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(VGAE, self).__init__()
        self.input_dim = config.dataset.dim_node
        self.hidden1_dim = config.model.dim_hidden
        self.hidden2_dim = config.model.dim_hidden
        self.device = config.device
        self.classifier = Classifier(config)
        self.num_predictor = Num_predictor(config)
        self.Edge_attr_predictor = Edge_attr_predictor(config)
        # self.attr = Classifier(config)
        # self.feat_encoder = GCNFeatExtractor(config)
        if config.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif config.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()

        if kwargs.get('without_embed'):
            self.atom_encoder = Identity()
        else:
            self.atom_encoder = AtomEncoder(config.model.dim_hidden, config)

        if config.dataset.dataset_type == 'mol':  #and config.dataset.dataset_name != 'DD' and config.dataset.dataset_name != 'NCI1'
            self.edge_feat = True
            self.base_gcn = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                            nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)), config)
            # torch_geometric.nn.GraphConv
            self.base_gcn2 = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                   nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                                   nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)), config)
            self.gcn_mean = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                            nn.BatchNorm1d(2 * config.model.dim_hidden),
                                            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)), config)
            self.gcn_logstddev = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                            nn.BatchNorm1d(2 * config.model.dim_hidden),
                                            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)), config)
        else:
            self.edge_feat = False
            self.base_gcn = gnn.GINConv(nn.Sequential(nn.Linear(config.dataset.dim_node, 2 * config.model.dim_hidden),
                                                      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            self.base_gcn2 = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            # torch_geometric.nn.GraphConv
            self.gcn_mean = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                      nn.BatchNorm1d(2 * config.model.dim_hidden),
                                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            self.gcn_logstddev = gnn.GINConv(
                nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                              nn.BatchNorm1d(2 * config.model.dim_hidden),
                              nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

        if config.dataset.dataset_name != 'DD' and config.dataset.dataset_name != 'NCI1':
            self.no_edge_feat = True
        else:
            self.no_edge_feat = False
    # def encode(self, X):
    #     hidden = self.base_gcn(X)
    #     self.mean = self.gcn_mean(hidden)
    #     self.logstd = self.gcn_logstddev(hidden)
    #     gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
    #     sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
    #     return sampled_z

    def forward(self, *args, **kwargs):
        # Z = self.encode(X)
        # A_pred = dot_product_decode(Z)

        if self.edge_feat:
            data = kwargs.get('data')
            assert data is not None
            if data.bridge is not None:
                bridge = data.bridge
            else:
                bridge = None
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            bridge_num = data.bridge_num
            frag_1 = data.frag_1
            frag_2 = data.frag_2
            # x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, **kwargs)
            x = self.atom_encoder(x)
            hidden = self.base_gcn(x, edge_index, edge_attr)
            hidden = self.base_gcn2(hidden, edge_index, edge_attr)
            self.mean = self.gcn_mean(hidden, edge_index, edge_attr)
            self.logstd = self.gcn_logstddev(hidden, edge_index, edge_attr)
            gaussian_noise = torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)
            sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
            # sampled_z = torch.exp(self.logstd) + self.mean
        else:
            # x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            data = kwargs.get('data')
            assert data is not None
            if data.bridge is not None:
                bridge = data.bridge
            else:
                bridge = None
            x = data.x
            edge_index = data.edge_index
            bridge_num = data.bridge_num
            frag_1 = data.frag_1
            frag_2 = data.frag_2

            hidden = self.base_gcn(x, edge_index)
            hidden = self.base_gcn2(hidden, edge_index)
            self.mean = self.gcn_mean(hidden, edge_index)
            self.logstd = self.gcn_logstddev(hidden, edge_index)
            gaussian_noise = torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)
            sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean

        out_readout = self.readout(sampled_z, data.batch, data.batch[-1] + 1)
        num_pred = self.num_predictor(out_readout)
        num_loss = l1_loss(num_pred.squeeze(), bridge_num.squeeze())

        # bridge_feat = torch.zeros((torch.matmul(frag_1,frag_2).sum(),self.hidden2_dim * 2), device=self.device)
        bridge_idx = torch.zeros((2,torch.matmul(frag_1, frag_2).sum()), dtype=torch.long, device=self.device)
        # s = 0
        graph_done = 0
        block = 0
        for k in range(frag_1.shape[0]):
            bridge_idx[0][block: block+frag_1[k] * frag_2[k]] = torch.arange(0, frag_1[k]).unsqueeze(1).expand(frag_1[k], frag_2[k]).reshape(-1) + graph_done
            bridge_idx[1][block: block + frag_1[k] * frag_2[k]] = torch.arange(0, frag_2[k]).expand(frag_1[k], frag_2[k]).reshape(-1) + graph_done + frag_1[k]
            # for i in range(frag_1[k]):
            #     for j in range(frag_2[k]):
            #         bridge_idx[s][0] = graph_done+i
            #         bridge_idx[s][1] = graph_done+frag_1[k]+j
            #         # bridge_feat[s] = torch.cat((sampled_z[graph_done+i],sampled_z[graph_done+frag_1[k]+j]))
            #         s = s+1
            graph_done = graph_done + frag_1[k] + frag_2[k]
            block = block + frag_1[k] * frag_2[k]

        temp = sampled_z[bridge_idx.reshape(-1)]
        bridge_feat = torch.cat((temp[:int(temp.shape[0]/2)], temp[int(temp.shape[0]/2):]),1)
        A_pred = torch.sigmoid(self.classifier(bridge_feat))

        kl_divergence = 0.5 / sampled_z.shape[0] * (
                1 + 2 * self.logstd - self.mean ** 2 - torch.exp(self.logstd) ** 2).sum(1).mean()

        # if self.no_edge_feat:
        #     self.edge_feat = False
        edge_attr_loss = None
        bridge_attr = None
        if self.edge_feat and self.no_edge_feat:
            attr_pred = self.Edge_attr_predictor(torch.cat((sampled_z[edge_index[0]], sampled_z[edge_index[1]]), 1))
            edge_attr_loss = l1_loss(attr_pred, edge_attr)
            bridge_attr = self.Edge_attr_predictor(bridge_feat)

        # s = 0
        # num_bridge = 2
        # for k in range(frag_1.shape[0]):
        #     score = A_pred[s:s + frag_1[k] * frag_2[k]]
        #     s = s + frag_1[k] * frag_1[k]
        #     v, indices = torch.topk(score.squeeze(), num_bridge)
        #     indices

        return A_pred, bridge, kl_divergence, edge_attr_loss, bridge_attr, num_loss, num_pred


@register.model_register
class MDVGAE(nn.Module):
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(MDVGAE, self).__init__()
        self.input_dim = config.dataset.dim_node
        self.hidden1_dim = config.model.dim_hidden
        self.hidden2_dim = config.model.dim_hidden
        self.device = config.device
        self.classifier = Classifier(config)
        self.classifier2 = Classifier(config)
        self.classifier3 = Classifier(config)
        self.Edge_attr_predictor = Edge_attr_predictor(config)
        # self.attr = Classifier(config)
        # self.feat_encoder = GCNFeatExtractor(config)
        if kwargs.get('without_embed'):
            self.atom_encoder = Identity()
        else:
            self.atom_encoder = AtomEncoder(config.model.dim_hidden, config)

        if config.dataset.dataset_type == 'mol' and config.dataset.dataset_name != 'DD' and config.dataset.dataset_name != 'NCI1':
            self.edge_feat = True
            self.base_gcn = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                            nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            # torch_geometric.nn.GraphConv
            self.base_gcn2 = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                   nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                                   nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            self.gcn_mean = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                            nn.BatchNorm1d(2 * config.model.dim_hidden),
                                            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            self.gcn_logstddev = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                            nn.BatchNorm1d(2 * config.model.dim_hidden),
                                            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
        else:
            self.edge_feat = False
            self.base_gcn = gnn.GINConv(nn.Sequential(nn.Linear(config.dataset.dim_node, 2 * config.model.dim_hidden),
                                                      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            self.base_gcn2 = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            # torch_geometric.nn.GraphConv
            self.gcn_mean = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                      nn.BatchNorm1d(2 * config.model.dim_hidden),
                                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            self.gcn_logstddev = gnn.GINConv(
                nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                              nn.BatchNorm1d(2 * config.model.dim_hidden),
                              nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

    # def encode(self, X):
    #     hidden = self.base_gcn(X)
    #     self.mean = self.gcn_mean(hidden)
    #     self.logstd = self.gcn_logstddev(hidden)
    #     gaussian_noise = torch.randn(X.size(0), self.hidden2_dim)
    #     sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
    #     return sampled_z

    def forward(self, *args, **kwargs):
        # Z = self.encode(X)
        # A_pred = dot_product_decode(Z)

        if self.edge_feat:
            data = kwargs.get('data')
            assert data is not None
            if data.bridge is not None:
                bridge = data.bridge
            else:
                bridge = None
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            frag_1 = data.frag_1
            frag_2 = data.frag_2
            # x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, **kwargs)
            x = self.atom_encoder(x)
            hidden = self.base_gcn(x, edge_index, edge_attr)
            hidden = self.base_gcn2(hidden, edge_index, edge_attr)
            self.mean = self.gcn_mean(hidden, edge_index, edge_attr)
            self.logstd = self.gcn_logstddev(hidden, edge_index, edge_attr)
            gaussian_noise = torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)
            sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
            gaussian_noise2 = torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)
            sampled_z2 = gaussian_noise2 * torch.exp(self.logstd) + self.mean
            gaussian_noise3 = torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)
            sampled_z3 = gaussian_noise3 * torch.exp(self.logstd) + self.mean
            # sampled_z = torch.exp(self.logstd) + self.mean
        else:
            # x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            data = kwargs.get('data')
            assert data is not None
            if data.bridge is not None:
                bridge = data.bridge
            else:
                bridge = None
            x = data.x
            edge_index = data.edge_index
            frag_1 = data.frag_1
            frag_2 = data.frag_2

            hidden = self.base_gcn(x, edge_index)
            hidden = self.base_gcn2(hidden, edge_index)
            self.mean = self.gcn_mean(hidden, edge_index)
            self.logstd = self.gcn_logstddev(hidden, edge_index)
            gaussian_noise = torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)
            sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
            gaussian_noise2 = torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)
            sampled_z2 = gaussian_noise2 * torch.exp(self.logstd) + self.mean
            gaussian_noise3 = torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)
            sampled_z3 = gaussian_noise3 * torch.exp(self.logstd) + self.mean

        # bridge_feat = torch.zeros((torch.matmul(frag_1,frag_2).sum(),self.hidden2_dim * 2), device=self.device)
        bridge_idx = torch.zeros((2,torch.matmul(frag_1, frag_2).sum()), dtype=torch.long, device=self.device)
        # s = 0
        graph_done = 0
        block = 0
        for k in range(frag_1.shape[0]):
            bridge_idx[0][block: block+frag_1[k] * frag_2[k]] = torch.arange(0, frag_1[k]).unsqueeze(1).expand(frag_1[k], frag_2[k]).reshape(-1) + graph_done
            bridge_idx[1][block: block + frag_1[k] * frag_2[k]] = torch.arange(0, frag_2[k]).expand(frag_1[k], frag_2[k]).reshape(-1) + graph_done + frag_1[k]
            # for i in range(frag_1[k]):
            #     for j in range(frag_2[k]):
            #         bridge_idx[s][0] = graph_done+i
            #         bridge_idx[s][1] = graph_done+frag_1[k]+j
            #         # bridge_feat[s] = torch.cat((sampled_z[graph_done+i],sampled_z[graph_done+frag_1[k]+j]))
            #         s = s+1
            graph_done = graph_done + frag_1[k] + frag_2[k]
            block = block + frag_1[k] * frag_2[k]

        temp = sampled_z[bridge_idx.reshape(-1)]
        bridge_feat = torch.cat((temp[:int(temp.shape[0]/2)], temp[int(temp.shape[0]/2):]),1)
        A_pred1 = self.classifier(bridge_feat)
        temp2 = sampled_z2[bridge_idx.reshape(-1)]
        bridge_feat2 = torch.cat((temp2[:int(temp2.shape[0] / 2)], temp2[int(temp2.shape[0] / 2):]), 1)
        A_pred2 = self.classifier2(bridge_feat2)
        temp3 = sampled_z3[bridge_idx.reshape(-1)]
        bridge_feat3 = torch.cat((temp3[:int(temp3.shape[0] / 2)], temp3[int(temp3.shape[0] / 2):]), 1)
        A_pred3 = self.classifier3(bridge_feat3)
        A_pred = (torch.sigmoid(A_pred1)+torch.sigmoid(A_pred2)+torch.sigmoid(A_pred3))/3

        kl_divergence = 0.5 / sampled_z.shape[0] * (
                1 + 2 * self.logstd - self.mean ** 2 - torch.exp(self.logstd) ** 2).sum(1).mean()

        edge_attr_loss = None
        bridge_attr = None
        if self.edge_feat:
            attr_pred = self.Edge_attr_predictor(torch.cat((sampled_z[edge_index[0]], sampled_z[edge_index[1]]), 1))
            edge_attr_loss = l1_loss(attr_pred, edge_attr)
            bridge_attr = self.Edge_attr_predictor(bridge_feat)

        # s = 0
        # num_bridge = 2
        # for k in range(frag_1.shape[0]):
        #     score = A_pred[s:s + frag_1[k] * frag_2[k]]
        #     s = s + frag_1[k] * frag_1[k]
        #     v, indices = torch.topk(score.squeeze(), num_bridge)
        #     indices

        return A_pred, bridge, kl_divergence, edge_attr_loss, bridge_attr


class Classifier(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):

        super(Classifier, self).__init__()
        # self.classifier = nn.Sequential(*(
        #         [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_ffn), nn.BatchNorm1d(2 * config.model.dim_ffn)] +
        #         [nn.ReLU(), nn.Linear(2 * config.model.dim_ffn, config.dataset.num_classes)]
        # ))
        self.classifier = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden * 2, 1)]
        ))

    def forward(self, feat: Tensor) -> Tensor:
        r"""
        Applies a linear transformation to feature representations.

        Args:
            feat (Tensor): feature representations

        Returns (Tensor):
            label predictions

        """
        return self.classifier(feat)


class Num_predictor(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):

        super(Num_predictor, self).__init__()
        # self.classifier = nn.Sequential(*(
        #         [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_ffn), nn.BatchNorm1d(2 * config.model.dim_ffn)] +
        #         [nn.ReLU(), nn.Linear(2 * config.model.dim_ffn, config.dataset.num_classes)]
        # ))
        self.classifier = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden, 1)]
        ))

    def forward(self, feat: Tensor) -> Tensor:
        r"""
        Applies a linear transformation to feature representations.

        Args:
            feat (Tensor): feature representations

        Returns (Tensor):
            label predictions

        """
        return self.classifier(feat)


class Edge_attr_predictor(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):

        super(Edge_attr_predictor, self).__init__()
        # self.classifier = nn.Sequential(*(
        #         [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_ffn), nn.BatchNorm1d(2 * config.model.dim_ffn)] +
        #         [nn.ReLU(), nn.Linear(2 * config.model.dim_ffn, config.dataset.num_classes)]
        # ))
        self.predictor = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden * 2, config.dataset.dim_edge)]
        ))

    def forward(self, feat: Tensor) -> Tensor:
        r"""
        Applies a linear transformation to feature representations.

        Args:
            feat (Tensor): feature representations

        Returns (Tensor):
            label predictions

        """
        return self.predictor(feat)


# class GraphConvSparse(nn.Module):
#     def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
#         super(GraphConvSparse, self).__init__(**kwargs)
#         self.weight = glorot_init(input_dim, output_dim)
#         self.adj = adj
#         self.activation = activation
#
#     def forward(self, inputs):
#         x = inputs
#         x = torch.mm(x, self.weight)
#         x = torch.mm(self.adj, x)
#         outputs = self.activation(x)
#         return outputs
#
#
# def dot_product_decode(Z):
#     A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
#     return A_pred
#
#
# def glorot_init(input_dim, output_dim):
#     init_range = np.sqrt(6.0 / (input_dim + output_dim))
#     initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
#     return nn.Parameter(initial)

# class GAE(nn.Module):
# 	def __init__(self,adj):
# 		super(GAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
#
# 	def encode(self, X):
# 		hidden = self.base_gcn(X)
# 		z = self.mean = self.gcn_mean(hidden)
# 		return z
#
# 	def forward(self, X):
# 		Z = self.encode(X)
# 		A_pred = dot_product_decode(Z)
# 		return A_pred
