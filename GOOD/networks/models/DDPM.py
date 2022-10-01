
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Function
from torch import Tensor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from torch.nn.functional import l1_loss
from .Classifiers import Classifier
from .GINs import GINFeatExtractor, GINEConv
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
import torch_geometric.nn as gnn
from .GINvirtualnode import vGINMolEncoder, vGINEncoder
from typing import Tuple
from torch_geometric.data import Batch, Data

@register.model_register
class DDPM_vGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        # self.encoder = vGINFeatExtractor(config)
        # self.classifier = Classifier(config)
        # self.Edge_attr_MLP = Edge_attr_MLP(config)
        # self.Edge_idx_MLP = Edge_idx_MLP(config)
        # if config.dataset.dataset_name != 'GOODSST2':
        #     self.Edge_attr_MLP2 = Edge_attr_MLP(config)
        #     self.Edge_idx_MLP2 = Edge_idx_MLP(config)

        # self.dc = nn.Linear(config.model.dim_hidden, config.dataset.num_envs)

        # self.dropout = nn.Dropout(config.model.dropout_rate)
        # self.graph_repr = None
        self.config = config

        self.num_bridge = 1 if config.dataset.dataset_name == 'GOODSST2' else 2
        self.T = 1000

        self.gamma = GammaNetwork()
        self.device = config.device
        if config.dataset.dataset_type == 'mol':
            # self.encoder = vGINMolEncoder(config)
            # self.encoder = vGINEncoder(config)
            self.edge_feat = True
            self.attr_predictor = nn.Linear(config.model.dim_hidden * 2, config.dataset.dim_edge)
            self.base_gin = GINEConv(
                nn.Sequential(nn.Linear(config.dataset.dim_node + 1, 2 * config.model.dim_hidden),
                              nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                              nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            self.base_gin2 = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                       nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                                       nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            # torch_geometric.nn.GraphConv
            self.base_gin3 = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                       nn.BatchNorm1d(2 * config.model.dim_hidden),
                                                       nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
        else:
            # self.encoder = vGINEncoder(config)
            self.edge_feat = False
            self.base_gin = gnn.GINConv(
                nn.Sequential(nn.Linear(config.dataset.dim_node + 1, 2 * config.model.dim_hidden),
                              nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                              nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            self.base_gin2 = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                       nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                                       nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
            # torch_geometric.nn.GraphConv
            self.base_gin3 = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                                       nn.BatchNorm1d(2 * config.model.dim_hidden),
                                                       nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

        if config.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif config.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()

        self.upfill = nn.Linear(1, 100)
        self.bridge_predictor = nn.Linear(2 * config.model.dim_hidden + 100, 1)
        self.num_predictor = nn.Linear(config.model.dim_hidden, 1)


        # if noise_schedule == 'learned':
        #     assert loss_type == 'vlb', 'A noise schedule can only be learned with a vlb objective'
        #     self.gamma = GammaNetwork()
        # else:
        #     self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=1000, precision=1e-4)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        The DANN-vGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            [label predictions, domain predictions]

        """
        # out_readout = self.encoder(*args, **kwargs)
        # self.graph_repr = out_readout
        #
        # out = self.classifier(out_readout)
        #
        data = kwargs.get('data')
        assert data is not None
        # if data.bridge is not None:
        #     bridge = data.bridge
        # else:
        #     bridge = None
        # x = data.x
        # edge_index = data.edge_index
        # edge_attr = data.edge_attr

        # self.lam = 0.5
        batch_size = data.batch[-1] + 1

        # Sample t
        t_int = torch.randint(0, self.T + 1, (batch_size,1), device=self.device).float()
        s_int = t_int - 1
        t = t_int / self.T
        s = s_int / self.T

        # Masks for t=0 and t>0
        # t_is_zero = (t_int == 0).squeeze().float()
        # t_is_not_zero = 1 - t_is_zero

        # Compute gamma_t and gamma_s according to the noise schedule
        gamma_t = self.inflate_batch_array(self.gamma(t), t)
        gamma_s = self.inflate_batch_array(self.gamma(s), t)

        # Compute alpha_t and sigma_t from gamma
        alpha_t = self.alpha(gamma_t, t)
        sigma_t = self.sigma(gamma_t, t)
        # if training:
        new_batch = []
        bridge_gt_list = None
        eps_t = None
        vlb = None
        # noise1_list = []
        # noise2_list = []
        for i in range(batch_size):

            data_a = data[i]
            frag_1 = int(data_a.x.shape[0] / 2)
            frag_2 = data_a.x.shape[0] - frag_1
            edge_mask = ((data_a.edge_index[0] < frag_1) & (data_a.edge_index[1] >= frag_1)) | (
                        (data_a.edge_index[0] >= frag_1) & (data_a.edge_index[1] < frag_1))

            di_edge_mask = (data_a.edge_index[0] < frag_1) & (data_a.edge_index[1] >= frag_1)
            noise1 = torch.randn((frag_1, frag_2), device=self.device)
            bridge_gt = torch.zeros((frag_1, frag_2), device=self.device)
            m = data_a.edge_index[:, di_edge_mask]
            bridge_gt[m[0], m[1] - frag_1] = 1
            # mean_noise1 = torch.mean(noise1, 1, True)
            # noise1_list.append(torch.zeros_like(mean_noise1) if torch.isnan(mean_noise1).sum() else mean_noise1)
            noise_bridge = alpha_t[i] * bridge_gt + sigma_t[i] * noise1
            if bridge_gt_list is not None:
                bridge_gt_list = torch.cat((bridge_gt_list, noise_bridge.reshape(-1)))
            else:
                bridge_gt_list = noise_bridge.reshape(-1)

            if eps_t is not None:
                eps_t = torch.cat((eps_t, noise1.reshape(-1)))
            else:
                eps_t = noise1.reshape(-1)

            if vlb is not None:
                vlb = torch.cat((vlb, (gamma_s[i] - gamma_t[i]).squeeze().expand(frag_1*frag_2)))
            else:
                vlb = (gamma_s[i] - gamma_t[i]).squeeze().expand(frag_1*frag_2)

            edge_idx = (data_a.edge_index[:, ~edge_mask])
            # if self.edge_feat:
            #     noise2 = torch.randn(data_a.edge_attr[di_edge_mask].size(), device=self.device)
            #     # noise2_list.append(torch.mean(noise2, 0, True))
            #     mean_noise2 = torch.mean(noise2, 0, True)
            #     noise2_list.append(torch.zeros_like(mean_noise2) if torch.isnan(mean_noise2).sum() else mean_noise2)
            #     noise_bridge_attr = abs(alpha_t[i] * data_a.edge_attr[di_edge_mask] + sigma_t[i] * noise2)
            #     if noise_bridge_attr.shape[0]>0:
            #         for co in range(noise_bridge_attr.shape[0]):
            #             temp = torch.norm(data.edge_attr - noise_bridge_attr[co], dim=1).argmin().unsqueeze(0)
            #             if co == 0:
            #                 bridge_attr_idx = temp
            #             else:
            #                 bridge_attr_idx = torch.cat((bridge_attr_idx, temp))
            #         noise_bridge_attr = data.edge_attr[bridge_attr_idx]

            if self.edge_feat:
                edge_attr = data_a.edge_attr[~edge_mask]

            x = torch.cat((data_a.x, t_int[i].type(data_a.x.dtype).expand(data_a.x.shape[0]).unsqueeze(dim=1)), dim=1)

            if self.edge_feat:
                new_batch.append(
                    Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, frag_1=frag_1, frag_2=frag_2, bridge_num=torch.tensor(m.shape[1]).to(self.device), y=data_a.y))
                # org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, edge_attr=data_a.edge_attr, y=data_a.y))
            else:
                new_batch.append(
                    Data(x=x, edge_index=edge_idx, frag_1=frag_1, frag_2=frag_2, bridge_num=torch.tensor(m.shape[1]).to(self.device), y=data_a.y))
                # new_batch_2.append(Data(x=x_2, edge_index=edge_idx_2, y=data_a.y))
                # org_batch.append(Data(x=data_a.x, edge_index=data_a.edge_index, y=data_a.y))

        data = Batch.from_data_list(new_batch)

        # hidden = self.base_gcn(x, edge_index)
        # hidden = self.base_gcn2(hidden, edge_index)
        # self.mean = self.gcn_mean(hidden, edge_index)
        # self.logstd = self.gcn_logstddev(hidden, edge_index)
        # gaussian_noise = torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)
        # sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        #
        # out_readout = self.readout(sampled_z, data.batch, data.batch[-1] + 1)

        if self.edge_feat:
            hidden = self.base_gin(data.x, data.edge_index, data.edge_attr)
            hidden = self.base_gin2(hidden, data.edge_index, data.edge_attr)
            hidden = self.base_gin3(hidden, data.edge_index, data.edge_attr)
            # feat = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch, batch_size)
        else:
            hidden = self.base_gin(data.x, data.edge_index)
            hidden = self.base_gin2(hidden, data.edge_index)
            hidden = self.base_gin3(hidden, data.edge_index)
            # feat = self.encoder(data.x, data.edge_index, data.batch, batch_size)

        frag_1 = data.frag_1
        frag_2 = data.frag_2

        bridge_idx = torch.zeros((2, torch.matmul(frag_1, frag_2).sum()), dtype=torch.long, device=self.device)
        # s = 0
        graph_done = 0
        block = 0
        for k in range(frag_1.shape[0]):
            bridge_idx[0][block: block + frag_1[k] * frag_2[k]] = torch.arange(0, frag_1[k]).unsqueeze(1).expand(
                frag_1[k], frag_2[k]).reshape(-1) + graph_done
            bridge_idx[1][block: block + frag_1[k] * frag_2[k]] = torch.arange(0, frag_2[k]).expand(frag_1[k],
                                                                                                    frag_2[k]).reshape(
                -1) + graph_done + frag_1[k]
            # for i in range(frag_1[k]):
            #     for j in range(frag_2[k]):
            #         bridge_idx[s][0] = graph_done+i
            #         bridge_idx[s][1] = graph_done+frag_1[k]+j
            #         # bridge_feat[s] = torch.cat((sampled_z[graph_done+i],sampled_z[graph_done+frag_1[k]+j]))
            #         s = s+1
            graph_done = graph_done + frag_1[k] + frag_2[k]
            block = block + frag_1[k] * frag_2[k]

        temp = hidden[bridge_idx.reshape(-1).detach()]
        bridge_feat = torch.cat((temp[:int(temp.shape[0] / 2)], temp[int(temp.shape[0] / 2):], self.upfill(bridge_gt_list.unsqueeze(1))), 1)

        eps_t_hat = torch.sigmoid(self.bridge_predictor(bridge_feat))
        e1 = (eps_t_hat.squeeze()-eps_t.squeeze())**2
        SNR_weight = (self.SNR(vlb) - 1).squeeze()
        l_loss = self.T * 0.5 * SNR_weight * e1
        l_loss = l_loss.sum() / l_loss.shape[0]
        # l2_loss = e1.mean()

        edge_attr_loss = None
        # bridge_attr = None
        if self.edge_feat:
            attr_pred = self.attr_predictor(torch.cat((hidden[data.edge_index[0]], hidden[data.edge_index[1]]), 1))
            edge_attr_loss = l1_loss(attr_pred, data.edge_attr)
            # bridge_attr = self.attr_predictor(bridge_feat)

        out_readout = self.readout(hidden, data.batch, data.batch[-1] + 1)
        num_pred = self.num_predictor(out_readout)
        num_loss = l1_loss(num_pred.squeeze(), data.bridge_num.squeeze())


        # eps_t_hat_1 = self.Edge_idx_MLP(feat)
        # eps_t_hat_1s = self.Edge_idx_MLP2(feat)
        # if self.edge_feat:
        #     eps_t_hat_2 = self.Edge_attr_MLP(feat)
        #     eps_t_hat_2s= self.Edge_attr_MLP2(feat)
        #     e2 = (eps_t_hat_2 - torch.cat(noise2_list, dim=0).reshape(-1, 3).detach()) ** 2 + (
        #                 eps_t_hat_2s - torch.cat(noise2_list, dim=0).reshape(-1, 3).detach()) ** 2
        # e1 = (eps_t_hat_1 - torch.cat(noise1_list, dim=0).reshape(-1,2).detach())**2 + (eps_t_hat_1s - torch.cat(noise1_list, dim=0).reshape(-1,2).detach())**2


        # Sample noise
        # Note: only for linker
        # eps_t = self.sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), mask=linker_mask)
        # torch.randn(hidden.shape[0], self.hidden2_dim, device=self.device)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        # Note: keep fragments unchanged
        # z_t = alpha_t * xh + sigma_t * eps_t
        # z_t = xh * fragment_mask + z_t * linker_mask

        # Neural net prediction
        # eps_t_hat = self.dynamics.forward(
        #     xh=z_t,
        #     t=t,
        #     node_mask=node_mask,
        #     linker_mask=linker_mask,
        #     context=context,
        #     edge_mask=edge_mask,
        # )
        # eps_t_hat = eps_t_hat * linker_mask

        # Computing basic error (further used for computing NLL and L2-loss)
        # error_t = self.sum_except_batch((eps_t - eps_t_hat) ** 2)
        # if self.edge_feat:
        #     error_t = e1.view(e1.size(0), -1).sum(-1)/(2*batch_size.detach()) + e2.view(e2.size(0), -1).sum(-1)/(3*batch_size.detach())
        # else:
        #     error_t = e1.view(e1.size(0), -1).sum(-1) / (2 * batch_size.detach())
        # l2_loss = error_t.mean()/2

        # Computing L2-loss for t>0
        # normalization = (self.n_dims + self.in_node_nf) * self.numbers_of_nodes(linker_mask)
        # l2_loss = error_t / normalization
        # l2_loss = l2_loss.mean()

        # The KL between q(z_T | x) and p(z_T) = Normal(0, 1) (should be close to zero)
        # kl_prior = self.kl_prior(xh, linker_mask).mean()

        # Computing NLL middle term
        # SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        # loss_term_t = self.T * 0.5 * SNR_weight * error_t
        # loss_term_t = (loss_term_t * t_is_not_zero).sum() / t_is_not_zero.sum()
        return l_loss, edge_attr_loss, num_loss

    @torch.no_grad()
    def sample(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        data = kwargs.get('data')
        assert data is not None
        # if data.bridge is not None:
        #     bridge = data.bridge
        # else:
        #     bridge = None
        # x = data.x
        # edge_index = data.edge_index
        # edge_attr = data.edge_attr

        # self.lam = 0.5
        batch_size = data.batch[-1] + 1
        # linker_mask = torch.nonzero(data.bridge_mask.reshape(-1)).squeeze()
        feat = None

        frag_1 = data.frag_1
        frag_2 = data.frag_2

        bridge_idx = torch.zeros((2, torch.matmul(frag_1, frag_2).sum()), dtype=torch.long, device=self.device)
        # s = 0
        graph_done = 0
        block = 0
        for k in range(frag_1.shape[0]):
            bridge_idx[0][block: block + frag_1[k] * frag_2[k]] = torch.arange(0, frag_1[k]).unsqueeze(1).expand(
                frag_1[k], frag_2[k]).reshape(-1) + graph_done
            bridge_idx[1][block: block + frag_1[k] * frag_2[k]] = torch.arange(0, frag_2[k]).expand(frag_1[k],
                                                                                                    frag_2[k]).reshape(
                -1) + graph_done + frag_1[k]
            # for i in range(frag_1[k]):
            #     for j in range(frag_2[k]):
            #         bridge_idx[s][0] = graph_done+i
            #         bridge_idx[s][1] = graph_done+frag_1[k]+j
            #         # bridge_feat[s] = torch.cat((sampled_z[graph_done+i],sampled_z[graph_done+frag_1[k]+j]))
            #         s = s+1
            graph_done = graph_done + frag_1[k] + frag_2[k]
            block = block + frag_1[k] * frag_2[k]

        z_t = data.z_t
        for si in reversed(range(0, self.T)):
            # s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            # t_array = s_array + 1
            # s_array = s_array / self.T
            # t_array = t_array / self.T
            s = torch.full((1, 1), fill_value=si, device=self.device)
            t = s + 1
            s = s / self.T
            t = t / self.T
            # z = self.sample_p_zs_given_zt_only_linker()
            gamma_s = self.gamma(s)
            gamma_t = self.gamma(t)

            sigma2_t_given_s = -self.expm1(self.softplus(gamma_s) - self.softplus(gamma_t))
            log_alpha2_t = F.logsigmoid(-gamma_t)
            log_alpha2_s = F.logsigmoid(-gamma_s)
            log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

            alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

            sigma_s = self.sigma(gamma_s, target_tensor=t)
            sigma_t = self.sigma(gamma_t, target_tensor=t)

            x = torch.cat((data.x, (s+1).squeeze().type(data.x.dtype).expand(data.x.shape[0]).unsqueeze(dim=1)), dim=1)
            if self.edge_feat:
                hidden = self.base_gin(x, data.edge_index, data.edge_attr)
                hidden = self.base_gin2(hidden, data.edge_index, data.edge_attr)
                hidden = self.base_gin3(hidden, data.edge_index, data.edge_attr)
                # feat = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch, batch_size)
            else:
                hidden = self.base_gin(x, data.edge_index)
                hidden = self.base_gin2(hidden, data.edge_index)
                hidden = self.base_gin3(hidden, data.edge_index)
                # feat = self.encoder(data.x, data.edge_index, data.batch, batch_size)

            temp = hidden[bridge_idx.reshape(-1)]
            bridge_feat = torch.cat((temp[:int(temp.shape[0] / 2)], temp[int(temp.shape[0] / 2):], self.upfill(z_t.unsqueeze(1))), 1)

            eps_t_hat = torch.sigmoid(self.bridge_predictor(bridge_feat))

            # Compute mu for p(z_s | z_t)
            # Sample z_s given the parameters derived from zt
            # Compute sigma for p(z_s | z_t)
            sigma = sigma_t_given_s * sigma_s / sigma_t
            # linker_mask = torch.nonzero(data.bridge_mask.reshape(-1)).squeeze()
            mu1 = ((z_t / alpha_t_given_s).squeeze() - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t_hat.squeeze()).squeeze()
            z_t = torch.normal(mean=mu1, std=sigma).squeeze()


        # x, h = self.sample_p_xh_given_z0_only_linker()
        gamma_0 = gamma_s
        sigma_x = self.SNR(-0.5 * gamma_0)
        x = torch.cat((data.x, s.squeeze().type(data.x.dtype).expand(data.x.shape[0]).unsqueeze(dim=1)), dim=1)
        if self.edge_feat:
            hidden = self.base_gin(x, data.edge_index, data.edge_attr)
            hidden = self.base_gin2(hidden, data.edge_index, data.edge_attr)
            hidden = self.base_gin3(hidden, data.edge_index, data.edge_attr)
            # feat = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch, batch_size)
        else:
            hidden = self.base_gin(x, data.edge_index)
            hidden = self.base_gin2(hidden, data.edge_index)
            hidden = self.base_gin3(hidden, data.edge_index)
            # feat = self.encoder(data.x, data.edge_index, data.batch, batch_size)

        temp = hidden[bridge_idx.reshape(-1)]
        bridge_feat = torch.cat(
            (temp[:int(temp.shape[0] / 2)], temp[int(temp.shape[0] / 2):], self.upfill(z_t.unsqueeze(1))), 1)

        eps_t_hat = torch.sigmoid(self.bridge_predictor(bridge_feat))

        # Compute mu for p(z_s | z_t)
        # Sample z_s given the parameters derived from zt
        # Compute sigma for p(z_s | z_t)

        # mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0)
        sigma_t = self.sigma(gamma_t, target_tensor=t)
        alpha_t = self.alpha(gamma_t, target_tensor=t)
        # x_pred = 1. / alpha_t * (z_t - sigma_t * eps_t)
        sigma = sigma_t_given_s * sigma_s / sigma_t
        # linker_mask = torch.nonzero(data.bridge_mask.reshape(-1)).squeeze()
        mu1 = (1. / alpha_t * (z_t.squeeze() - sigma_t * eps_t_hat.squeeze()).squeeze()).squeeze()
        z_s = torch.normal(mean=mu1, std=sigma_x).squeeze()

        bridge_attr = None
        if self.edge_feat:
            # attr_pred = self.attr_predictor(torch.cat((hidden[data.edge_index[0]], hidden[data.edge_index[1]]), 1))
            # edge_attr_loss = l1_loss(attr_pred, edge_attr)
            bridge_attr = self.attr_predictor(torch.cat((temp[:int(temp.shape[0] / 2)], temp[int(temp.shape[0] / 2):]), 1))

        out_readout = self.readout(hidden, data.batch, data.batch[-1] + 1)
        num_pred = self.num_predictor(out_readout)

        return z_s, bridge_attr, num_pred

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -self.expm1(self.softplus(gamma_s) - self.softplus(gamma_t)),
            target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,),
        or possibly more empty axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    @staticmethod
    def expm1(x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)


class Edge_idx_MLP(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):

        super(Edge_idx_MLP, self).__init__()
        # self.classifier = nn.Sequential(*(
        #         [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_ffn), nn.BatchNorm1d(2 * config.model.dim_ffn)] +
        #         [nn.ReLU(), nn.Linear(2 * config.model.dim_ffn, config.dataset.num_classes)]
        # ))
        self.predictor = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden, 2)]
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


class Edge_attr_MLP(torch.nn.Module):
    r"""
    Applies a linear transformation to complete classification from representations.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):

        super(Edge_attr_MLP, self).__init__()
        # self.classifier = nn.Sequential(*(
        #         [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_ffn), nn.BatchNorm1d(2 * config.model.dim_ffn)] +
        #         [nn.ReLU(), nn.Linear(2 * config.model.dim_ffn, config.dataset.num_classes)]
        # ))
        self.predictor = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden, 3)]
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


class vGINFeatExtractor(GNNBasic):
    r"""
        vGIN feature extractor using the :class:`~vGINEncoder` or :class:`~vGINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
            **kwargs: `without_readout` will output node features instead of graph features.
    """
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(vGINFeatExtractor, self).__init__(config)
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            self.encoder = vGINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = vGINEncoder(config, **kwargs)
            self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        vGIN feature extractor using the :class:`~vGINEncoder` or :class:`~vGINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, **kwargs)
            out_readout = self.encoder(x, edge_index, edge_attr, batch, batch_size)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            out_readout = self.encoder(x, edge_index, batch, batch_size)
        return out_readout


# @register.model_register
# class DANN_GIN(GNNBasic):
#     r"""
#     The Graph Neural Network modified from the `"Domain-Adversarial Training of Neural Networks"
#     <https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper and `"How Powerful are Graph Neural
#     Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.
#
#     Args:
#         config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.num_envs`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
#     """
#
#     def __init__(self, config: Union[CommonArgs, Munch]):
#         super().__init__(config)
#         self.encoder = GINFeatExtractor(config)
#         self.classifier = Classifier(config)
#
#         self.dc = nn.Linear(config.model.dim_hidden, config.dataset.num_envs)
#
#         self.dropout = nn.Dropout(config.model.dropout_rate)
#         self.graph_repr = None
#         self.config = config
#
#     def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
#         r"""
#         The DANN-GIN model implementation.
#
#         Args:
#             *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
#             **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
#
#         Returns (Tensor):
#             [label predictions, domain predictions]
#
#         """
#         out_readout = self.encoder(*args, **kwargs)
#         self.graph_repr = out_readout
#
#         dc_out = GradientReverseLayerF.apply(out_readout, self.config.train.alpha)
#         dc_out = self.dc(dc_out)
#
#         out = self.classifier(out_readout)
#         return out, dc_out




class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""

    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        positive_weight = F.softplus(self.weight)
        return F.linear(x, positive_weight, self.bias)