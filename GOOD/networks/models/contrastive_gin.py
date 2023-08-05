import torch
import torch.nn as nn
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from torch import Tensor
from .BaseGNN import GNNBasic

import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from transformers import BertModel, BertConfig
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

@register.model_register
class GINSimclr(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
    # def __init__(
    #         self,
    #         temperature,
    #         gin_hidden_dim,
    #         gin_num_layers,
    #         drop_ratio,
    #         graph_pooling,
    #         bert_hidden_dim,
    #         bert_pretrain,
    #         projection_dim,
    #         lr,
    #         weight_decay,
    # ):
    #     super().__init__()
    #     self.save_hyperparameters()
        self.graph_classifier = Classifier(config)
        self.text_classifier = Classifier(config)

        self.temperature = 0.1
        self.gin_hidden_dim = 300
        self.gin_num_layers = 5
        self.drop_ratio = 0.0
        self.graph_pooling = 'sum'

        self.bert_hidden_dim = 768
        self.bert_pretrain = 0  # 0: sci-bert

        self.projection_dim = 256

        self.lr = 0.0001
        self.weight_decay = 1e-5

        self.graph_encoder = GNN(
            num_layer=self.gin_num_layers,
            emb_dim=self.gin_hidden_dim,
            gnn_type='gin',
            # virtual_node=True,
            # residual=False,
            drop_ratio=self.drop_ratio,
            JK='last',
            # graph_pooling=self.graph_pooling,
        )
        # print(self.graph_encoder.state_dict().keys())
        # ckpt = torch.load('/data/xinerli/pretrain-gnns/chem/model_gin/supervised_contextpred.pth')
        # print(ckpt.keys())
        # missing_keys, unexpected_keys = self.graph_encoder.load_state_dict(ckpt, strict=False)
        # print(missing_keys)
        # print(unexpected_keys)

        # Text Encoder
        if self.bert_pretrain:
            self.text_encoder = TextEncoder(pretrained=False)
        else:
            self.text_encoder = TextEncoder(pretrained=True)
        
        # Smiles Encoder (same as text encoder)
        if self.bert_pretrain:
            self.smiles_encoder = TextEncoder(pretrained=False)
        else:
            self.smiles_encoder = TextEncoder(pretrained=True)
            
        # if self.bert_pretrain:
        #     print("bert load kvplm")
        #     ckpt = torch.load('kvplm_pretrained/ckpt_KV_1.pt')
        #     if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in ckpt:
        #         pretrained_dict = {"main_model."+k[20:]: v for k, v in ckpt.items()}
        #     elif 'bert.embeddings.word_embeddings.weight' in ckpt:
        #         pretrained_dict = {"main_model."+k[5:]: v for k, v in ckpt.items()}
        #     else:
        #         pretrained_dict = {"main_model."+k[12:]: v for k, v in ckpt.items()}
            # print(pretrained_dict.keys())
            # print(self.text_encoder.state_dict().keys())
            # self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            # self.smiles_encoder.load_state_dict(pretrained_dict, strict=False)
            # missing_keys, unexpected_keys = self.text_encoder.load_state_dict(pretrained_dict, strict=False)
            # print(missing_keys)
            # print(unexpected_keys)
        # self.feature_extractor.freeze()


        self.graph_proj_head = nn.Sequential(
          nn.Linear(self.gin_hidden_dim, self.gin_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.gin_hidden_dim, self.projection_dim)
        )
        self.text_proj_head = nn.Sequential(
          nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )
        self.smiles_proj_head = nn.Sequential(
          nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )

    def forward(self, *args, **kwargs):

        data = kwargs.get('data')
        assert data is not None

        graph_rep = self.graph_encoder(data)
        graph_rep = self.graph_proj_head(graph_rep)

        smiles_rep = self.smiles_encoder(data.smiles_tokens, data.smiles_mask)
        smiles_rep = self.smiles_proj_head(smiles_rep)

        graph_out = self.graph_classifier(graph_rep)
        text_out = self.text_classifier(smiles_rep)

        # text1_rep = self.text_encoder(text1, mask1)
        # text1_rep = self.text_proj_head(text1_rep)
        #
        # text2_rep = self.text_encoder(text2, mask2)
        # text2_rep = self.text_proj_head(text2_rep)
        return graph_out, text_out

    # def forward(self, features_graph, features_text):
    #     batch_size = features_graph.size(0)
    #
    #     # normalized features
    #     features_graph = F.normalize(features_graph, dim=-1)
    #     features_text = F.normalize(features_text, dim=-1)
    #
    #     # cosine similarity as logits
    #     logits_per_graph = features_graph @ features_text.t() / self.temperature
    #     logits_per_text = logits_per_graph.t()
    #
    #     labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
    #     loss_graph = F.cross_entropy(logits_per_graph, labels)
    #     loss_text = F.cross_entropy(logits_per_text, labels)
    #     loss = (loss_graph + loss_text) / 2
    #
    #     return logits_per_graph, logits_per_text, loss

    # def configure_optimizers(self):
    #     # High lr because of small dataset and small model
    #     optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     return optimizer
    #
    # def training_step(self, batch, batch_idx):
    #     graph, smiles, mask, text1, mask1, text2, mask2 = batch
    #
    #     graph_rep = self.graph_encoder(graph)
    #     graph_rep = self.graph_proj_head(graph_rep)
    #
    #     smiles_rep = self.smiles_encoder(smiles, mask)
    #     smiles_rep = self.smiles_proj_head(smiles_rep)
    #
    #     text1_rep = self.text_encoder(text1, mask1)
    #     text1_rep = self.text_proj_head(text1_rep)
    #
    #     text2_rep = self.text_encoder(text2, mask2)
    #     text2_rep = self.text_proj_head(text2_rep)
    #
    #     _, _, loss1 = self.forward(graph_rep, text1_rep)
    #     _, _, loss2 = self.forward(graph_rep, text2_rep)
    #     _, _, loss3 = self.forward(graph_rep, smiles_rep)
    #
    #     loss = (loss1 + loss2 + loss3) / 3.0
    #
    #     self.log("train_loss", loss)
    #     return loss

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("GINSimclr")
    #     # train mode
    #     parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
    #     # GIN
    #     parser.add_argument('--gin_hidden_dim', type=int, default=300)
    #     parser.add_argument('--gin_num_layers', type=int, default=5)
    #     parser.add_argument('--drop_ratio', type=float, default=0.0)
    #     parser.add_argument('--graph_pooling', type=str, default='sum')
    #     # Bert
    #     parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    #     parser.add_argument('--bert_pretrain', action='store_false', default=True)
    #     parser.add_argument('--projection_dim', type=int, default=256)
    #     # optimization
    #     parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
    #     parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
    #     return parent_parser


class TextEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(TextEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model
            self.main_model = BertModel.from_pretrained('/data/xinerli/MoMu/Pretrain/bert_pretrained/')
        else:
            config = BertConfig(vocab_size=31090, )
            self.main_model = BertModel(config)

        self.dropout = nn.Dropout(0.1)
        # self.hidden_size = self.main_model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        device = input_ids.device
        typ = torch.zeros(input_ids.shape).long().to(device)
        output = self.main_model(input_ids, token_type_ids=typ, attention_mask=attention_mask)['pooler_output']  # b,d
        logits = self.dropout(output)
        return logits


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
            [nn.Linear(256, config.dataset.num_classes)]
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


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        self.pool = global_mean_pool

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        h_graph = self.pool(node_representation, batch)

        return h_graph


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__(aggr="add")
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        # print('--------------------')
        # print('x:', x.shape)
        # print('edge_index:',edge_index.shape)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # self_loop_attr[:,0] = 4 #bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # print('edge_attr:',edge_attr.shape)
        # print('self_loop_attr:',self_loop_attr.shape)
        # print('--------------------')
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)