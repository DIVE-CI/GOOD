import copy
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from skimage.restoration import denoise_tv_chambolle
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree, to_dense_adj, dense_to_sparse

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed

class DummyDataset(InMemoryDataset):

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):
        super().__init__(root, transform, pre_transform)


@register.dataloader_register
class GMixupDataLoader(Munch):

    def __init__(self, *args, **kwargs):
        super(GMixupDataLoader, self).__init__(*args, **kwargs)

    @classmethod
    def setup(cls, dataset, config: Union[CommonArgs, Munch]):
        r"""
            Create a PyG data loader.

            Args:
                dataset: A GOOD dataset.
                config: Required configs:
                    ``config.train.train_bs``
                    ``config.train.val_bs``
                    ``config.train.test_bs``
                    ``config.model.model_layer``
                    ``config.train.num_steps(for node prediction)``

            Returns:
                A PyG dataset loader.

            """
        reset_random_seed(config)
        aug_path = os.path.join(config.dataset.dataset_root, config.dataset.dataset_name, config.dataset.domain,
                                'processed', f'aug_data_params_{config.ood.ood_param}_{config.ood.extra_param}_seed_{config.random_seed}.pt')
        aug_train_set = DummyDataset('dummy_root', 'dummy_name')
        if os.path.exists(aug_path):
            aug_train_set.data, aug_train_set.slices = torch.load(aug_path)
        else:
            aug_train_set.data, aug_train_set.slices = aug_train_set.collate(g_mixup_generate(dataset['train'], config))
            torch.save((aug_train_set.data, aug_train_set.slices), aug_path)
        for key in dataset.keys():
            if key in ['train', 'id_val', 'id_test', 'val', 'test']:
                if dataset[key] is None:
                    continue
                dataset[key].data.x = dataset[key].data.x.float()

        loader = {'train': DataLoader(aug_train_set, batch_size=config.train.train_bs, shuffle=True),
                  'eval_train': DataLoader(dataset['train'], batch_size=config.train.val_bs, shuffle=False),
                  'id_val': DataLoader(dataset['id_val'], batch_size=config.train.val_bs, shuffle=False) if dataset.get(
                      'id_val') else None,
                  'id_test': DataLoader(dataset['id_test'], batch_size=config.train.test_bs,
                                        shuffle=False) if dataset.get(
                      'id_test') else None,
                  'val': DataLoader(dataset['val'], batch_size=config.train.val_bs, shuffle=False),
                  'test': DataLoader(dataset['test'], batch_size=config.train.test_bs, shuffle=False)}

        return cls(loader)


def g_mixup_generate(orig_train_set, config: Union[CommonArgs, Munch]):
    orig_train_set = list(orig_train_set)
    # --- definitions ---
    train_nums = len(orig_train_set)
    aug_ratio = config.ood.ood_param
    aug_num = config.ood.extra_param[0]
    lam_range = config.ood.extra_param[1]
    has_node_features = hasattr(orig_train_set[0], 'x') and getattr(orig_train_set[0], 'x') is not None

    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
        orig_train_set)
    print(f"#IN#median num nodes of training graphs: {median_num_nodes}")

    resolution = int(median_num_nodes)

    class_graphs = split_class_graphs(orig_train_set, has_node_features)
    graphons = []

    if has_node_features:
        graphons = []
        for label, graphs, node_x in class_graphs:
            print(f"#IN#label: {label}, num_graphs: {len(graphs)}, node_x shape: {node_x[0].shape}")
            align_graphs_list, aligned_node_x, normalized_node_degrees, max_num, min_num = align_x_graphs(
                graphs, node_x, padding=True, N=resolution)
            print(f"#IN#aligned graph {align_graphs_list[0].shape}, node feature {aligned_node_x.shape}")

            graphon = universal_svd(align_graphs_list, threshold=0.2)
            graphons.append((label, graphon, aligned_node_x))

        for label, graphon, node_x in graphons:
            print(f"#IN#graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}")

    else:
        for label, graphs in class_graphs:
            print(f"#IN#label: {label}, num_graphs:{len(graphs)}")
            align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
                graphs, padding=True, N=resolution)
            print(f"#IN#aligned graph {align_graphs_list[0].shape}")

            graphon = universal_svd(align_graphs_list, threshold=0.2)
            graphons.append((label, graphon))

        for label, graphon in graphons:
            print(f"#IN#graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}")

    num_sample = int(train_nums * aug_ratio / aug_num)
    lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(aug_num,))

    reset_random_seed(config)
    new_graph = []
    for lam in lam_list:
        print(f"#IN#lam: {lam}")
        print(f"#IN#num_sample: {num_sample}")
        two_graphons = random.sample(graphons, 2)
        new_graph += two_graphons_mixup(two_graphons, la=lam, num_sample=num_sample, has_node_features=has_node_features)
        print(f"#IN#label: {new_graph[-1].y}")

    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
        new_graph)
    print(f"#IN#avg num nodes of new graphs: {avg_num_nodes}")
    print(f"#IN#avg num edges of new graphs: {avg_num_edges}")
    print(f"#IN#avg density of new graphs: {avg_density}")
    print(f"#IN#median num nodes of new graphs: {median_num_nodes}")
    print(f"#IN#median num edges of new graphs: {median_num_edges}")
    print(f"#IN#median density of new graphs: {median_density}")

    train_set = new_graph + orig_train_set

    # dataset = new_graph + dataset
    print(f"#IN#real aug ratio: {len(new_graph) / train_nums}")

    train_set = prepare_dataset(train_set)
    return train_set


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        # print( data.x.shape )
        return data


def prepare_synthetic_dataset(dataset):
    max_degree = 0
    degs = []
    for data in dataset:
        degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, degs[-1].max().item())

    for data in dataset:
        degs = degree(data.edge_index[0], dtype=torch.long)

        data.x = F.one_hot(degs.to(torch.int64), num_classes=max_degree + 1).to(torch.float)
        print(data.x.shape)

    return dataset


def prepare_dataset(dataset, transform=lambda x: x):
    for data in dataset:
        if ('y1' not in data):
            data.y1 = data.y
            data.y2 = data.y
            data.lam = 1.0
        if ('y' not in data):
            data.y = data.y1
        if ('x' not in data):
            data = transform(data)
        if ('num_nodes' not in data):
            data['num_nodes'] = data.x.shape[0]
    return dataset


def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()


def align_graphs(graphs: List[np.ndarray],
                 padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees
    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        max_num = max(max_num, N)

        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)
        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

    return aligned_graphs, normalized_node_degrees, max_num, min_num


def align_x_graphs(graphs: List[np.ndarray], node_x: List[np.ndarray], padding: bool = False, N: int = None) -> Tuple[
    List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees
    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    max_num = max(max_num, N)
    min_num = min(num_nodes)

    aligned_node_x = np.zeros((max_num, node_x[0].shape[1]))
    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        # print(node_x)
        # print(node_x.shape)
        sorted_node_x = node_x[i][idx, :]
        # if max_num < N:
        #     max_num = max(max_num, N)
        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)

            # added
            aligned_node_x[:num_i, :] += sorted_node_x


        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

            # added
    if N:
        aligned_node_x = aligned_node_x[:N]

    aligned_node_x = aligned_node_x / len(graphs)
    return aligned_graphs, aligned_node_x, normalized_node_degrees, max_num, min_num


def two_graphons_mixup(two_graphons, la=0.5, num_sample=20, has_node_features=False):
    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]

    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    if has_node_features:
        new_x = la * two_graphons[0][2] + (1 - la) * two_graphons[1][2]
        sample_graph_x = torch.from_numpy(new_x).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):
        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)
        if A.shape[0] == 0:
            continue
        num_nodes = int(torch.max(edge_index)) + 1
        if has_node_features:
            pyg_graph = Data(x=sample_graph_x[:num_nodes], edge_index=edge_index, y1=torch.tensor(two_graphons[0][0]),
                             y2=torch.tensor(two_graphons[1][0]), lam=la, num_nodes=num_nodes, y=sample_graph_label)
        else:
            pyg_graph = Data(edge_index=edge_index, y1=torch.tensor(two_graphons[0][0]),
                             y2=torch.tensor(two_graphons[1][0]), lam=la, num_nodes=num_nodes, y=sample_graph_label)

        sample_graphs.append(pyg_graph)

    return sample_graphs


# def two_x_graphons_mixup(two_x_graphons, la=0.5, num_sample=20):
#     label = la * two_x_graphons[0][0] + (1 - la) * two_x_graphons[1][0]
#     new_graphon = la * two_x_graphons[0][1] + (1 - la) * two_x_graphons[1][1]
#     new_x = la * two_x_graphons[0][2] + (1 - la) * two_x_graphons[1][2]
#
#     sample_graph_label = torch.from_numpy(label).type(torch.float32)
#     sample_graph_x = torch.from_numpy(new_x).type(torch.float32)
#     # print(new_graphon)
#
#     sample_graphs = []
#     for i in range(num_sample):
#         sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
#         sample_graph = np.triu(sample_graph)
#         sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))
#
#         sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
#         sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]
#
#         A = torch.from_numpy(sample_graph)
#         edge_index, _ = dense_to_sparse(A)
#
#         num_nodes = int(torch.max(edge_index)) + 1
#
#         pyg_graph = Data(x=sample_graph_x[:num_nodes], edge_index=edge_index, y1=torch.tensor(two_x_graphons[0][0]),
#                          y2=torch.tensor(two_x_graphons[1][0]), lam=la, num_nodes=num_nodes, y=sample_graph_label)
#         # pyg_graph = Data()
#         # pyg_graph.y = sample_graph_label
#         # pyg_graph.x = sample_graph_x[:num_nodes]
#         # pyg_graph.edge_index = edge_index
#         # pyg_graph.num_nodes = num_nodes
#         sample_graphs.append(pyg_graph)
#
#         # print(edge_index)
#     return sample_graphs


def graphon_mixup(dataset, la=0.5, num_sample=20):
    graphons = estimate_graphon(dataset, universal_svd)

    two_graphons = random.sample(graphons, 2)
    # for label, graphon in two_graphons:
    #     print( label, graphon )
    # print(two_graphons[0][0])

    label = la * two_graphons[0][0] + (1 - la) * two_graphons[1][0]
    new_graphon = la * two_graphons[0][1] + (1 - la) * two_graphons[1][1]

    print("new label:", label)
    # print("new graphon:", new_graphon)

    # print( label )
    sample_graph_label = torch.from_numpy(label).type(torch.float32)
    # print(new_graphon)

    sample_graphs = []
    for i in range(num_sample):
        sample_graph = (np.random.rand(*new_graphon.shape) < new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]

        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        # print(sample_graph.shape)

        # print(sample_graph)

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes

        sample_graphs.append(pyg_graph)
        # print(edge_index)
    return sample_graphs


def estimate_graphon(dataset, graphon_estimator):
    y_list = []
    for data in dataset:
        y_list.append(tuple(data.y.tolist()))
        # print(y_list)
    num_classes = len(set(y_list))

    all_graphs_list = []
    for graph in dataset:
        adj = to_dense_adj(graph.edge_index)[0].numpy()
        all_graphs_list.append(adj)

    # print(len(all_graphs_list))

    graphons = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]

        aligned_adj_list, normalized_node_degrees, max_num, min_num = align_graphs(c_graph_list, padding=True, N=400)

        graphon_c = graphon_estimator(aligned_adj_list, threshold=0.2)

        graphons.append((np.array(class_label), graphon_c))

    return graphons


def estimate_one_graphon(aligned_adj_list: List[np.ndarray], method="universal_svd"):
    if method == "universal_svd":
        graphon = universal_svd(aligned_adj_list, threshold=0.2)
    else:
        graphon = universal_svd(aligned_adj_list, threshold=0.2)

    return graphon


# def split_class_x_graphs(dataset):
#     y_list = []
#     for data in dataset:
#         y_list.append(tuple(data.y.tolist()))
#         # print(y_list)
#     num_classes = len(set(y_list))
#
#     all_graphs_list = []
#     all_node_x_list = []
#     for graph in dataset:
#         adj = to_dense_adj(graph.edge_index)[0].numpy()
#         all_graphs_list.append(adj)
#         all_node_x_list.append(graph.x.numpy())
#
#     class_graphs = []
#     for class_label in set(y_list):
#         c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
#         c_node_x_list = [all_node_x_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
#         class_graphs.append((np.array(class_label), c_graph_list, c_node_x_list))
#
#     return class_graphs


def split_class_graphs(dataset, has_node_features=False):
    if has_node_features:
        y_list = []
        unique_y_list = []
        all_graphs_list = []
        all_node_x_list = []
        for graph in dataset:
            if graph.edge_index.shape[1] == 0:
                continue
            adj = to_dense_adj(graph.edge_index)[0].numpy()
            all_graphs_list.append(adj)
            all_node_x_list.append(graph.x.numpy())

            y = graph.y
            if y not in unique_y_list:
                unique_y_list.append(y)
            y_list.append(y)
            # y_list.append(tuple(data.y.tolist()))


        class_graphs = []
        for class_label in unique_y_list:
            c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
            c_node_x_list = [all_node_x_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
            class_graphs.append((np.array(class_label), c_graph_list, c_node_x_list))

        return class_graphs
    else:
        y_list = []
        unique_y_list = []
        all_graphs_list = []
        for graph in dataset:
            if graph.edge_index.shape[1] == 0:
                continue
            adj = to_dense_adj(graph.edge_index)[0].numpy()
            all_graphs_list.append(adj)
            y = graph.y
            if y not in unique_y_list:
                unique_y_list.append(y)
            y_list.append(y)
            # y_list.append(tuple(data.y.tolist()))
            # print(y_list)


        class_graphs = []
        for class_label in unique_y_list:
            c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
            class_graphs.append((np.array(class_label), c_graph_list))

        return class_graphs


def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.
    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon


def sorted_smooth(aligned_graphs: List[np.ndarray], h: int) -> np.ndarray:
    """
    Estimate a graphon by a sorting and smoothing method
    Reference:
    S. H. Chan and E. M. Airoldi,
    "A Consistent Histogram Estimator for Exchangeable Graph Models",
    Proceedings of International Conference on Machine Learning, 2014.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param h: the block size
    :return: a (k, k) step function and  a (r, r) estimation of graphon
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0, keepdim=True).unsqueeze(0)
    else:
        sum_graph = aligned_graphs.unsqueeze(0)  # (1, 1, N, N)

    # histogram of graph
    kernel = torch.ones(1, 1, h, h) / (h ** 2)
    # print(sum_graph.size(), kernel.size())
    graphon = torch.nn.functional.conv2d(sum_graph, kernel, padding=0, stride=h, bias=None)
    graphon = graphon[0, 0, :, :].numpy()
    # total variation denoising
    graphon = denoise_tv_chambolle(graphon, weight=h)
    return graphon


def stat_graph(graphs_list: List[Data]):
    num_total_nodes = []
    num_total_edges = []
    for graph in graphs_list:
        num_total_nodes.append(graph.num_nodes)
        num_total_edges.append(graph.edge_index.shape[1])
    avg_num_nodes = sum(num_total_nodes) / len(graphs_list)
    avg_num_edges = sum(num_total_edges) / len(graphs_list) / 2.0
    avg_density = avg_num_edges / (avg_num_nodes * avg_num_nodes)

    median_num_nodes = np.median(num_total_nodes)
    median_num_edges = np.median(num_total_edges)
    median_density = median_num_edges / (median_num_nodes * median_num_nodes)

    return avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density


class onehot_y(BaseTransform):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data):
        data.y = F.one_hot(data.y, num_classes=self.num_classes).to(torch.float)[0]

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}'
