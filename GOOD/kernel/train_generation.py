r"""Training pipeline: training/evaluation structure, batch training.
"""
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from GOOD.kernel.evaluation import evaluate
# from GOOD.kernel.train import train
from GOOD.networks.model_manager import config_model
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.logger import pbar_setting
from GOOD.utils.train import nan2zero_get_mask
import os


def train_batch_generation(model: torch.nn.Module, data: Batch, ood_algorithm: BaseOODAlg, pbar,
                config: Union[CommonArgs, Munch]) -> dict:
    r"""
    Train a batch. (Project use only)

    Args:
        model (torch.nn.Module): The GNN model.
        data (Batch): Current batch of data.
        ood_algorithm (BaseOODAlg: The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    Returns:
        Calculated loss.
    """
    data = data.to(config.device)

    mask, targets = nan2zero_get_mask(data, 'train', config)
    node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
    data, targets, mask, node_norm = ood_algorithm.input_preprocess(data, targets, mask, node_norm, model.training,
                                                                    config)
    edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None

    config.train_helper.optimizer.zero_grad()

    model_output = model(data=data, edge_weight=edge_weight, ood_algorithm=ood_algorithm)
    raw_pred = ood_algorithm.output_postprocess(model_output)

    loss = ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, config)
    loss = ood_algorithm.loss_postprocess(loss, data, mask, config)
    loss.backward()

    config.train_helper.optimizer.step()

    return {'loss': loss.detach()}


def train_generation(model: torch.nn.Module, model_main: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]], ood_algorithm_gen: BaseOODAlg, ood_algorithm: BaseOODAlg,
          config: Union[CommonArgs, Munch]):
    r"""
    Training pipeline. (Project use only)

    Args:
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """
    # config model
    print('#D#Config model')
    config_model(model, 'train', config)

    # Load training utils
    print('#D#Load training utils')
    config.train_helper.set_up(model, config)
    tik = time.time()

    # train the model
    for epoch in range(config.train.ctn_epoch, int(config.ood.extra_param[7])):  #
        config.train.epoch = epoch
        print(f'#IN#Epoch {epoch}:')

        mean_loss = 0
        spec_loss = 0

        # ood_algorithm.stage_control(config)

        # if config.dataset.dataset_name == 'GOODCora' and config.dataset.shift_type == 'concept':
        #     config.dataset.env_range = (loader['train'].data.word.max() - loader['train'].data.word.min())/10.0
        #     config.dataset.global_mean_pyx = loader['train'].data.word.min()
        # elif config.dataset.dataset_name == 'GOODArxiv' and config.dataset.shift_type == 'concept':
        #     config.dataset.global_mean_pyx = loader['train'].data.time.mean()

        pbar = tqdm(enumerate(loader['train']), total=len(loader['train']), **pbar_setting)
        for index, data in pbar:
            if data.batch is not None and (data.batch[-1] < config.train.train_bs - 1):
                continue

            # Parameter for DANN
            p = (index / len(loader['train']) + epoch) / config.train.max_epoch
            config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # train a batch
            train_stat = train_batch_generation(model, data, ood_algorithm_gen, pbar, config)
            # if index % 100 == 0 and config.dataset.dataset_name != 'GOODSST2':
            #     with torch.cuda.device(config.device):
            #         torch.cuda.empty_cache()
            mean_loss = (mean_loss * index + ood_algorithm_gen.mean_loss) / (index + 1)

            # if config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup', 'GraphMix', 'feat_aug']:
            if ood_algorithm_gen.spec_loss is not None:
                spec_loss = (spec_loss * index + ood_algorithm_gen.spec_loss) / (index + 1)
                pbar.set_description(f'M/S Loss: {mean_loss:.4f}/{spec_loss:.4f}')
            else:
                pbar.set_description(f'Loss: {mean_loss:.4f}')

        # Eval training score

        # Epoch val
        print('#IN#\nEvaluating...')
        # if config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup', 'GraphMix', 'feat_aug']:
        if ood_algorithm_gen.spec_loss is not None:
            print(f'#IN#Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}')
        else:
            print(f'#IN#Approximated average training loss {mean_loss.cpu().item():.4f}')

        epoch_train_stat = evaluate_gen(model, loader, ood_algorithm_gen, 'eval_train', config)
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()
        id_val_stat = evaluate_gen(model, loader, ood_algorithm_gen, 'id_val', config)
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()
        id_test_stat = evaluate_gen(model, loader, ood_algorithm_gen, 'id_test', config)
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()
        val_stat = evaluate_gen(model, loader, ood_algorithm_gen, 'val', config)
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()
        test_stat = evaluate_gen(model, loader, ood_algorithm_gen, 'test', config)
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()

        # checkpoints save
        config.train_helper.save_epoch_gen(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, config)

        # --- scheduler step ---
        config.train_helper.scheduler.step()

    with torch.cuda.device(config.device):
        torch.cuda.empty_cache()
    print('#IN#Generative model Training end.')
    print('#IM#Total generative training time: {:.4f}s'.format(time.time() - tik))
    gen_ckpt = torch.load(os.path.join(config.ckpt_dir, f'best_gen.ckpt'), map_location=config.device)
    model.load_state_dict(gen_ckpt['state_dict'])

    config.train.lr = config.ood.extra_param[6]
    # config.train.test_bs = 256
    train(model, model_main, loader, ood_algorithm, config)



def train_batch(gen_model: torch.nn.Module, model: torch.nn.Module, data: Batch, ood_algorithm: BaseOODAlg, pbar,
                config: Union[CommonArgs, Munch]) -> dict:
    r"""
    Train a batch. (Project use only)

    Args:
        model (torch.nn.Module): The GNN model.
        data (Batch): Current batch of data.
        ood_algorithm (BaseOODAlg: The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    Returns:
        Calculated loss.
    """
    data = data.to(config.device)

    mask, targets = nan2zero_get_mask(data, 'train', config)
    node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
    data, targets, mask, node_norm = ood_algorithm.input_generation(gen_model, data, targets, mask, node_norm, model.training,
                                                                    config)
    edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None

    config.train_helper.optimizer.zero_grad()

    model_output = model(data=data, edge_weight=edge_weight, ood_algorithm=ood_algorithm)
    raw_pred = ood_algorithm.output_postprocess(model_output)

    loss = ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, config)
    loss = ood_algorithm.loss_postprocess(loss, data, mask, config)
    loss.backward()

    config.train_helper.optimizer.step()

    return {'loss': loss.detach()}



def train(gen_model: torch.nn.Module, model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]], ood_algorithm: BaseOODAlg,
          config: Union[CommonArgs, Munch]):
    r"""
    Training pipeline. (Project use only)

    Args:
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """
    # config model
    print('#D#Config model')
    config_model(model, 'train', config)

    # Load training utils
    print('#D#Load training utils')
    config.train_helper.set_up(model, config)
    tik = time.time()

    config.metric.best_stat['score'] = None
    config.metric.id_best_stat['score'] = None

    # train the model
    for epoch in range(config.train.ctn_epoch, config.train.max_epoch):
        config.train.epoch = epoch
        print(f'#IN#Epoch {epoch}:')

        mean_loss = 0
        spec_loss = 0

        ood_algorithm.stage_control(config)

        if config.dataset.dataset_name == 'GOODCora' and config.dataset.shift_type == 'concept':
            config.dataset.env_range = (loader['train'].data.word.max() - loader['train'].data.word.min())/10.0
            config.dataset.global_mean_pyx = loader['train'].data.word.min()
        elif config.dataset.dataset_name == 'GOODArxiv' and config.dataset.shift_type == 'concept':
            config.dataset.global_mean_pyx = loader['train'].data.time.mean()

        pbar = tqdm(enumerate(loader['train']), total=len(loader['train']), **pbar_setting)
        for index, data in pbar:
            if data.batch is not None and (data.batch[-1] < config.train.train_bs - 1):
                continue

            # Parameter for DANN
            p = (index / len(loader['train']) + epoch) / config.train.max_epoch
            config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # train a batch
            train_stat = train_batch(gen_model, model, data, ood_algorithm, pbar, config)
            # with torch.cuda.device(config.device):
            #     torch.cuda.empty_cache()
            mean_loss = (mean_loss * index + ood_algorithm.mean_loss) / (index + 1)

            # if config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup', 'GraphMix', 'feat_aug']:
            if ood_algorithm.spec_loss is not None:
                spec_loss = (spec_loss * index + ood_algorithm.spec_loss) / (index + 1)
                pbar.set_description(f'M/S Loss: {mean_loss:.4f}/{spec_loss:.4f}')
            else:
                pbar.set_description(f'Loss: {mean_loss:.4f}')

        # Eval training score
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()

        # Epoch val
        print('#IN#\nEvaluating...')
        # if config.ood.ood_alg not in ['ERM', 'GroupDRO', 'Mixup', 'GraphMix', 'feat_aug']:
        if ood_algorithm.spec_loss is not None:
            print(f'#IN#Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}')
        else:
            print(f'#IN#Approximated average training loss {mean_loss.cpu().item():.4f}')

        epoch_train_stat = evaluate(model, loader, ood_algorithm, 'eval_train', config)
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()
        id_val_stat = evaluate(model, loader, ood_algorithm, 'id_val', config)
        id_test_stat = evaluate(model, loader, ood_algorithm, 'id_test', config)
        val_stat = evaluate(model, loader, ood_algorithm, 'val', config)
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()
        test_stat = evaluate(model, loader, ood_algorithm, 'test', config)
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()

        # checkpoints save
        config.train_helper.save_epoch(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, config)

        # --- scheduler step ---
        config.train_helper.scheduler.step()

    print('#IM#Total training time: {:.4f}s'.format(time.time() - tik))
    print('#IN#Training end.')




@torch.no_grad()
def evaluate_gen(model: torch.nn.Module,
             loader: Dict[str, DataLoader],
             ood_algorithm: BaseOODAlg,
             split: str,
             config: Union[CommonArgs, Munch]
             ):
    r"""
    This function is design to collect data results and calculate scores and loss given a dataset subset.
    (For project use only)

    Args:
        model (torch.nn.Module): The GNN model.
        loader (Dict[str, DataLoader]): A DataLoader dictionary that use ``split`` as key and Dataloaders as values.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
            'val', and 'test'.
        config (Union[CommonArgs, Munch]): Required configs are ``config.device`` (torch.device),
            ``config.model.model_level``, ``config.metric`` (:class:`GOOD.utils.metric.Metric`),
            ``config.ood`` (:class:`GOOD.utils.args.OODArgs`). Refer to :ref:`configs:Config file` for more details.

    Returns:
        A score and a loss.

    """
    stat = {'score': None, 'loss': None}
    if loader.get(split) is None:
        return stat
    model.eval()

    loss_all = []
    mask_all = []
    pred_all = []
    target_all = []
    pbar = tqdm(loader[split], desc=f'Eval {split.capitalize()}', total=len(loader[split]), **pbar_setting)
    for index, data in enumerate(pbar):
        # if index % 2 == 0:
        # with torch.cuda.device(config.device):
        #     torch.cuda.empty_cache()
        data: Batch = data.to(config.device)

        mask, targets = nan2zero_get_mask(data, split, config)
        if mask is None:
            return stat
        node_norm = torch.ones((data.num_nodes,), device=config.device) if config.model.model_level == 'node' else None
        data, targets, mask, node_norm = ood_algorithm.input_preprocess(data, targets, mask, node_norm, model.training,
                                                                        config)
        model_output = model(data=data, edge_weight=None, ood_algorithm=ood_algorithm)
        raw_preds = ood_algorithm.output_postprocess(model_output)

        loss = ood_algorithm.loss_calculate(raw_preds, targets, mask, node_norm, config)
        loss = ood_algorithm.loss_postprocess(loss, data, mask, config)

        # --------------- Loss collection ------------------
        # loss: torch.tensor = config.metric.loss_func(raw_preds, targets, reduction='none') * mask
        # mask_all.append(mask)
        loss_all.append(loss)

        # ------------- Score data collection ------------------
        # pred, target = eval_data_preprocess(targets, raw_preds, mask, config)
        # pred_all.append(pred)
        # target_all.append(target)

    # ------- Loss calculate -------
    # loss_all = torch.cat(loss_all)
    # mask_all = torch.cat(mask_all)
    stat['loss'] = sum(loss_all)/len(loss_all)  # loss_all.sum() / loss_all.shape[0]
    stat['score'] = stat['loss']

    # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
    # stat['score'] = eval_score(pred_all, target_all, config)

    print(f'#IN#\n{split.capitalize()} \n'
          f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

    model.train()

    return {'score': stat['score'], 'loss': stat['loss']}