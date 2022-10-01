import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from torch_geometric.data import Batch

from GOOD import config_summoner, args_parser
from GOOD.data import load_dataset
from GOOD.definitions import STORAGE_DIR, ROOT_DIR
from GOOD.utils.graph_visualize import plot_calculation_graph
from GOOD.utils.initial import reset_random_seed


# from PIL import Image
import pandas as pd

allowed_datasets = ['GOODMotif', 'GOODSST2']
allowed_methods = ['ERM', 'IRM', 'VREx', 'DANN', 'Coral', 'LISA', 'GrouDRO', 'DIR', 'DropNode','DropEdge','MaskFeat', 'GMixup', 'Mixup', 'SODAug'] #['ASAP', 'DIR', 'GSAT', 'CIGA', 'GEI']
config_paths = []
config_root = Path(ROOT_DIR, 'configs', 'final_configs')
# for dataset_path in config_root.iterdir():
#     if not dataset_path.is_dir():
#         continue
#     if dataset_path.name not in allowed_datasets:
#         continue
#     # single_dataset_paths = []
#     for domain_path in dataset_path.iterdir():
#         if not domain_path.is_dir() or domain_path.name == 'size':
#             continue
#         for shift_path in domain_path.iterdir():
#             if not shift_path.is_dir():
#                 continue
#             if shift_path.name != 'covariate':
#                 continue
#             for ood_config_path in shift_path.iterdir():
#                 if ood_config_path.stem in allowed_methods:
#                     # single_dataset_paths.append(str(ood_config_path))
#                     config_paths.append(str(ood_config_path))



plt.grid(False)

from tqdm import tqdm
from GOOD.utils.logger import pbar_setting

method_curves = dict()

# pbar = tqdm(enumerate(config_paths), total=len(config_paths), **pbar_setting)
# for i, config_path in pbar:
#     if ('CIGA' in config_path or 'DIR' in config_path) and 'Motif' in config_path:
#         args = args_parser(['--config_path', config_path])
#     else:
#         args = args_parser(['--config_path', config_path, '--exp_round', '1'])
#     config = config_summoner(args)
#     with open(config.log_path, 'r') as f:
#         file_content = f.readlines()
#         filter_content = [line for line in file_content if ('Epoch' in line or 'Test' in line) and ('ID' not in line and 'OOD' not in line)]
#         for i in range(len(filter_content)-1, -1, -1):
#             if 'Epoch 0:' in filter_content[i]:
#                 last_run_content = filter_content[i:]
#                 break
#         else:
#             raise Exception("Didn't find the start of the run.")
#         assert len(last_run_content) == 600 or 300
#         acc = []
#         loss = []
#         for line in last_run_content:
#             if 'ACC' in line:
#                 acc.append(float(line.split(':')[-1].strip()))
#             # elif 'Loss' in line:
#             #     loss.append(float(line.split(':')[-1].strip()))
#         assert len(acc) == 200 or 100
#
    # if method_curves.get(config.ood.ood_alg) is None:
    #     method_curves[config.ood.ood_alg] = [None, None]
#     if 'Motif' in config_path:
#         method_curves[config.ood.ood_alg][0] = np.array(acc)
#     elif 'SST2' in config_path:
#         method_curves[config.ood.ood_alg][1] = np.array(acc)

allowed_methods = ['ERM', 'IRM', 'VREx', 'DANN', 'Coral', 'LISA', 'GroupDRO','EERM','SRGNN', 'FLAG', 'DropEdge',
                       'MaskFeat', 'Mixup', 'FeatX']  # ['ASAP', 'DIR', 'GSAT', 'CIGA', 'GEI']

for method in allowed_methods:
    with open('plot_cbas/'+method+'.log', 'r') as f:
        file_content = f.readlines()
        filter_content = [line for line in file_content if ('Epoch' in line or 'Test' in line) and ('ID' not in line and 'OOD' not in line)]
        for i in range(len(filter_content)-1, -1, -1):
            if 'Epoch 0:' in filter_content[i]:
                last_run_content = filter_content[i:]
                break
        else:
            raise Exception("Didn't find the start of the run.")
        assert len(last_run_content) == 600 or 300
        acc = []
        loss = []
        for line in last_run_content:
            if 'ACC' in line:
                acc.append(float(line.split(':')[-1].strip()))
            elif 'Loss' in line:
                loss.append(float(line.split(':')[-1].strip()))
        # assert len(acc) == 200 or 100

    if method_curves.get(method) is None:
        method_curves[method] = [None, None]
    method_curves[method][0] = np.array(acc)
    method_curves[method][1] = np.array(loss)






plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
method_order = ['ERM', 'IRM', 'VREx', 'DANN', 'Coral', 'ASAP', 'DIR', 'GSAT', 'CIGA', 'GEI']
fig, axes = plt.subplots(1, 2, dpi=300, figsize=(12, 4.8))

for i, method in enumerate(allowed_methods):
    acc1, acc2 = method_curves[method]
    method = 'LECI' if method == 'GEI' else method
    x1 = np.arange(len(acc1))
    x2 = np.arange(len(acc2))
    pd_acc1, pd_acc2 = pd.DataFrame(acc1), pd.DataFrame(acc2)
    linestyle = '--' if i < 5 else '-'
    axes[0].plot(x1, pd_acc1[0].rolling(20).mean(), label=method, linestyle=linestyle)
    axes[1].plot(x2, pd_acc2[0].rolling(20).mean(), label=method, linestyle=linestyle)

    axes[0].set_ylabel('Test ACC', fontsize=22)
    axes[1].set_ylabel('Test Loss', fontsize=22)

    axes[0].set_xlabel('Epoch', fontsize=22)
    axes[1].set_xlabel('Epoch', fontsize=22)

    axes[0].set_title('GOOD-CBAS-color ACC', fontsize=22)
    axes[1].set_title('GOOD-CBAS-color Loss', fontsize=22)

    # axes[0].legend(loc=7, prop={'size': 10})
    # axes[1].legend()
    # ax.fill_between(standard_ticks, y1, y2, alpha=.5, linewidth=0)
    # ax.axhline(y=ERM_results[dataset_key], color='r', linestyle='--', label='ERM')
    # ax.plot(standard_ticks, (y1 + y2) / 2, '.-', linewidth=2, label='LECI')
    # ax.set(xticks=standard_ticks, xticklabels=x)
    # ax.set_ylabel('Test metric', fontsize=22)
    # ax.set_xlabel(f'$\lambda_\mathtt{{{self.hparam}}}$', fontsize=22)
    # ax.legend(loc=1, prop={'size': 16})
    # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0, hspace=0)

    # fig.tight_layout()
    # fig.subplots_adjust(right=0.5)
fig.legend(*axes[0].get_legend_handles_labels(),loc=7)
plt.subplots_adjust(left=0.08, bottom=0.15, right=0.85, top=0.9, wspace=0.2, hspace=0)
# plt.show()
save_path = os.path.join(STORAGE_DIR, 'figures', 'stability_study')
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = ROOT_DIR
fig.savefig(os.path.join(save_path, f'5.png'))