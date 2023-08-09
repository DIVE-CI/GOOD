

## Installation 

### Conda dependencies

Depends on [PyTorch (>=1.6.0)](https://pytorch.org/get-started/previous-versions/), [PyG (>=2.0)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), and
[RDKit (>=2020.09.5)](https://www.rdkit.org/docs/Install.html). For more details: [conda environment](/../../blob/main/environment.yml)

> Note that we currently test on PyTorch (==1.10.1), PyG (==2.0.3), RDKit (==2020.09.5); thus we strongly encourage to install these versions.
>
> Attention! Due to a known issue, please install PyG through Pip to avoid incompatibility.

### Pip

#### Installation for Project usages (recommended)

```shell
pip install -e .
```

## Quick Tutorial

### Module usage

#### Datasets
```python
# Directly import
from GOOD.data.good_datasets.good_hiv import GOODHIV
hiv_datasets, hiv_meta_info = GOODHIV.load(dataset_root, domain='scaffold', shift='covariate', generate=False)
```

#### GNNs
*To use exact GNN*
```python
from GOOD.networks.models.GCNs import GCN
model = GCN(config)
```
*To only use parts of GNN*
```python
from GOOD.networks.models.GINvirtualnode import GINEncoder
encoder = GINEncoder(config)
```

#### Algorithms

```python
from GOOD.ood_algorithms.algorithms.VREx import VREx
ood_algorithm = VREx(config)
# Then you can provide it to your model for necessary ood parameters, 
# and use its hook-like function to process your input, output, and loss.
```

### Project usage

Use the command line script `goodtg` (GOOD to go) to access the main function located at `GOOD.kernel.pipeline:main`.
Choosing a config file in `configs/GOOD_configs`, start a task:

```shell
goodtg --config_path GOOD_configs/GOODCMNIST/color/concept/DANN.yaml
```

Specifically, the task is clearly divided into three steps:
1. **Config**
```python
from GOOD import config_summoner
from GOOD.utils.args import args_parser
from GOOD.utils.logger import load_logger
args = args_parser()
config = config_summoner(args)
load_logger(config)
```
2. **Loader**
```python
from GOOD.kernel.pipeline import initialize_model_dataset
from GOOD.ood_algorithms.ood_manager import load_ood_alg
model, loader = initialize_model_dataset(config)
ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
```
3. **Train/test pipeline**
```python
from GOOD.kernel.pipeline import load_task
load_task(config.task, model, loader, ood_algorithm, config)
```
