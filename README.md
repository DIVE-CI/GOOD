

## Installation 

### Conda dependencies

GOOD depends on [PyTorch (>=1.6.0)](https://pytorch.org/get-started/previous-versions/), [PyG (>=2.0)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), and
[RDKit (>=2020.09.5)](https://www.rdkit.org/docs/Install.html). For more details: [conda environment](/../../blob/main/environment.yml)

> Note that we currently test on PyTorch (==1.10.1), PyG (==2.0.3), RDKit (==2020.09.5); thus we strongly encourage to install these versions.
>
> Attention! Due to a known issue, please install PyG through Pip to avoid incompatibility.

### Pip

#### Installation for only modules

```shell
pip install graph-ood
```

#### Installation for Project usages (recommended)

```shell
git clone https://github.com/divelab/GOOD.git && cd GOOD
pip install -e .
```

## Quick Tutorial

### Module usage

#### GOOD datasets
There are two ways to import 8 GOOD datasets with 14 domain selections and a total 42 splits, but for simplicity, we only show one of them.
Please refer to [Tutorial](https://good.readthedocs.io/en/latest/tutorial.html) for more details.
```python
# Directly import
from GOOD.data.good_datasets.good_hiv import GOODHIV
hiv_datasets, hiv_meta_info = GOODHIV.load(dataset_root, domain='scaffold', shift='covariate', generate=False)
```

#### GOOD GNNs
The best and fair way to compare algorithms with the leaderboard is to use the same and similar graph encoder structure;
therefore, we provide GOOD GNN APIs to support. Here, we use an objectified dictionary `config` to pass parameters. More
details about the config: [Documents of config](https://good.readthedocs.io/en/latest/configs.html) In this case, we 
recommend using this package for project usages.

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

#### GOOD algorithms
Try to apply OOD algorithms to your own models?
```python
from GOOD.ood_algorithms.algorithms.VREx import VREx
ood_algorithm = VREx(config)
# Then you can provide it to your model for necessary ood parameters, 
# and use its hook-like function to process your input, output, and loss.
```

### Project usage

It is a good beginning to make it work directly. Here, we provide the command line script `goodtg` (GOOD to go) to access the main function located at `GOOD.kernel.pipeline:main`.
Choosing a config file in `configs/GOOD_configs`, we can start a task:

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

Please refer to [Tutorial](https://good.readthedocs.io/en/latest/tutorial.html) for more details.

## Reproducibility

For reproducibility, we provide full configurations used to obtain leaderboard results in [configs/GOOD_configs](/../../blob/main/configs/GOOD_configs).

We further provide two tests: dataset regeneration test and test result check.

### Dataset regeneration test

This test regenerates all datasets again and compares them with the datasets used in the original training process locates.
Test details can be found at [test_regenerate_datasets.py](/../../blob/main/test/test_reproduce_full/test_regenerate_datasets.py).
For a quick review, we provide a [full regeneration test report](https://drive.google.com/file/d/1jIShh3eBXAQ_oQCFL9AVU3OpUlVprsbo/view?usp=sharing).

### Leaderboard results test

This test loads [all checkpoints in round 1](https://drive.google.com/file/d/17FfHYCP0-wwUILPD-PczwjjrYQHKxU-l/view?usp=sharing) and
compares their results with saved ones. Test details can be found at [test_reproduce_round1.py](/../../blob/main/test/test_reproduce_full/test_reproduce_round1.py).
For a quick review, we also post our [full round1 reproduce report](https://drive.google.com/file/d/1kR4k0E0y6Rtcx4WbjevSxKviHrkx3G1y/view?usp=sharing).

These reports are in `html` format. Please download them and open them in your browser.: )

**Training plots:**
The training plots for all algorithms in round 1 can be found [HERE](https://drive.google.com/file/d/1-UsWstrF1cxk7MExRV-37emGi4spQtj0/view?usp=sharing).

### Sampled tests

In order to keep the validity of our code all the time, we link our project with circleci service and provide several 
sampled tests to go through (because of the limitation of computational resources in CI platforms).

## Citing GOOD
If you find this repository helpful, please cite our [paper](https://arxiv.org/abs/2206.08452).
```
@article{gui2022good,
  title={{GOOD}: A Graph Out-of-Distribution Benchmark},
  author={Gui, Shurui and Li, Xiner and Wang, Limei and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2206.08452},
  year={2022}
}
```

## Discussion

Please submit [new issues](/../../issues/new) or start [a new discussion](/../../discussions/new) for any technical or other questions.

## Contact

Please feel free to contact [Shurui Gui](mailto:shurui.gui@tamu.edu), [Xiner Li](mailto:lxe@tamu.edu), or [Shuiwang Ji](mailto:sji@tamu.edu)!

