# Anonymous repo for paper review

This paper follows the GOOD benchmark [1] structure.
**The method name in this repo is called GEI instead of LECI**.

The main method codes are in `GOOD/networks/models/GEIGNN.py` and `GOOD/ood_algorithms/algorithms/GEI.py`.

Note that **there are many unused experimental hyperparameters and components** in
GEIGNN.py. Therefore, for your convenience, we list the major points that need attentions, since
they are corresponding to the techniques mentioned in the submitted paper.

* self.EA: Environment adversarial component.
* self.LA: Label adversarial component.
* GradientReverseLayerF: Gradient reverse component.
* `set_masks(GradientReverseLayerF.apply(edge_att, self.EA * self.config.train.alpha), self.ea_gnn)` in line 129 of GEIGNN.py:
Applying environment discriminator directly on $A_C$. (**graph-specific**)
* ...

For more information, please refer to the code. 

How to run the code:

1. Install the prerequisite packages:
```shell
conda create -y -n LECI python=3.8
conda activate LECI
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -y pyg -c pyg
conda install -y -c conda-forge rdkit==2020.09.5
```
2. Clone and install this project `pip install -e .`
3. Run our method: 
* `goodtg --config_path GOOD_configs/GOODMotif/basis/covariate/GEI.yaml --gpu_idx 0`
* `goodtg --config_path GOOD_configs/GOODMotif/size/covariate/GEI.yaml --gpu_idx 0`
* `goodtg --config_path GOOD_configs/GOODMotif2/basis/covariate/GEI.yaml --gpu_idx 0`
* `goodtg --config_path GOOD_configs/GOODCMNIST/color/covariate/GEI.yaml --gpu_idx 0`
* `goodtg --config_path GOOD_configs/GOODHIV/scaffold/covariate/GEI.yaml --gpu_idx 0`
* `goodtg --config_path GOOD_configs/GOODHIV/size/covariate/GEI.yaml --gpu_idx 0`
* `goodtg --config_path GOOD_configs/LBAPcore/assay/covariate/GEI.yaml --gpu_idx 0`
* `goodtg --config_path GOOD_configs/GOODSST2/length/covariate/GEI.yaml --gpu_idx 0`
* `goodtg --config_path GOOD_configs/GOODTwitter/length/covariate/GEI.yaml --gpu_idx 0`

We also provide an example anonymous log `./Motif-base.log` for your reference, in case that
code cannot be run because of the environment configuration. (Note that we remove all the folder paths and time frames in it 
to make it totally anonymous.)

[1] Good: A graph out-of-distribution benchmark. arXiv preprint arXiv:2206.08452.