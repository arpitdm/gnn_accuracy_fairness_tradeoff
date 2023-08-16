# Overview:

This repository provides support for downloading and preprocessing datasets, building and training GNN models, and implementing algorithmic fairness interventions as described in our paper on *Disparity, Inequality, and Accuracy of Graph Neural Networks for Node Classification (CIKM 2023) with Arpit Merchant and Carlos Castillo*.

### Installation
Our experiments were conducted on a Linux machine with 32 cores, (maximum) 100GB RAM, and a V100 GPU. We used the following packages/frameworks:
1. Python/3.7
	For convenience regarding Python packages, we provide [environment.yml](environment.yml). Create a conda environment called `postprocess_gnn` with `conda env create -f environment.yml`.
2. GCC/12.2.0, Snap/6.0 (`DeepWalk`).
	Download the source code from [here](https://snap.stanford.edu/snap/download.html) and [install](https://snap.stanford.edu/snap/install.html) into `./embeddings/snap`.

### Datasets:
We conduct our experiments on 4 datasets namely `German`, `Credit`, `Penn94`, `Region-Z`. To download, preprocess, and save any of these datasets to disk, specifically `./tmp`, execute the following:

```python load_dataset.py --dataset_name <dataset-name>```

For further options, see `def parse_dataset_args` in [parse_args.py](parse_args.py). If you use any of these datasets in your research, please cite the original authors.

To add your own datasets, extend the `CustomInMemoryDataset` class in [utils.py](dataloader/utils.py) as per [load_dataset.py](load_dataset.py). We use a standard stratified train-val-test split. To create your own custom split, extend `BaseTransform` like in [TrainValTestMask](utils.py).


### Interventions:

Key Arguments:
1. Select a `--dataset_name`.
2. Select `--locus` from `pretrain`, `intrain`, and `posttrain` indicating where the intervention will be applied.
3. Select appropriate intervention algorithm from `original`, `unaware`, `EDITS`, `PFR`, `NIFTY`, and `Blackbox-Pred` (PostProcess).
4. Select `--model_name` from `GCN`, `GraphSAGE`, and `GIN`.
5. Select `seed`.
6. Select hyperparameters.
7. Select logging options.

For a complete list of available options, please refer to:
`python fair_train.py --help`. 
See below for sample usage (not necessarily optimal) options. If you use any baseline interventions for your experiments, please cite the original authors.

Results are logged in a CSV file with one line per execution of `fair_train.py`. This will store all the hyperparameters and optional arguments used in that execution. Intermediate data and model files are also stored to disk for re-use. Appropriate `recompute` flags trigger a rebuild. 

### Sample Usage:

(a) `Original` `GCN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=pretrain --pretrain_algo=original --debias_X=0 --debias_A=0 --train_size=0.6 --model_name=GCN --epochs=2000 --hidden=128 --lr=1e-4 --weight_decay=1e-5 --seed=1 --exp_logfilename=original.csv --verbose 1`
```
AUC-ROC: 0.6828605200945627
F1-Score: 0.8150470219435736
Parity: 0.04140786749482406
Equality: 0.01186521120075934
```

(b) `Unaware` `GCN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=pretrain --pretrain_algo=unaware --debias_X=1 --debias_A=0 --train_size=0.6 --model_name=GCN --epochs=2000 --hidden=128 --lr=1e-4 --weight_decay=1e-5 --seed=1 --exp_logfilename=unaware.csv --verbose 1`
```
AUC-ROC: 0.6756501182033098
F1-Score: 0.822429906542056
Parity: 0.009661835748792313
Equality: 0.02491694352159468
```

(c) `EDITS-X` `GCN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=pretrain --pretrain_algo=EDITS --debias_X=1 --debias_A=0 --train_size=0.6 --model_name=GCN --epochs=1000 --hidden=128 --lr=1e-4 --weight_decay=1e-5 --seed=1 --edits_epochs=100 --exp_logfilename=edits.csv --verbose 1`
```
AUC-ROC: 0.6892434988179669
F1-Score: 0.8063492063492064
Parity: 0.012422360248447228
Equality: 0.009017560512577072
```

(e) `PFR-AX` `GCN` on the `German` data:

`python fair_train.py --dataset_name German --locus pretrain --pretrain_algo PFR --debias_X 1 --debias_A 1 --train_size 0.6 --model_name GCN --epochs 1000 --hidden=256 --lr=1e-4 --weight_decay 1e-5 --seed 1 --embed_algo DeepWalk --pfr_k 26 --pfr_quantiles 4 --pfr_nn_k 50 --pfr_t 2 --pfr_gamma 0.5 --pfr_q 0.5 --pfr_A_k 128 --pfr_A_quantiles 10 --pfr_A_nn_k 10 --pfr_A_t 2 --pfr_A_gamma 0.5 --pfr_A_q 0.5 --invert_algo Adjacency_Similarity --as_create_method soft_consistency --rounds 10 --exp_logfilename=pfr.csv --verbose 1`
```
AUC-ROC: 0.6297872340425532
F1-Score: 0.8036253776435045
Parity: 0.03588681849551423
Equality: 0.018747033697199877
```

(f) `NIFTY` `GIN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=intrain --intrain_algo=NIFTY --train_size=0.6 --model_name=GCN --encoder_name=GCN --epochs=1500 --hidden=128 --lr=1e-4 --weight_decay=1e-5 --seed=1 --drop_edge_rate_1=0.001 --drop_edge_rate_2=0.001 --drop_feature_rate_1=0.01 --drop_feature_rate_2=0.01 --sim_coeff=0.1  --exp_logfilename=nifty.csv --verbose 1`
```
AUC-ROC: 0.6747044917257683
F1-Score: 0.8065573770491803
Parity: 0.05555555555555558
Equality: 0.0170859041290935
```

(g) `PostProcess` `GCN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=posttrain --posttrain_algo=Blackbox-Pred --train_size=0.6 --model_name=GCN --epochs=2000 --hidden=128 --lr=1e-4 --weight_decay=1e-5 --seed=1 --flip_frac=0.2 --exp_logfilename=blackbox.csv --verbose 1`
```
AUC-ROC: 0.6800709219858155
F1-Score: 0.8162499999999999
Parity: 0.02553485162180813
Equality: 0.025818699572852388
```

Update: Refactored GNN model definition into an `Encoder`-`Classifier` architecture for general use and drop-in replacements. Refactored training pipeline to reduce convolution. Logging with Hydra.

