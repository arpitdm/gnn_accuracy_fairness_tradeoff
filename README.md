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
See below for sample usage options. If you use any baseline interventions for your experiments, please cite the original authors.

Results are logged in a CSV file with one line per execution of `fair_train.py`. This will store all the hyperparameters and optional arguments used in that execution. Intermediate data and model files are also stored to disk for re-use. Appropriate `recompute` flags trigger a rebuild. 

### Sample Usage:

(a) `Original` `GIN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=pretrain --pretrain_algo=original --debias_X=0 --debias_A=0 --train_size=0.6 --model_name=GIN --epochs=1000 --hidden=128 --lr=5e-5 --dropout=0.1 --weight_decay=1e-5 --seed=1 --n_estimators=100 --exp_logfilename=original.csv --verbose=1`
```
AUC-ROC: 0.7119385342789598
F1-Score: 0.8307692307692308
Parity: 0.061766735679779194
Equality: 0.02776459420977684
```

(b) `Unaware` `GIN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=pretrain --pretrain_algo=unaware --debias_X=1 --debias_A=0 --train_size=0.6 --model_name=GIN --epochs=1000 --hidden=64 --lr=5e-5 --dropout=0.1 --weight_decay=1e-5 --seed=1 --n_estimators=100 --exp_logfilename=unaware.csv --verbose=1`
```
AUC-ROC: 0.71725768321513
F1-Score: 0.8303030303030303
Parity: 0.00552104899930983
Equality: 0.04081632653061229
```

(c) `EDITS` `GIN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=pretrain --pretrain_algo=EDITS --debias_X=1 --debias_A=0 --train_size=0.6 --model_name=GIN --epochs=500 --hidden=64 --lr=1e-4 --dropout=0.1 --weight_decay=1e-5 --seed=5 --edits_epochs=20 --edits_nfeat_out=10 --edits_lr=1e-3 --edits_dropout=0.0 --edits_weight_decay=1e-7 --edits_adj_lambda=0.1 --edits_layer_threshold=2 --exp_logfilename=edits.csv --verbose=1`
```
AUC-ROC: 0.6851063829787234
F1-Score: 0.825301204819277
Parity: 0.0200138026224983
Equality: 0.007356430944470804
```

(e) `PFR-AX` `GIN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=pretrain --pretrain_algo=PFR --debias_X=0 --debias_A=1 --train_size=0.6 --model_name=GIN --epochs=500 --hidden=64 --lr=1e-4 --dropout=0.1 --weight_decay=1e-5 --seed=2 --embed_algo=DeepWalk --dim=128 --exp_logfilename=pfr.csv --pfr_k=26 --pfr_quantiles=4 --pfr_nn_k=50 --pfr_t=2 --pfr_gamma=0.5 --pfr_q=0.5 --pfr_A_k=128 --pfr_A_quantiles=4 --pfr_A_nn_k=50 --pfr_A_t=10 --pfr_A_gamma=0.1 --pfr_A_q=0.5 --invert_algo=Adjacency_Similarity --invert_maxiter=50 --as_create_method=soft_consistency --rounds=20 --exp_logfilename=pfr.csv --verbose=1`
```
AUC-ROC: 0.660047281323877
F1-Score: 0.8203592814371257
Parity: 0.034851621808143496
Equality: 0.04081632653061229
```

(f) `NIFTY` `GIN` on the `German` data:

`python fair_train.py --dataset_name=German --locus=intrain --intrain_algo=NIFTY --debias_X=0 --debias_A=0 --train_size=0.6 --model_name=GIN --epochs=1500 --hidden=128 --lr=1e-4 --dropout=0.1 --weight_decay=1e-5 --seed=4 --n_estimators=100 --embed_algo=DeepWalk --dim=128 --max_depth=1000 --drop_feature_rate_1=0.001 --drop_feature_rate_2=0.001 --sim_coeff=0.1 --drop_edge_rate_1=0.001 --drop_edge_rate_2=0.001 --exp_logfilename=nifty.csv --verbose=1`
```
AUC-ROC: 0.6570921985815602
F1-Score: 0.8073394495412844
Parity: 0.039337474120082816
Equality: 0.058376839107736056
```

Update: Refactored GNN model definition into an `Encoder`-`Classifier` architecture for general use and drop-in replacements. Refactored training pipeline to reduce convolution. Logging with Hydra.

