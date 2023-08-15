import logging
import argparse
import numpy as np
from pathlib import Path

import torch_geometric
from torch_geometric.utils import homophily

from dataloader.german import German
from dataloader.credit import Credit
from dataloader.fairgnn_pokec import FairGNNPokec
from dataloader.linkx import LINKX

from parse_args import parse_dataset_args
from dataloader.utils import AddRemainingSelfLoops, TrainValTestMask


def load_dataset(args: argparse.Namespace) -> torch_geometric.data.Data:
	r"""
	Creates a `data` object for GNN training. If raw files are unavailable on disk, downloads and processes them. 

	Args:
		- `args` (`argparse.Namespace`): Argparser object containing relevant 
			arguments obtained from command-line.

	Returns:
		- `data` (`torch_geometric.data.Data`): Data object containing 
			processed adjacency, attributes, labels, and, random and stratified train-val-test split masks.
	
	Notes:
		- Does not use the train-val-test split logic from NIFTY and subsequent literature due to a quirk in the `label_number` logic that results in potentially not using a subset of nodes of the graph. See `TrainValTestMask` for a stratified random split.
		- Has single binary sensitive attribute and single binary label. sensitive attribute is always the last column of `data.X_original`.
	"""
	pre_transform = AddRemainingSelfLoops()
	transform = TrainValTestMask(train_size=args.train_size,val_size=args.val_size,seed=args.dataset_seed)
	
	if args.dataset_name == "German":
		dataset = German(root='./tmp/German',pre_transform=pre_transform, transform=transform)
	elif args.dataset_name == 'Recidivism':
		dataset = Recidivism(root='./tmp/Recidivism',pre_transform=pre_transform,transform=transform)
	elif args.dataset_name == 'Credit':
		dataset = Credit(root='./tmp/Credit',pre_transform=pre_transform,transform=transform)
	elif args.dataset_name in ["Region-N", "Region-Z"]:
		dataset = FairGNNPokec(root=f'./tmp/{args.dataset_name}',pre_transform=pre_transform,transform=transform)
	elif args.dataset_name in ['Penn94', 'Reed98', 'Amherst41', 'Cornell5', 'Johnshopkins55']:
		dataset = LINKX(root=f'./tmp/{args.dataset_name}',pre_transform=pre_transform,transform=transform)
	elif args.dataset_name == 'TwitchGamer':
		dataset = TwitchGamer(root='./tmp/TwitchGamer',pre_transform=pre_transform,transform=transform)
	else:
		raise ValueError("Dataset not available.")

	data = dataset[0]
	data.data_dir = Path(dataset.processed_dir).parent
	data.processed_dir = Path(dataset.processed_dir)

	return data


def compute_statistics(args:argparse.Namespace, data: torch_geometric.data.Data) -> str:
	r"""
	Computes dataset statistics including edge-homophily with respect to sensitive attribute and label.

	Args:
		- `args` (`argparse.Namespace`): Argparser object containing relevant 
			arguments obtained from command-line.
		- `data` (torch_geometric.data.Data): The data object.
	"""
	# edge homophily w.r.t sens and label
	h_edge_sens = np.round(homophily(data.edge_index_original,data.X_original[:,data.sens_idx],method="edge"),2)
	h_edge_label = np.round(homophily(data.edge_index_original,data.Y_original,method="edge"),2)

	stats_keys = ['Dataset', 'num_nodes', 'num_edges', 'num_classes', 'num_features', 'sensitive_attribute', 'predict_attribute', 'sens_attr_homophily', 'label_homophily']
	stats_keys = ','.join(stats_keys)
	stats = [args.dataset_name, data.num_nodes, data.num_edges, data.Y_original.unique().shape[0], data.X_original.shape[1], data.sens_attr, data.predict_attr, h_edge_sens, h_edge_label]
	stats_vals = ','.join([str(s) for s in stats])

	logpath = Path('results')
	logpath.mkdir(parents=True, exist_ok=True)
	logfname = logpath / 'dataset_statistics.csv'
	logging.basicConfig(filename=logfname, level=getattr(logging, 'INFO'))
	logging.info(stats_keys)
	logging.info(stats_vals)

	print(f"Sensitive Attribute Homophily: {h_edge_sens}")
	print(f"Label Homophily: {h_edge_label}")


def main():
    # parse args
	parser = argparse.ArgumentParser()
	parser = parse_dataset_args(parser)
	args = parser.parse_args()
	args.parser = parser

	# load data
	data = load_dataset(args)

	# compute statistics
	print(f"Dataset: {args.dataset_name}")
	print(f"|V|: {data.num_nodes}, |E|: {data.num_edges}")
	print(f"Number of Attributes: {data.num_attributes}, Sensitive Attribute: {data.sens_attr}, Predict: {data.predict_attr}")
	if args.compute_dataset_stats:
		compute_statistics(args, data)
 
if __name__ == '__main__':
	main()