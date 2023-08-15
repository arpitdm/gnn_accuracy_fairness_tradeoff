import time
import warnings
import numpy as np
from pathlib import Path
from torch_geometric.utils import to_dense_adj

from parse_args import *
from utils import get_embed_algo_strings
from load_dataset import load_dataset
from embeddings.spectral_embedding import create_Spectral_Embedding
from embeddings.deepwalk import create_DeepWalk_Embedding
from embeddings.gosh import create_GOSH_Embedding


def load_adjacency(args, data):
	# if not args.debias_A:
	args.A_type = 'original'
	data.A = to_dense_adj(data.edge_index_original).numpy()[0]
	data.A_fname = data.processed_dir / f'{args.dataset_name}_edge_index_original.edgelist'
	return args, data	
	

def create_embedding(args, data):
	if args.embed_algo == 'Spectral_Embedding':
		U = create_Spectral_Embedding(args, data, data.A)
	elif args.embed_algo == 'DeepWalk':
		U = create_DeepWalk_Embedding(args, data, data.A_fname)
	elif args.embed_algo == 'GOSH':
		U = create_GOSH_Embedding(args, data, data.A_fname)
	else:
		raise NotImplementedError
	return U


def main():
	warnings.filterwarnings("ignore")

	# parse arguments
	parser = argparse.ArgumentParser()
	parser = parse_dataset_args(parser)
	parser = parse_embedding_args(parser)
	parser = parse_pretrain_args(parser)
	parser = parse_invert_args(parser)
	args = parser.parse_args()
	args.parser = parser

	# load dataset
	data = load_dataset(args)

	# get adjacency matrix and its edgelist fname
	# choice between original, inverted, debias+inverted
	args, data = load_adjacency(args, data)
	print(f"Adjacency file-name: {data.A_fname}")

	# get fname where embedding is to be stored
	data.U_param_str, data.U_savepath, data.U_fname = get_embed_algo_strings(args, data)
	print(f"Embedding file-name: {data.U_fname}")

	# create/load embedding of adjacency matrix chosen above
	t0 = time.time()
	data.U = create_embedding(args, data)
	t1 = time.time()
	print(f"Time to compute embedding: {np.round(t1-t0)} seconds")
	print(f"data.U.shape: {data.U.shape}")


if __name__ == '__main__':
	main()