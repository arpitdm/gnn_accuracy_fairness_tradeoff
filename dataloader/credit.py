import numpy as np
import pandas as pd
import os.path as osp
import scipy.sparse as sp
from typing import Optional, Callable, List

import torch
from torch_geometric.data import extract_zip
from torch_geometric.utils import from_scipy_sparse_matrix

from dataloader.utils import CustomInMemoryDataset, CustomData


class Credit(CustomInMemoryDataset):
	# name and URLs are set as class variables
	name = 'Credit'

	URLs = ['https://github.com/chirag126/nifty/raw/main/dataset/credit/credit.csv', 'https://github.com/chirag126/nifty/raw/main/dataset/credit/credit_edges.txt.zip']

	def __init__(self, root, split: str = "default", transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):       
		self.split = split # not used in our experiments
		super().__init__(root, pre_transform=pre_transform, transform=transform)

	def download(self):
		super(Credit, self).download()
		extract_zip(osp.join(self.raw_dir, 'credit_edges.txt.zip'), self.raw_dir)

	@property
	def raw_file_names(self) -> List[str]:
		# Names of raw file names
		names = ['.csv','_edges.txt.zip']
		return [f'{self.name.lower()}{name}' for name in names]

	def create_adjacency_matrix(self, features:pd.DataFrame) -> sp.csr_matrix:
		# builds edges (NIFTY) and converts to sp.csr_matrix with self-loops
		edges_unordered = self.build_edges(fname="credit_edges.txt",x=features, thresh=0.7, method='NIFTY')
		adj = self.unordered_edges_to_adjacency_matrix(num_nodes=features.shape[0], edges_unordered=edges_unordered)
		return adj

	def process(self):
		# load raw node attribute and label data
		idx_XY = pd.read_csv(osp.join(self.raw_dir,"credit.csv"))

		# create binary sensitive attribute array: sens
		# 1 indicates protected class
		sens_attr = "Age"
		sens = idx_XY[sens_attr].values.astype(int)

		# create binary node label array: Y
		predict_attr = "NoDefaultNextMonth"
		labels = idx_XY[predict_attr].values

		# create node attribute matrix: X
		# remove irrelevant attributes
		header = list(idx_XY.columns)
		header.remove(predict_attr)
		sens_idx = header.index(sens_attr)
		header = header[:sens_idx] + header[(sens_idx+1):] + [header[sens_idx]]
		sens_idx = -1
		features = sp.csr_matrix(idx_XY[header], dtype=np.float32)

		# create adjacency matrix and edge_index
		adj = self.create_adjacency_matrix(idx_XY[header])
		edge_index_original,_ = from_scipy_sparse_matrix(adj)

		# convert to pytorch Tensors
		sens = torch.FloatTensor(sens)
		Y_original = torch.LongTensor(labels)
		X_original = torch.FloatTensor(np.array(features.todense()))

		# create ranked list for PFR (here, MaxBillAmountOverLast6Months)
		Ys = np.array(idx_XY['MaxBillAmountOverLast6Months'])

		# create PyG Data (data.pt) object, save to processed/
		# also save original (processed) graph in edgelist format to processed/
		_ = CustomData(edge_index_original=edge_index_original,X_original=X_original,Y_original=Y_original,sens=sens,predict_attr=predict_attr,sens_attr=sens_attr,sens_idx=sens_idx,Ys=Ys, header=header, pre_transform=self.pre_transform, processed_paths=self.processed_paths, processed_dir=self.processed_dir, name=self.name, edge_index_str="edge_index_original")