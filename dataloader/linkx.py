import numpy as np
import pandas as pd
from typing import List
import scipy.sparse as sp
from scipy.io import loadmat

import torch
from torch_geometric.utils import from_scipy_sparse_matrix

from dataloader.utils import CustomInMemoryDataset, CustomData 


class LINKX(CustomInMemoryDataset):    
	def __init__(self, root, transform=None, pre_transform=None):
		self.name = root.split('/')[-1]
		self.URLs = [f'https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100/{self.name}.mat']
		super().__init__(root, pre_transform=pre_transform, transform=transform)

	@property
	def raw_file_names(self) -> List[str]:
		names = [f'{self.name}.mat']
		return names
    
	def create_adjacency_matrix(self, mat: dict) -> sp.csr_matrix:
		adj = sp.csr_matrix(mat["A"]).astype(int)
		adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
		adj.setdiag(1, k=0)
		return adj

	def process(self):
		# load raw edge, node attribute, and label data
		mat = loadmat(self.raw_paths[0])
		header = ['Major', 'Gender', 'F2', 'F3', 'F4', 'Year', 'F6']
		idx_XY = pd.DataFrame(mat['local_info'].astype(int), columns=header)

		# create binary sensitive attribute array: sens
		# 1 indicates protected class
		sens_attr = 'Gender'
		sens = idx_XY[sens_attr].values.astype(int)
		sens[sens <= 1] = 0
		sens[sens == 2] = 1
		idx_XY[sens_attr] = sens
	
		# create binary node label array: Y
		predict_attr = 'Year'
		def binarize_label(x):
			if x >= 2008:
				return 1
			else:
				return 0
		idx_XY['Year'] = idx_XY['Year'].apply(binarize_label)
		labels = idx_XY[predict_attr].values

		# create node attribute matrix: X
		# remove irrelevant attributes
		header.remove(predict_attr)
		header.remove(sens_attr)
		header = header + [sens_attr]
		sens_idx = -1
		features = sp.csr_matrix(idx_XY[header], dtype=np.float32)

		# create adjacency matrix and edge_index
		adj = self.create_adjacency_matrix(mat)
		edge_index_original,_ = from_scipy_sparse_matrix(adj)

		# convert to pytorch Tensors
		sens = torch.FloatTensor(sens)
		Y_original = torch.LongTensor(labels)
		X_original = torch.FloatTensor(np.array(features.todense()))

		# create ranked list for PFR (here, F6)
		Ys = np.array(idx_XY['F6'])

		# create PyG Data (data.pt) object, save to processed/
		# also save original (processed) graph in edgelist format to processed/
		_ = CustomData(edge_index_original=edge_index_original,X_original=X_original,Y_original=Y_original,sens=sens,predict_attr=predict_attr,sens_attr=sens_attr,sens_idx=sens_idx,Ys=Ys, header=header, pre_transform=self.pre_transform, processed_paths=self.processed_paths, processed_dir=self.processed_dir, name=self.name, edge_index_str="edge_index_original")