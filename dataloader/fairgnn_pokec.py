import shutil
import numpy as np
import pandas as pd
import os.path as osp
import scipy.sparse as sp
from typing import Optional, Callable, List

import torch
from torch_geometric.utils import from_scipy_sparse_matrix

from dataloader.utils import CustomInMemoryDataset, CustomData


class FairGNNPokec(CustomInMemoryDataset):

	def __init__(self, root, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):       
		self.name = root.split('/')[-1]
		url_base = 'https://github.com/EnyanDai/FairGNN/raw/main/dataset/pokec'
		if self.name == 'Region-Z':
			self.f_ext = ''
		elif self.name == 'Region-N':
			self.f_ext = '_2'
		self.URLs = [f'{url_base}/region_job{self.f_ext}.csv', f'{url_base}/region_job_relationship{self.f_ext}.txt']
		super().__init__(root, pre_transform=pre_transform, transform=transform)
	
	@property
	def raw_file_names(self) -> List[str]:
		# Names of raw file names
		names = [self.name+ext for ext in ['.content','.edges']]
		return names

	def download(self):
		super(FairGNNPokec, self).download()
		shutil.copy(osp.join(self.raw_dir, f'region_job{self.f_ext}.csv'), osp.join(self.raw_dir, f'{self.name}.content'))
		shutil.copy(osp.join(self.raw_dir, f'region_job_relationship{self.f_ext}.txt'), osp.join(self.raw_dir, f'{self.name}.edges'))

	def create_adjacency_matrix(self, idx:np.array, num_nodes:int) -> sp.csr_matrix:
		# takes idx from attribute+label file to map to edgelist file
		edges_unordered = self.build_edges(f'{self.name}.edges')
		adj = self.unordered_edges_to_adjacency_matrix(num_nodes=num_nodes, edges_unordered=edges_unordered, idx=idx)
		return adj

	def process(self):
		# load node attribute, and label data
		idx_XY = pd.read_csv(osp.join(self.raw_dir,f'{self.name}.content'))

		# create binary sensitive attribute array: sens
		# 1 indicates protected class
		sens_attr = "region"
		sens = idx_XY[sens_attr].values.astype(int)

		# create binary node label array: Y
		predict_attr = "I_am_working_in_field"
		labels = idx_XY[predict_attr].values
		labels[labels >= 0] = 0
		labels[labels == -1] = 1

		# create node attribute matrix: X
		# remove irrelevant attributes
		header = list(idx_XY.columns)
		header.remove(predict_attr)
		header.remove('user_id')
		sens_idx = header.index(sens_attr)
		header = header[:sens_idx] + header[(sens_idx+1):] + [header[sens_idx]]
		sens_idx = -1  
		features = sp.csr_matrix(idx_XY[header], dtype=np.float32)

		# create adjacency matrix and edge_index
		# get col_idx of sensitive attribute from features matrix
		idx = idx_XY['user_id'].to_numpy()
		adj = self.create_adjacency_matrix(idx, features.shape[0])
		edge_index_original,_ = from_scipy_sparse_matrix(adj)

		# convert to pytorch Tensors
		sens = torch.FloatTensor(sens)
		Y_original = torch.LongTensor(labels)
		X_original = torch.FloatTensor(np.array(features.todense()))

		# create ranked list for PFR (here, spoken_languages_indicator)
		Ys = np.array(idx_XY['spoken_languages_indicator'])

		# create PyG Data (data.pt) object, save to processed/
		# also save original (processed) graph in edgelist format to processed/
		_ = CustomData(edge_index_original=edge_index_original,X_original=X_original,Y_original=Y_original,sens=sens,predict_attr=predict_attr,sens_attr=sens_attr,sens_idx=sens_idx,Ys=Ys, header=header, pre_transform=self.pre_transform, processed_paths=self.processed_paths, processed_dir=self.processed_dir, name=self.name, edge_index_str="edge_index_original")