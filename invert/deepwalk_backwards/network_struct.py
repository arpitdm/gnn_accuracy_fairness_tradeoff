"""
Modified implementation of DeepWalking Backwards. Original implementation by Konstantinos Sotiropoulos (ksotirop@bu.edu) 
Link: https://github.com/konsotirop/Invert_Embeddings.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy
from scipy.sparse.csgraph import connected_components


class Network:
	def __init__( self ):
		self.Adjacency = None
		self.Labels = None
		self.Embedding = None
		self.LR_Embedding = None
		self.G = None
		self.labeled = False
		self.rank = None

	def loadNetwork( self, A, Labels="NA", binarize=False ):
		"""
		Loads a network and its labels (if available)
		"""
		self.Adjacency = A
		if binarize:
			self.Adjacency.data = 1. * (self.Adjacency.data > 0)
	
		# Load labels - if available
		if isinstance(Labels, (np.ndarray, np.generic)):
			self.Labels = Labels
			self.labeled = True
		else:
			self.Labels = [None]
			self.labeled = False
		#print("Labeled?", self.labeled)
		return

	def SBM( self, sizes, probs ):
		self.G = nx.stochastic_block_model( sizes, probs )
		self.Adjacency = nx.to_scipy_sparse_matrix(self.G)
		self.Labels = []
		[self.Labels.extend([i for _ in range(sizes[i])]) for i in range(len(sizes))]
		self.Labels = np.array( self.Labels )
		self.labeled = True
		return

	def standardize( self ):
		"""
		Make the graph undirected and select only the nodes
		belonging to the largest connected component.

		:param adj_matrix: sp.spmatrix
			Sparse adjacency matrix
		:param labels: array-like, shape [n]

		:return:
			standardized_adj_matrix: sp.spmatrix
			Standardized sparse adjacency matrix.
			standardized_labels: array-like, shape [?]
			Labels for the selected nodes.
		"""
		# copy the input
		standardized_adj_matrix = self.Adjacency.copy()

		# make the graph unweighted
		standardized_adj_matrix[standardized_adj_matrix != 0] = 1

		# make the graph undirected
		standardized_adj_matrix = standardized_adj_matrix.maximum(standardized_adj_matrix.T)

		# select the largest connected component
		_, components = connected_components(standardized_adj_matrix)
		c_ids, c_counts = np.unique(components, return_counts=True)
		id_max_component = c_ids[c_counts.argmax()]
		select = components == id_max_component
		standardized_adj_matrix = standardized_adj_matrix[select][:, select]
		if self.labeled:
			standardized_labels = self.Labels[select]
		else:
			standardized_labels = None

		# remove self-loops
		standardized_adj_matrix = standardized_adj_matrix.tolil()
		standardized_adj_matrix.setdiag(0)
		standardized_adj_matrix = standardized_adj_matrix.tocsr()
		standardized_adj_matrix.eliminate_zeros()

		self.Adjacency, self.Labels = standardized_adj_matrix, standardized_labels

		return
	def k_core( self, k ):
		"""
		Keeps the k-core of the input graph
		:param k: int 
		"""
		self.setNetworkXGraph()
		core_numbers = nx.core_number( self.G )
		select = [key for key,v in core_numbers.items() if v >=k]
		self.Adjacency = self.Adjacency[select][:, select]
		self.Labels =  self.Labels[select]
		
	def netmf_embedding(self, T, skip_max=False):
		"""
		Calculates the NetMF embedding for the network
		Parameters:
			rank (int): Low-rank approximation
			T (int): Optimization Window 
		""" 
		# Calculate embedding

		n = self.Adjacency.shape[0]
		lap, deg_sqrt = sp.sparse.csgraph.laplacian(self.Adjacency, normed=True, return_diag=True)
		lam, W = np.linalg.eigh((sp.sparse.identity(n) - lap).todense())
		perm = np.argsort(-np.abs(lam))
		lam, W = lam[perm], W[:,perm]
	
		deg_inv_sqrt_diag = sp.sparse.spdiags(1./deg_sqrt, 0, n, n)
		vol = self.Adjacency.sum()
		lam_trans = sp.sparse.spdiags(lam[1:] * (1-lam[1:]**T) / (1-lam[1:]), 0, n-1, n-1)
		if skip_max:
			self.Embedding =  np.log(1 + vol/T * deg_inv_sqrt_diag @ W[:,1:] @ lam_trans @ W[:,1:].T @ deg_inv_sqrt_diag)
		else:
			self.Embedding =  np.log(np.maximum(1., 1 + vol/T * deg_inv_sqrt_diag @ W[:,1:] @ lam_trans @ W[:,1:].T @ deg_inv_sqrt_diag))
		return

	def low_rank_embedding( self, rank ):
		# Low-rank approximation
		w, v = np.linalg.eigh( self.Embedding )
		order = np.argsort(np.abs(w))[::-1]
		w, v = w[order[:rank]], v[:,order[:rank]]
		self.LR_Embedding = v @ np.diag(w) @ v.T
		
		return

	def closenessCentrality( self ):
		cc = nx.closeness_centrality( self.G )
		return cc
	
	def pageRank( self ):
		pr_vector = nx.pagerank( self.G )
		return pr_vector

	def getAdjacency( self ):
		return self.Adjacency

	def get_LR_embedding( self ):
		return self.LR_Embedding

	def setNetworkXGraph( self ):
		self.G = nx.from_scipy_sparse_matrix(self.Adjacency)
		self.G.remove_edges_from(nx.selfloop_edges(self.G))
		
		return
		
	def getNetworkXGraph( self ):
		if not self.G:
			self.setNetworkXGraph()
		return self.G

	def getNodesVolume( self ):
		return self.Adjacency.shape[0], np.array(self.Adjacency.sum(axis=1)).flatten().sum()

	def getLabels( self ):
		return self.Labels
	
	def isLabeled( self ):
		return self.labeled