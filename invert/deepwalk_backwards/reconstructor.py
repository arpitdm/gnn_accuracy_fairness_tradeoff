import torch
import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.io, scipy.optimize
from scipy.sparse import csgraph, triu
from sklearn import preprocessing

binarize_top = lambda M, num_ones : M >= np.quantile(M, 1 - num_ones / M.size) 


class Reconstructor:
	def __init__( self, number_of_nodes, volume ):
		self.n = number_of_nodes
		self.vol  = volume
		self.Adjacency = None
		self.Adjacency_Binarized = None
		self.G	= None
		self.G_complete = None

	def loadNetwork( self, elts, method="coin_toss", device=torch.device("cpu"), dtype=torch.double):
		elts_tensor = torch.tensor(elts, device=device, dtype=dtype, requires_grad=True)
		adj_recon = torch.zeros(self.n,self.n, device=device, dtype=dtype)
		#shift = 0.
		#for i in range(10):
		#	shift = shift - (torch.sigmoid(elts_tensor+shift).sum() - (self.vol/2)) / (torch.sigmoid(elts_tensor+shift) * (1. - torch.sigmoid(elts_tensor+shift))).sum()
		#	adj_recon[np.triu_indices(self.n,1)] = torch.sigmoid(elts_tensor+shift)
		adj_recon[np.triu_indices(self.n,1)] = elts_tensor
		adj_recon = adj_recon + adj_recon.T
		adj_recon = adj_recon.detach().numpy()
		self.Adjacency_Binarized = np.copy(adj_recon)
		self.Adjacency = scipy.sparse.csc_matrix( adj_recon )
		self.binarize( method ) # Binarize adjacency matrix

	def binarize( self, method="coin_toss" ):
		if method=="add_edge":# DELETE
			# For every node add its highest entry in an effort not to disconnect the graph
			max_row = np.argmax(self.Adjacency_Binarized, axis=1)
			for i in range(len(max_row)):
				self.Adjacency_Binarized[i,max_row[i]] = 1.0
				self.Adjacency_Binarized[max_row[i],i] = 1.0
			self.Adjacency_Binarized = scipy.sparse.csc_matrix( binarize_top(self.Adjacency_Binarized, int(self.vol))).astype('int')
		elif method == "maxst":
			max_st = csgraph.minimum_spanning_tree( -1.*self.Adjacency_Binarized )
			row,col = max_st.nonzero()
			for i in range(len(row)):
				self.Adjacency_Binarized[row[i],col[i]] = 1.0
				self.Adjacency_Binarized[col[i],row[i]] = 1.0
			self.Adjacency_Binarized = scipy.sparse.csc_matrix( binarize_top(self.Adjacency_Binarized, int(self.vol))).astype('int')
		elif method == "threshold":
			max_st = csgraph.minimum_spanning_tree( -1.*self.Adjacency_Binarized )
			row,col = max_st.nonzero()
			for i in range(len(row)):
				self.Adjacency_Binarized[row[i],col[i]] = 1.0
				self.Adjacency_Binarized[col[i],row[i]] = 1.0
			self.Adjacency_Binarized = scipy.sparse.csc_matrix( preprocessing.binarize( self.Adjacency_Binarized, 0.5 ) )
		elif method == "coin_toss2":
			matrix_values = self.Adjacency[np.triu_indices(self.n,1)]
			coin_tosses = np.random.rand(1,self.n*(self.n-1)//2)
			print(matrix_values.shape, coin_tosses.shape)
			outcome = (matrix_values >= coin_tosses).astype(float)
			self.Adjacency_Binarized = np.zeros((self.n,self.n))
			self.Adjacency_Binarized[np.triu_indices(self.n,1)] = outcome
			self.Adjacency_Binarized = self.Adjacency_Binarized + self.Adjacency_Binarized.T
			self.Adjacency_Binarized = scipy.sparse.csc_matrix( self.Adjacency_Binarized )
		elif method == "coin_toss":
			n = self.Adjacency_Binarized.shape[0]
			adj_mst = np.where(csgraph.minimum_spanning_tree(-1*self.Adjacency_Binarized).todense() != 0, 1., 0.)
			adj_mst += adj_mst.T
			adj_probs = np.copy(self.Adjacency_Binarized)
			adj_probs[adj_mst == 1.] = 1.
			adj_probs[adj_mst == 0.] *= self.Adjacency_Binarized.sum() / adj_probs.sum()
			self.Adjacency_Binarized = 1. * (np.random.rand(*adj_probs.shape) < adj_probs)
			self.Adjacency_Binarized[np.triu_indices(n)] = 0
			self.Adjacency_Binarized += self.Adjacency_Binarized.T
			self.Adjacency_Binarized = sp.csr_matrix(self.Adjacency_Binarized)
		return

	def netmf_embedding(self, T, skip_max=False):
		"""
		Calculates the NetMF embedding for the network
		Parameters:
			rank (int): Low-rank approximation
			T (int): Optimization Window 
		"""
		# Calculate embedding

		n = self.Adjacency.shape[0]
		lap, deg_sqrt = sp.csgraph.laplacian(self.Adjacency, normed=True, return_diag=True)
		lam, W = np.linalg.eigh((sp.identity(n) - lap).todense())
		perm = np.argsort(-np.abs(lam))
		lam, W = lam[perm], W[:,perm]

		deg_inv_sqrt_diag = sp.spdiags(1./deg_sqrt, 0, n, n)
		vol = self.Adjacency.sum()
		lam_trans = sp.spdiags(lam[1:] * (1-lam[1:]**T) / (1-lam[1:]), 0, n-1, n-1)
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

	def getAdjacency( self ):
		return self.Adjacency

	def getAdjacencyBinarized( self ):
		return self.Adjacency_Binarized

	def setNetworkXGraph( self ):
		self.G = nx.from_scipy_sparse_matrix(self.Adjacency_Binarized)
		self.G.remove_edges_from(nx.selfloop_edges(self.G))

		return

	def getNetworkXGraph( self ):
		if not self.G:
			self.setNetworkXGraph()
		return self.G

	def setNetworkXComplete( self ):
		self.G_complete = nx.from_scipy_sparse_matrix( triu(self.Adjacency, 1) )
		self.G_complete.remove_edges_from(nx.selfloop_edges(self.G_complete))
		
		return

	def getNetworkXComplete( self  ):
		if not self.G_complete:
			self.setNetworkXComplete()
		return self.G_complete
	
	def closenessCentrality( self ):
		cc = nx.closeness_centrality( self.G )
		return cc

	def pageRank( self, binarized=False, num_iterations=100, d=0.85 ):
		if binarized:
			pr_vector = nx.pagerank( self.G )
		else:
			A  = self.getAdjacency()
			deg = A.sum(axis=0)
			deg = np.array( deg )
			deg_inv = np.diag( 1./deg[0] )
			M = deg_inv @  A
			N = M.shape[1]
			pr_vector = np.random.rand(N, 1)
			pr_vector = pr_vector / np.linalg.norm(pr_vector, 1)
			M_hat = (d * M.T + (1 - d) / N)
			for i in range(num_iterations):
				pr_vector = M_hat @ pr_vector
			pr_vector = dict(zip(range(N), pr_vector))
		return pr_vector