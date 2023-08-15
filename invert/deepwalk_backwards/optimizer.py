import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
import scipy.sparse, scipy.io, scipy.optimize
from scipy.special import expit
import torch
import math


class Optimizer:
	def __init__(self, adjacency, embedding, fname='Spectral_Embedding',device=torch.device("cpu"), dtype=torch.double):
		self.n = adjacency.shape[0]
		self.rank = str(embedding.shape[1])
		self.device = device
		self.dtype = dtype
		self.shift = 0.
		self.filename = fname
		deg = np.array(adjacency.sum(axis=1)).flatten()
		self.vol = deg.sum()
		self.deg = torch.tensor(deg, device=self.device, dtype=self.dtype, requires_grad=False)
		self.adjacency = torch.tensor( adjacency.todense(), device=self.device, dtype=self.dtype, requires_grad=False)
		self.embedding = embedding @ embedding.T
		print(f"adj.shape: {adjacency.shape} | emb.shape: {embedding.shape}")
		self.pmi = torch.tensor(self.embedding, device=self.device, dtype=self.dtype, requires_grad=False)


	def learnNetwork( self, max_iter=50, method='autoshift' ):
		
		# FUNCTIONS
		def pmi_loss_10_elt_param(elts, n, logit_mode='raw', vol=0., skip_max=False, given_edges=False ):
			elts_tensor = torch.tensor(elts, device=self.device, dtype=self.dtype, requires_grad=True)
			adj_recon = torch.zeros(n,n, device=self.device, dtype=self.dtype)
			if logit_mode == 'individual':
				if not given_edges:
					adj_recon[np.triu_indices(n,1)] = torch.sigmoid(elts_tensor)
				else:
					adj_recon[np.triu_indices(n,1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges, activation=True, shift=0.)
			elif logit_mode == 'raw':
				if not given_edges:
					adj_recon[np.triu_indices(n,1)] = elts_tensor
				else:
					adj_recon[np.triu_indices(n,1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges, activation=False)
			elif logit_mode == 'softmax':
				adj_recon[np.triu_indices(n,1)] = torch.nn.functional.softmax(elts_tensor, dim=0) * (vol/2)
			elif logit_mode == 'autoshift':
				self.shift = 0.
				for i in range(10):
					self.shift = self.shift - (torch.sigmoid(elts_tensor+self.shift).sum() - (vol/2)) / (torch.sigmoid(elts_tensor+self.shift) * (1. - torch.sigmoid(elts_tensor+self.shift))).sum()
				if not given_edges:
					adj_recon[np.triu_indices(n,1)] = torch.sigmoid(elts_tensor+self.shift)
				else:
					adj_recon[np.triu_indices(n,1)] = returnLearnedEdges(n, adj_recon, elts_tensor, given_edges, activation=True, shift=self.shift)
		
			adj_recon = adj_recon + adj_recon.T
			deg_recon = adj_recon.sum(dim=0)
			vol_recon = deg_recon.sum()

			p_recon = (1. / deg_recon)[:,np.newaxis] * adj_recon
			p_recon_2 = p_recon @ p_recon
	
			p_recon_5 = (p_recon_2 @ p_recon_2) @ p_recon
			p_geo_series_recon = ( ((p_recon + p_recon_2) @ (torch.eye(n) + p_recon_2)) + p_recon_5 ) @ (torch.eye(n) + p_recon_5)
				
			if skip_max:
				pmi_recon_exact = torch.log((vol_recon/10.) * p_geo_series_recon * (1. / deg_recon)[np.newaxis,:])
			else:
				pmi_recon_exact = torch.log(torch.clamp((vol_recon/10.) * p_geo_series_recon * (1. / deg_recon)[np.newaxis,:], min=1.))
			loss_pmi = (pmi_recon_exact - self.pmi).pow(2).sum() / (self.pmi).pow(2).sum()
			loss_deg = (deg_recon - self.deg).pow(2).sum() / self.deg.pow(2).sum()
			loss_vol = (vol_recon - self.vol).pow(2) / (self.vol**2)
	
			loss = loss_pmi
			print('{}. Loss: {}\t PMI: {}\t Vol: {}\t Deg: {:.2f}'.format(self.iter_num, math.sqrt( loss.item() ), math.sqrt( loss_pmi.item() ), loss_vol.item(), loss_deg.item()))
			loss.backward()
			with torch.no_grad():
				if torch.isnan(loss):
					pass
			gradients = elts_tensor.grad.numpy().flatten()
			return loss, gradients

		def callback_elt_param(x_i):
			self.elts = x_i
			self.iter_num += 1
			# if self.iter_num % 5 == 0:
			# 	np.save( 'adj_recon/' +  self.filename + '_' + self.rank +'_recon_elts.npy', expit(self.elts + self.shift.detach().numpy()))
		
		# MAIN OPTIMIZATION
		np.random.seed()
		self.elts = np.random.uniform(0,1, size=(self.n*self.n-self.n) // 2 )
		self.iter_num = 0
		self.elts *= 0
		res = scipy.optimize.minimize(pmi_loss_10_elt_param, x0=self.elts, 
							  args=(self.n,'autoshift',self.vol, False), jac=True, method='L-BFGS-B',
							 callback=callback_elt_param,
							  tol=np.finfo(float).eps, 
								  options={'maxiter':max_iter, 'ftol':np.finfo(float).eps, 'gtol':np.finfo(float).eps}
							 )
		return expit(self.elts + self.shift.detach().numpy())