import time
import torch
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix

from debias.orig_PFR import create_orig_PFR_data
from debias.EDITS import *

from utils import get_X_debias_strings, get_U_debias_strings, get_A_debias_strings, get_embed_algo_strings
from utils import sparse_mx_to_torch_sparse_tensor
from create_embedding import load_adjacency, create_embedding
from invert.dw_backwards import run_DeepWalk_Backwards
from invert.adjacency_similarity import run_Adjacency_Similarity
from compute_statistics import compute_debias_statistics


def pretrain_PFR(args, data):
	if args.debias_X:
		data.X_debias_param_str, data.X_debias_savepath, data.X_debias_fname = get_X_debias_strings(args, data)
		# load or create debiased node attributes
		if data.X_debias_fname.exists() and not args.recompute_debiasing:
			X_debias = np.load(data.X_debias_fname)
		else:
			t0 = time.time()
			X_debias = create_orig_PFR_data(args, data, debias="X")
			t1 = time.time()
			print(f"Time for debiasing: {t1-t0}")
			print(f"Shape of original attributes: {data.X_original.numpy().shape}")
			print(f"Shape of debiased attributes: {X_debias.shape}")

			with open(data.X_debias_fname, 'wb') as fp:
				np.save(fp, X_debias)

		if args.compute_debias_stats:
			results_df = compute_debias_statistics(args, data, X_debias, 'X')

		X_debias = torch.FloatTensor(X_debias)
	else:
		X_debias = data.X_original
		
	if args.debias_A:
		# load original embedding
		print("load original embedding")
		print(f"{args.embed_algo}, {args.recompute_embedding}")
		args, data = load_adjacency(args, data)
		data.U_param_str, data.U_savepath, data.U_fname = get_embed_algo_strings(args, data) 
		data.U = create_embedding(args, data)
		
		# load or create debiased node embedding
		print("load debiased embedding")
		data.U_debias_param_str, data.U_debias_savepath, data.U_debias_fname = get_U_debias_strings(args, data)
		if data.U_debias_fname.exists() and not args.recompute_debiasing:
			data.U_debias = np.load(data.U_debias_fname)
		else:
			t0 = time.time() 
			data.U_debias = create_orig_PFR_data(args, data, debias="U")

			t1 = time.time()
			print(f"Time for debiasing: {t1-t0}")

			with open(data.U_debias_fname, 'wb') as fp:
				np.save(fp, data.U_debias)
		
		# invert debiased embedding to graph
		print("load debiased graph")
		data.A_debias_param_str, data.A_debias_savepath, data.A_debias_fname = get_A_debias_strings(args, data)       
		if data.A_debias_fname.exists() and not args.recompute_inversion:
			A_debias = np.load(data.A_debias_fname)
		else:
			if args.invert_algo == 'DW_Backwards':
				args.emb_type = 'debiased'
				data.A = csr_matrix(data.A)
				A_debias = run_DeepWalk_Backwards(args, data, data.U_debias)
				A_debias = A_debias.todense()
			elif args.invert_algo == 'Adjacency_Similarity':
				args.emb_type = 'debiased'
				data.A = csr_matrix(data.A)
				A_debias = run_Adjacency_Similarity(args, data, data.U_debias)
				# A_debias = A_debias.todense()
			with open(data.A_debias_fname, 'wb') as fp:
				np.save(fp, A_debias)
   
		A_debias_sp = csr_matrix(A_debias)
		edge_index_debias,_ = from_scipy_sparse_matrix(A_debias_sp)
	else:
		edge_index_debias = data.edge_index_original
		
	return X_debias, edge_index_debias


def pretrain_EDITS(args, data):
    # get filenames for X_debias and A_debias
	data.X_debias_param_str, data.X_debias_savepath, data.X_debias_fname = get_X_debias_strings(args, data)
	data.A_debias_param_str, data.A_debias_savepath, data.A_debias_fname = get_A_debias_strings(args, data)

	# if said files exist and not recompute then directly load
	if data.X_debias_fname.exists() and data.A_debias_fname.exists() and not args.recompute_debiasing:
		X_debias = np.load(data.X_debias_fname)
		X_debias = torch.FloatTensor(X_debias)
  
		A_debias = np.load(data.A_debias_fname)
		A_debias = sp.csr_matrix(A_debias)
		edge_index_debias,_ = from_scipy_sparse_matrix(A_debias)
	# else run/rerun EDITS
	else:
		# normalize node attributes as per train.py in EDITS
		features_preserve = data.X_original.clone()
		X = data.X_original / data.X_original.norm(dim=0)
		X[:,-1] = data.X_original[:,-1]
	
		# create adjacency matrix as a Sparse FloatTensor as per train.py in EDITS
		args, data = load_adjacency(args, data)
		A = csr_matrix(data.A)
		A = sparse_mx_to_torch_sparse_tensor(A)
		
		# instantiate EDITS object
		edits = EDITS(args, nfeat=X.shape[1], node_num=X.shape[0], nfeat_out=int(X.shape[0]/args.edits_nfeat_out), adj_lambda=args.edits_adj_lambda, layer_threshold=args.edits_layer_threshold, dropout=args.edits_dropout)

		# convert to cuda
		if args.cuda:
			edits.cuda().half()
			A = A.cuda().half()
			X = X.cuda().half()
			features_preserve = features_preserve.cuda().half()
			labels = data.Y_original.cuda().half()
			idx_train = torch.LongTensor(data.idx_train).to(args.device)
			idx_val = torch.LongTensor(data.idx_val).to(args.device)
			idx_test = torch.LongTensor(data.idx_test).to(args.device)
			sens = data.sens.to(args.device)

		# execute edits to debias
		val_adv = []
		test_adv = []
		for epoch in tqdm(range(args.edits_epochs)):
			if epoch > 400:
				args.lr = 0.001
			edits.train()
			edits.optimize(A, X, idx_train, sens, epoch, args.edits_lr)
			A_debias, X_debias, predictor_sens, show, _ = edits(A, X)
			positive_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] > 0)
			negative_eles = torch.masked_select(predictor_sens[idx_val].squeeze(), sens[idx_val] <= 0)
			loss_val = - (torch.mean(positive_eles) - torch.mean(negative_eles))
			val_adv.append(loss_val.data)

		# convert dtypes and formats to be compatible with fair_train.py
		param = edits.state_dict()
		indices = torch.argsort(param["x_debaising.s"])[:4]
		for i in indices:
			features_preserve[:, i] = torch.zeros_like(features_preserve[:, i])
		X_debias = features_preserve
		X_debias = X_debias.float()

		# save to disk
		with open(data.X_debias_fname, 'wb') as fp:
			np.save(fp, X_debias.cpu().numpy())
		with open(data.A_debias_fname, 'wb') as fp:
			np.save(fp, A_debias.detach().cpu().numpy(),allow_pickle=True)
   
		A_debias = sp.csr_matrix(A_debias.detach().cpu().numpy())
		edge_index_debias,_ = from_scipy_sparse_matrix(A_debias)

	# choose whether only X, only A, or both X and A are debiased as per input arguments
	if not args.debias_X:
		X_debias = data.X_original
     
	if not args.debias_A:
		edge_index_debias = data.edge_index_original
     
	return X_debias, edge_index_debias