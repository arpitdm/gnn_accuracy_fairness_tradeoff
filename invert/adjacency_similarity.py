import numpy as np
import scipy.sparse as sp
from debias.utils import pw_dists


def run_Adjacency_Similarity(args, data, U):
	# prepare
	n = U.shape[0]
	m = int((data.A.sum()+n)/ 2)
	m_transf = 0
	A_transf = np.identity(n).astype(np.int32) # self-loops
	
	# load pairwise distances
	if args.emb_type == 'original':
		D_U_pw = pw_dists(args.emb_savepath, args.emb_str, U, args.recompute_debiasing)
	elif args.emb_type == 'debiased':
		D_U_pw = pw_dists(data.U_debias_savepath, data.U_debias_param_str, U, args.recompute_debiasing)

	# set each node to be far away from itself
	dmax = np.max(D_U_pw) + 1.0
	for i in range(n):
		D_U_pw[i,i] = dmax

	if args.as_create_method == 'equal_num_edges':
		iu1 = np.triu_indices(n)
		D_U_pw[iu1] = dmax
		I_m, J_m = np.unravel_index(np.argpartition(D_U_pw, kth=m-n, axis=None)[:(m-n)], (n,n)) # num_edges - n_self_loops
		A_transf[I_m,J_m] = 1
		A_transf[J_m,I_m] = 1
		assert (A_transf == A_transf.T).all()
	elif args.as_create_method == 'soft_consistency':
		degrees = np.array(np.sum(data.A, axis=0)).flatten().astype(np.int32)
		D_argsort = np.argsort(D_U_pw, axis=1).astype(np.int32).tolist()
		completed = np.zeros(n)
		curr_deg = np.zeros(n)

		for r in range(args.rounds): # loop over n nodes for r rounds
			for i in range(n): # for each node
				if not completed[i]: # if deg[i] edges not already added
					# get current deg[i] closest neighbors of i in embd
					neighbors_i_r = D_argsort[i][:degrees[i]]
					for j in neighbors_i_r: # for each neighbor j
						neighbors_j_r = D_argsort[j][:degrees[j]]
						# check if j's orig_deg is not yet achieved
						# if not, check if i in deg[j] closest neighbors of j
						if not completed[j] and i in neighbors_j_r:
							A_transf[i,j] = 1 # add undirected edge
							A_transf[j,i] = 1
							m_transf += 1
							if m_transf >= m: # if n_edges achieved, stop
								break
							# remove i from j's deg[j] closest neighbors
							neighbors_j_r.remove(i)
							neighbors_i_r.remove(j) # remove j from i's nbrs
							# update current degrees of i and j
							curr_deg[i] += 1
							curr_deg[j] += 1
							# if deg[i] edges have been added to i, mark & exit
							if curr_deg[i] >= degrees[i]:
								completed[i] = 1
								break
							# if deg[j] edges have been added to j, mark it
							if curr_deg[j] >= degrees[j]:
								completed[j] = 1


	print(f'Original Num Edges: {m} | {args.emb_type} Inverted Edges: {(A_transf.sum()/2)}')
	return A_transf
