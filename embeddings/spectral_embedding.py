from pathlib import Path

import numpy as np
import networkx as nx
import scipy.sparse as sp


def create_Spectral_Embedding(args, data, A):
	# load eigenvectors
	if Path(data.U_fname).exists() and not args.recompute_embedding:
		U = np.load(data.U_fname)
	# compute eigenvectors
	else:
		G = nx.from_numpy_array(A)
		L = nx.laplacian_matrix(G).todense().astype(np.float32)
		# uses the shift-invert mode
		w, U = sp.linalg.eigsh(L,k=args.dim+1,sigma=-1.0,which='LM',maxiter=args.maxiter,tol=args.tol)
		U = U[:,1:]
		np.save(data.U_fname, U)

	return U