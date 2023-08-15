from scipy.sparse import csc_matrix
from torch.nn.functional import one_hot

from invert.deepwalk_backwards.network_struct import *
from invert.deepwalk_backwards.optimizer import *
from invert.deepwalk_backwards.reconstructor import Reconstructor


def run_DeepWalk_Backwards(args, data, U):
	
	N_orig = Network()
	N_orig.loadNetwork(csc_matrix(data.A), one_hot(data.Y_original).numpy())

	# execute DW backwards optimization on transformed embedding
	P = Optimizer(csc_matrix(data.A), U)
	# if args.emb_type == 'original':
	# 	P = Optimizer(data.A, data.U)
	# 	# P = Optimizer(data.A, data.U, args.invert_maxiter)
	# elif args.emb_type == 'debiased':
	# 	P = Optimizer(data.A, data.U_debias)
	# 	# P = Optimizer(data.A, data.U_debias, args.invert_maxiter)
	elts = P.learnNetwork(max_iter=args.invert_maxiter)

	# reconstruct network from transformed embedding
	N_transf = Reconstructor( *N_orig.getNodesVolume() )
	N_transf.loadNetwork( elts, args.edge_create_method )
	N_transf.setNetworkXGraph()

	# return adjacency matrix of transformed network
	A_transf = N_transf.getAdjacencyBinarized().astype(np.int)
	A_transf = A_transf.astype(np.int)
 
	return A_transf
