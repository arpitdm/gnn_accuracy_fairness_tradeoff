import argparse


def parse_dataset_args(parser):
	dataset_general = parser.add_argument_group(title='dataset_general')
	dataset_general.add_argument('--compute_dataset_stats', action="store_true",
						default=False,
						help="whether to compute dataset statistics or not")

	dataset = parser.add_argument_group(title='dataset')
	dataset.add_argument('--dataset_name', type=str,
						default='German',
						choices=['German', 'Credit', 'Penn94', 'Region-Z'])

	dataset_train = parser.add_argument_group(title='dataset_train')
	dataset_train.add_argument('--dataset_seed', type=int,
						default=42,
						help='Random dataset seed.')
	dataset_train.add_argument('--train_size', type=float,
						default=0.2,
						help='Fraction of nodes to be used for training.')
	dataset_train.add_argument('--val_size', type=float,
						default=0.2,
						help='Fraction of nodes to be used for validation.')

	synthetic = parser.add_argument_group(title='synthetic')
	synthetic.add_argument('--n_nodes', type=int,
						default=100,
						help='Number of nodes.')

	return parser

	
def parse_embedding_args(parser):
	embed_general = parser.add_argument_group(title='embed_general')
	embed_general.add_argument('--recompute_embedding', action="store_true",
						default=False,
						help="whether to recompute embedding")
	embed_general.add_argument('--compute_embedding_stats', action="store_true",
						default=False,
						help="compute statistics of embedding")

	embedding = parser.add_argument_group(title='embedding')
	embedding.add_argument('--embed_algo', type=str,
						default='Spectral_Embedding',
						choices=['Spectral_Embedding', 'DeepWalk', 'GOSH'])
	embedding.add_argument('--dim', type=int,
						default="128",
						help="Number of embedding dimensions.")

	spectral_embedding = parser.add_argument_group(title='spectral_embedding')
	spectral_embedding.add_argument('--maxiter', type=int,
						default=1e4,
						help='Maximum number of Arnoldi iterations.')
	spectral_embedding.add_argument('--tol', type=float,
						default=1e-6,
						help='Relative accuracy for eigenvalues.')

	deepwalk = parser.add_argument_group(title='deepwalk')
	deepwalk.add_argument('--walk_length', type=int,
							default=80,
							help='Length of walk per source.')
	deepwalk.add_argument('--num_walks', type=int,
							default=10,
							help='Number of walks per source.')
	deepwalk.add_argument('--context_size', type=int,
							default=10,
							help='Context size for optimization.')
	deepwalk.add_argument('--dw_epochs', type=int,
							default=1,
							help='Number of epochs in SGD.')

	gosh = parser.add_argument_group(title='gosh')
	gosh.add_argument('--gosh_epochs', type=int,
							default=200,
							help='Number of epochs to run on the entirety of the graph.')
	gosh.add_argument('--gosh_s', type=int,
							default=3,
							help='Number of negative samples used with every positive update.')
	gosh.add_argument('--gosh_neg_wt', type=float,
							default=1,
							help='Scaling factor for gradients.')
	gosh.add_argument('--gosh_lr', type=float,
							default=0.025,
							help='Global learning rate of the model.')
	return parser


def parse_pretrain_args(parser):
	pretrain_general = parser.add_argument_group(title='pretrain_general')
	pretrain_general.add_argument('--compute_debias_stats',
						action="store_true",
						default=False,
						help="whether to compute comparative statistics about original and debiased data.")
	pretrain_general.add_argument('--recompute_debiasing', action="store_true",
						default=False,
						help="whether or not to rerun debias procedure.")

	pretrain = parser.add_argument_group(title='pretrain')
	pretrain.add_argument('--pretrain_algo', type=str,
						default="original",
						choices=['original', 'unaware','EDITS', 'PFR'],
						help="Name of data debiasing algorithm.")
	pretrain.add_argument('--debias_X', type=int,
						default=0,
						help="Choose original/debiased node attributes.")
	pretrain.add_argument('--debias_A', type=int,
						default=0,
						help="Choose original/debiased adjacency matrix.")

	pfr = parser.add_argument_group(title='pfr')
	pfr.add_argument('--pfr_k', type=int,
						default=2,
						help="number of latent dimensions of PFR.")
	pfr.add_argument('--pfr_quantiles', type=int,
						default=4,
						help="hyperparameter controlling number of quantiles for k-quantile graph of PFR.")
	pfr.add_argument('--pfr_nn_k', type=int,
						default=10,
						help="number of nearest neighbors of node for similarity matrix for PFR.")
	pfr.add_argument('--pfr_t', type=int,
						default=2,
						help="scaling parameter for similarity matrix for PFR.")
	pfr.add_argument('--pfr_gamma', type=float,
						default=1.0,
						help="hyperparam controlling the influence of W^F.")
	pfr.add_argument('--pfr_q', type=float,
						default=0.5,
						help="hyperparam controlling the q-quantile for creating the between group quantile graph, i.e. W^F.")
	pfr.add_argument('--pfr_A_k', type=int,
						default=2,
						help="number of latent dimensions of PFR.")
	pfr.add_argument('--pfr_A_quantiles', type=int,
						default=4,
						help="hyperparameter controlling number of quantiles for k-quantile graph of PFR.")
	pfr.add_argument('--pfr_A_nn_k', type=int,
						default=10,
						help="number of nearest neighbors of node for similarity matrix for PFR.")
	pfr.add_argument('--pfr_A_t', type=int,
						default=2,
						help="scaling parameter for similarity matrix for PFR.")
	pfr.add_argument('--pfr_A_gamma', type=float,
						default=1.0,
						help="hyperparam controlling the influence of W^F.")
	pfr.add_argument('--pfr_A_q', type=float,
						default=0.5,
						help="hyperparam controlling the q-quantile for creating the between group quantile graph, i.e. W^F.")

	edits = parser.add_argument_group(title='edits')
	edits.add_argument('--edits_epochs', type=int,
						default=50,
						help='Number of training epochs.')
	edits.add_argument('--edits_lr', type=float,
						default=3e-3,
						help='Initial learning rate.')
	edits.add_argument('--edits_weight_decay', type=float,
						default=1e-7,
						help='Weight decay (L2 loss on parameters).')
	edits.add_argument('--edits_nfeat_out', type=int,
						default=10,
						help='Number of hidden units is set to nfeats_original/edits_nfeat_out. That is, if original is 120 and edits nfeat_out is 10, then EDITS builds 12.')
	edits.add_argument('--edits_dropout', type=float,
						default=0.2,
						help='Dropout rate (1 - keep probability).')
	edits.add_argument('--edits_adj_lambda', type=float,
						default=1e-1,
						help='Adjacency lambda.')
	edits.add_argument('--edits_layer_threshold', type=float,
						default=2,
						help='Layer threshold.')

	return parser


def parse_invert_args(parser):
	invert_general = parser.add_argument_group(title='invert_general')
	invert_general.add_argument('--compute_invert_stats', action="store_true",
						default=False,
						help="whether or not to compute statistics of original and transformed graph.")
	invert_general.add_argument('--invert_logfilename', type=str, 
						default='invert_results.csv',
						help="logfile name for results of embedding inversion.")
	invert_general.add_argument('--recompute_inversion', action="store_true",
						default=False,
						help="whether or not to recompute embedding inversion.")

	invert = parser.add_argument_group(title='invert')
	invert.add_argument('--invert_algo', type=str,
						default='DW_Backwards',
						choices=['DW_Backwards', 'Adjacency_Similarity'])

	dw_back = parser.add_argument_group(title='dw_backwards')
	dw_back.add_argument('--edge_create_method', type=str,
						default='coin_toss',
						choices=['coin_toss', 'add_edge', 'maxst', 'threshold', 'coin_toss2'])
	dw_back.add_argument('--invert_maxiter', type=int,
						default=20,
						help="number of iterations.")

	as_back = parser.add_argument_group(title='adjacency_similarity')
	as_back.add_argument('--as_create_method', type=str,
						default='equal_num_edges',
						choices=['equal_num_edges', 'soft_consistency'])
	as_back.add_argument('--rounds', type=int,
						default=1,
						help='rounds for soft_consistency')
	return parser


def parse_intrain_args(parser):
	intrain = parser.add_argument_group(title='intrain')
	intrain.add_argument('--intrain_algo', type=str,
						default="NIFTY",
						choices=['NIFTY'],
						help="Name of model debiasing algorithm.")

	nifty = parser.add_argument_group(title='nifty')
	nifty.add_argument('--encoder_name', type=str,
						default='GCN',
						choices=['GCN', 'GraphSAGE', 'GIN'])
	nifty.add_argument('--proj_hidden', type=int,
						default=16,
						help='Number of hidden units in the projection layer of encoder.')
	nifty.add_argument('--drop_edge_rate_1', type=float,
						default=0.1,
						help='Drop edge for first augmented graph.')
	nifty.add_argument('--drop_edge_rate_2', type=float,
						default=0.1,
						help='Drop edge for second augmented graph.')
	nifty.add_argument('--drop_feature_rate_1', type=float,
						default=0.1,
						help='Drop feature for first augmented graph.')
	nifty.add_argument('--drop_feature_rate_2', type=float,
						default=0.1,
						help='Drop feature for second augmented graph.')
	nifty.add_argument('--sim_coeff', type=float,
						default=0.5,
						help='Regularization coeff for the self-supervised task.')
	return parser


def parse_posttrain_args(parser):
	posttrain = parser.add_argument_group(title='posttrain')
	posttrain.add_argument('--posttrain_algo', type=str,
						default="Blackbox-Pred",
						choices=['Blackbox-Pred'],
						help="Name of inference debiasing algorithm.")
	posttrain.add_argument('--flip_frac', type=float,
						default=0.1,
						help="Fraction of predictions to change.")
	return parser


def parse_experiment_args(parser):
	experiment_general = parser.add_argument_group(title='experiment_general')
	experiment_general.add_argument('--verbose', type=int,
						default=0,
						help='Enables verbose logs of experiment.')
	experiment_general.add_argument('--save_model', type=int,
						default=1,
						help='Save trained model to disk.')
	experiment_general.add_argument('--wandb', action="store_true",
						default=False,
						help="whether to turn on wandb logging or not")
	experiment_general.add_argument('--bestval', type=int,
						default=0,
						help="whether to test for model that achieves best score on validation set during training or use the model obtained at the end of training.")
	experiment_general.add_argument('--no-cuda', action="store_true",
						default=False,
						help="Disables CUDA training.") 
	experiment_general.add_argument('--exp_logfilename', type=str, 
						default='experiment_results.csv',
						help="logfile name for experiments")
	experiment_general.add_argument('--loglevel', type=str, 
						default='INFO',
						help="logging level")
	experiment_general.add_argument('--retrain_model', action="store_true",
						default=False,
						help="whether or not to retrain model.")
	experiment_general.add_argument('--device', type=str,
						default='',
						choices=['cpu', 'cuda', ''])
	experiment_general.add_argument('--trials', type=int,
						default=10,
						help='num trials for blackbox within a particular seeded model')

	train = parser.add_argument_group(title='train')
	train.add_argument('--locus', type=str,
						default='pretrain',
						choices=['pretrain','intrain','posttrain'],
						help="location where the fairness-inducing intervention occurs.")
	train.add_argument('--model_name', type=str,
						default='GCN',
						choices=['GCN', 'GraphSAGE', 'GIN'])
	train.add_argument('--seed', type=int,
						default=0,
						help='Random seed.')

	gnn = parser.add_argument_group(title='gnn')
	gnn.add_argument('--epochs', type=int,
						default=5,
						help='Number of training epochs.')
	gnn.add_argument('--lr', type=float,
						default=1e-3,
						help='Initial learning rate.')
	gnn.add_argument('--weight_decay', type=float,
						default=1e-5,
						help='Weight decay (L2 loss on parameters).')
	gnn.add_argument('--hidden', type=int,
						default=16,
						help='Number of hidden units.')
	gnn.add_argument('--dropout', type=float,
						default=0.5,
						help='Dropout rate (1 - keep probability).')

	random_forest = parser.add_argument_group(title='random_forest')
	random_forest.add_argument('--n_estimators', type=int,
						default=1000,
						help='Number of estimators.')
	random_forest.add_argument('--max_depth', type=int,
						default=2000,
						help='Maximum depth.')
	return parser