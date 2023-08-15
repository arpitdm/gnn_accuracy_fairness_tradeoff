import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def create_param_str(args, sep='_', object='synthetic', key=0):
	if object == 'synthetic':
		if args.dataset_name == 'SBM':
			arg_groups = ['synthetic', 'connected_caveman']
		else:
			arg_groups = ['synthetic', args.dataset_name.lower()]

	elif object == 'debiased_attributes':
		arg_groups = ['pretrain', args.pretrain_algo.lower()]

	elif object == 'original_embedding':
		arg_groups = ['embedding', args.embed_algo.lower()]

	elif object == 'inverted_embedding':
		arg_groups = ['embedding', args.embed_algo.lower(), 'invert', args.invert_algo.lower()]

	elif object == 'debiased_embedding':
		arg_groups = ['embedding', args.embed_algo.lower(), 'pretrain', args.pretrain_algo.lower()]

	elif object == 'debiased_graph':
		if args.pretrain_algo == 'EDITS':
			arg_groups = ['pretrain', args.pretrain_algo.lower()]
		else:
			arg_groups = ['embedding', args.embed_algo.lower(), 'pretrain', args.pretrain_algo.lower(), 'invert', args.invert_algo.lower()]

	elif object == 'model':
		if args.model_name == 'Random_Forest':
			model_type = 'random_forest'
		else:
			model_type = 'gnn'
		arg_groups = ['dataset','dataset_train','train',model_type,args.locus]
		if args.locus == 'pretrain':
			if args.pretrain_algo == 'EDITS':
				arg_groups += ['pretrain', args.pretrain_algo.lower()]
			elif args.pretrain_algo == 'original':
				arg_groups += []
			elif args.pretrain_algo == 'unaware':
				arg_groups += []
			else:
				if args.debias_X and not args.debias_A:
					arg_groups += [args.pretrain_algo.lower()]
				else:
					arg_groups += [args.pretrain_algo.lower(), 'embedding', args.embed_algo.lower(), 'pretrain', args.pretrain_algo.lower(), 'invert', args.invert_algo.lower()]				
				# if args.pretrain_algo in ['iFair', 'PFR']:
					# arg_groups += [args.pretrain_algo.lower()]
		elif args.locus == 'intrain' or args.locus == 'posttrain':
			arg_groups += [getattr(args,f'{args.locus}_algo')]

	elif object == 'result':
		arg_groups = ['dataset', 'dataset_train', 'train', 'gnn', 'random_forest', 'pretrain', 'embedding', 'spectral_embedding',  'deepwalk', 'gosh', 'ifair', 'pfr', 'invert', 'dw_backwards', 'adjacency_similarity', 'intrain', 'nifty', 'posttrain','blackbox']
	
	# initialize
	group_keys = []
	group_vals = []
	a_dict = args.__dict__

	# get values of args in relevant groups
	for g in args.parser._action_groups:
		if g.title in arg_groups:
			# get names of args in group g
			g_arg_names = [action.dest for action in g._group_actions]
			g_keys = sep.join(map(str,g_arg_names))
			group_keys.append(g_keys)
			g_vals = sep.join(map(str,[a_dict[arg] for arg in g_arg_names]))
			group_vals.append(g_vals)
	if key:
		param_str = sep.join(group_keys)
	else:
		param_str = sep.join(group_vals)
	return param_str


def get_embed_algo_strings(args, data, key=0):
	# A_type = whether embedding is of original, inverted, or debiased graph
	embed_param_str = create_param_str(args, object=f'{args.A_type}_embedding')
	embed_savepath = data.data_dir / f'{args.A_type}_embeddings' / args.embed_algo
	embed_savepath.mkdir(parents=True, exist_ok=True)
	embed_fname = (embed_savepath / (embed_param_str + '.npy')).as_posix()
	return embed_param_str, embed_savepath, embed_fname


def get_model_strings(args, data, key=0):
	model_param_str = create_param_str(args, object='model')
	model_savepath = data.data_dir / 'saved_models' / args.model_name
	model_savepath.mkdir(parents=True, exist_ok=True)
	model_fname = (model_savepath / (model_param_str + '.pt')).as_posix()
	return model_param_str, model_fname

def get_outcomes_strings(args, data, key=0):
	model_param_str = create_param_str(args, object='model')
	outcome_savepath = data.data_dir / 'saved_outcomes' / args.model_name
	outcome_savepath.mkdir(parents=True, exist_ok=True)
	outcome_fname = (outcome_savepath / (model_param_str + '_outcomes.npy')).as_posix()
	return outcome_fname

def get_X_debias_strings(args, data, object="debiased_attributes", key=0):
	X_debias_savepath = data.data_dir / "debiased_attributes"
	X_debias_savepath.mkdir(parents=True, exist_ok=True)
	X_debias_param_str = create_param_str(args, object=object, key=key)
	X_debias_fname = X_debias_savepath / (X_debias_param_str+".npy")
	print(f"Debiased Attribute savefile-name: {X_debias_fname}")
	return X_debias_param_str, X_debias_savepath, X_debias_fname


def get_U_debias_strings(args, data, object="debiased_embedding", key=0):
	U_debias_savepath = data.data_dir / "debiased_embeddings"
	U_debias_savepath.mkdir(parents=True, exist_ok=True)
	U_debias_param_str = create_param_str(args, object=object)
	U_debias_fname = U_debias_savepath / (U_debias_param_str+".npy")
	print(f"Debiased Embedding savefile-name: {U_debias_fname}")
	return U_debias_param_str, U_debias_savepath, U_debias_fname


def get_A_debias_strings(args, data, object="debiased_graph", key=0):
	A_debias_savepath = data.data_dir / "debiased_graphs"
	A_debias_savepath.mkdir(parents=True, exist_ok=True)
	A_debias_param_str = create_param_str(args, object=object)
	A_debias_fname = A_debias_savepath / (A_debias_param_str+".npy")
	print(f"Debiased Graph savefile-name: {A_debias_fname}")
	return A_debias_param_str, A_debias_savepath, A_debias_fname


def fair_metric(pred, labels, sens):
	idx_s0 = sens==0
	idx_s1 = sens==1
	idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
	idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
	parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
	equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
	return parity.item(), equality.item()


def eval_GNN(args, data, model):
	if args.locus in ['pretrain','posttrain'] and args.model_name != 'Random_Forest':
		model.eval()
		output = model(data.features, data.edge_index)
		output_preds = (output.squeeze()>0).type_as(data.labels)
		preds = output_preds[data.idx_test].cpu().numpy()
		labels = data.labels[data.idx_test].cpu().numpy()
		output = output[data.idx_test].detach().cpu().numpy()
		sens = data.sens[data.idx_test].cpu().numpy()
		
	elif args.locus == 'intrain' and args.intrain_algo == 'NIFTY':
		model.eval()
		emb = model(data.features, data.edge_index)
		output = model.predict(emb)
		output_preds = (output.squeeze()>0).type_as(data.labels)
		preds = output_preds[data.idx_test].cpu().numpy()
		labels = data.labels[data.idx_test].cpu().numpy()
		output = output[data.idx_test].detach().cpu().numpy()
		sens = data.sens[data.idx_test].cpu().numpy()
			
	return preds, labels, output, sens


def compute_perf_metrics(test_preds, test_labels, test_output, test_sens): 
	# compute accuracy scores
	test_auc_roc = roc_auc_score(test_labels, test_output)
	f1_s = f1_score(test_labels, test_preds)
	
	# compute statistical parity and equality of opportunity
	parity, equality = fair_metric(test_preds, test_labels, test_sens)

	return test_auc_roc, f1_s, parity, equality

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
