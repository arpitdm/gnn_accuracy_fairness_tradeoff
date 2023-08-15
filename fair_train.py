import time
import wandb
import torch
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from utils import *
from parse_args import *
from load_dataset import load_dataset

from models.gcn import train_GCN, GCN
from models.graphsage import train_GraphSAGE, SAGE
from models.gin import train_GIN, GIN
from models.nifty import train_NIFTY
from pretrain_interventions import *
from posttrain_blackbox import *


def parse_all_args():
	parser = argparse.ArgumentParser()
	parser = parse_dataset_args(parser)
	parser = parse_experiment_args(parser)
	parser = parse_embedding_args(parser)
	parser = parse_pretrain_args(parser)
	parser = parse_invert_args(parser)
	parser = parse_intrain_args(parser)
	parser = parse_posttrain_args(parser)
	args = parser.parse_args()
	# args.parser = parser
	return args, parser


def preprocess_data(args, data):    
	if args.pretrain_algo == 'original':
		print("original")
		# return data.X_original[:,-20:], data.edge_index_original
		return data.X_original, data.edge_index_original
	# unaware only removes the sensitive attribute column which is placed at the end of the feature matrix
	elif args.pretrain_algo == 'unaware':
		print("unaware")
		args.debias_X = 1
		return data.X_original[:, :-1], data.edge_index_original
		
	elif args.pretrain_algo == 'PFR':
		print(args.pretrain_algo)
		# return X_debias, edge_index_debias = 
		return pretrain_PFR(args, data)

	elif args.pretrain_algo == 'EDITS':
		print(args.pretrain_algo)
		return pretrain_EDITS(args, data)

def main():
	warnings.filterwarnings("ignore")

	# parse arguments
	args,parser = parse_all_args()
 	
	if args.wandb:
		wandb.init(project="postprocess_GNNs", entity="arpitdm")
		wandb.config = vars(args)
	else:
		wandb.init(mode="disabled")

	args.parser = parser

	# load dataset
	data = load_dataset(args)

	# set device
	if args.device == '':
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		args.device = device
	print(f'Device: {args.device}')
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	
	# seeds and determinism
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.allow_tf32 = False
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	t = time.time()
	
	# prepare data for training
	if args.locus == "pretrain":
		print("Debiasing Data (Pre-train)")
		print(f"Pretrain-Algo: {args.pretrain_algo}")
		data.features, data.edge_index = preprocess_data(args,data)
		print(f"Debias Node Attributes: {args.debias_X}")
		print(f"Debias Graph: {args.debias_A}")
		print(f"Model: {args.model_name}")
	elif args.locus == "intrain":
		print("Debiasing Model (In-train)")
		print(f"Model: {args.intrain_algo}-{args.encoder_name}")
		data.features = data.X_original
		data.edge_index = data.edge_index_original
	elif args.locus == "posttrain":
		print("Debiasing Predictions (Post-train)")
		data.features = data.X_original
		data.edge_index = data.edge_index_original

	# set labels vector
	data.labels = data.Y_original
	data.n_classes = data.labels.unique().shape[0] - 1
	data.n_features = data.features.shape[1]

	# convert to device
	data.features = data.features.to(args.device)
	data.sens = data.sens.to(args.device)
	data.edge_index = data.edge_index.to(args.device)
	data.labels = data.labels.to(args.device)
	data.idx_train = torch.LongTensor(data.idx_train).to(args.device)
	data.idx_val = torch.LongTensor(data.idx_val).to(args.device)
	data.idx_test = torch.LongTensor(data.idx_test).to(args.device)

	# get model parameters
	data.model_param_str, data.model_fname = get_model_strings(args, data)

	# run experiment
	if args.locus == 'pretrain':
		# train model
		if args.model_name == "GCN":
			val_auc_roc, avg_epoch_time, model = train_GCN(args, data)

		elif args.model_name == "GraphSAGE":
			val_auc_roc, avg_epoch_time, model = train_GraphSAGE(args, data)

		elif args.model_name == "GIN":
			val_auc_roc, avg_epoch_time, model = train_GIN(args, data)

		test_preds, test_labels, test_output, test_sens = eval_GNN(args, data, model)
		test_auc_roc, f1_s, parity, equality = compute_perf_metrics(test_preds, test_labels, test_output, test_sens)

	elif args.locus == 'intrain':
		# train model
		if args.intrain_algo == "NIFTY":
			val_auc_roc, avg_epoch_time, model = train_NIFTY(args, data)

		test_preds, test_labels, test_output, test_sens = eval_GNN(args, data, model)
		test_auc_roc, f1_s, parity, equality = compute_perf_metrics(test_preds, test_labels, test_output, test_sens)

	# update predictions after model training
	if args.locus == 'posttrain':
		# load original model trained on original data
		args.locus = 'pretrain'
		data.model_param_str, data.model_fname = get_model_strings(args, data)
		args.locus = 'posttrain'

		if args.model_name == 'GCN':
			model = GCN(nfeat=data.n_features,
						nhid=args.hidden,
						nclass=data.n_classes,
						dropout=args.dropout)
		elif args.model_name == 'GraphSAGE':
			model =	SAGE(nfeat=data.n_features,
						nhid=args.hidden,
						nclass=data.n_classes,
						dropout=args.dropout)    
		elif args.model_name == 'GIN':
			model =	GIN(nfeat=data.n_features,
						nhid=args.hidden,
						nclass=data.n_classes,
						dropout=args.dropout)    
		model = model.to(args.device)
		model.load_state_dict(torch.load(data.model_fname))
		val_auc_roc, avg_epoch_time = -1, -1
  
		if args.posttrain_algo == "Blackbox-Pred":
			test_preds, test_labels, test_output, test_sens = eval_GNN(args, data, model)
			test_auc_roc, f1_s, parity, equality, test_output, test_preds = run_BlackboxPred(args, data, test_preds, test_labels, test_output, test_sens)
   
	total_runtime = time.time() - t

	# log test outputs and predictions
	test_outcomes = np.stack([test_labels, test_sens, test_output[:,0], test_preds],axis=1)
	print(f"outcomes.shape: {test_outcomes.shape}")
	outcomes_fname = get_outcomes_strings(args, data, key=0)
	print(f"Outcomes File-name: {outcomes_fname}")
	np.save(outcomes_fname, test_outcomes)
	# np.save("changed_threshold_credit_gcn_5.npy", test_outcomes)

	# write results to log-file
	res_vals = create_param_str(args,sep=',',object='result')+f",{val_auc_roc:.4f},{test_auc_roc:.4f},{f1_s:.4f},{parity:.4f},{equality:.4f},{avg_epoch_time:.4f},{total_runtime:.4f}"
	res_keys = create_param_str(args,sep=',',object='result',key=1)+f",val_auc_roc,auc_roc,f1_s,parity,equality,avg_epoch_time,total_runtime"
	logpath = Path(f'results/{args.dataset_name}')
	logpath.mkdir(parents=True, exist_ok=True)
	logfname = logpath / args.exp_logfilename
	logging.basicConfig(filename=logfname, level=getattr(logging, args.loglevel.upper()))
	logging.info(res_keys)
	logging.info(res_vals) 
	print(f"Model fname: {data.model_fname}")
	print(f"Logfile: {logfname.as_posix()}")
 
	# print summary of results
	print(f"Dataset: {args.dataset_name}")
	print(f"Locus: {args.locus}")
	print(f"AUC-ROC: {test_auc_roc}")
	print(f"F1-Score: {f1_s}")
	print(f"Parity: {parity}")
	print(f"Equality: {equality}")

	# wandb logging
	wandb.log({"Epochs": args.epochs})
	wandb.log({"AUC-ROC": test_auc_roc})
	wandb.log({"F1-Score": f1_s})
	wandb.log({"Parity": parity})
	wandb.log({"Equality": equality})
 
if __name__ == '__main__':
	main()