import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from utils import fair_metric


def run_BlackboxPred(args, data, test_preds, test_labels, test_output, test_sens):
	# sensitive class is denoted by 1
	sens_val = 1

	# get all nodes who belong to protected class
	test_idx_sens_node = np.where(test_sens==sens_val)[0]
	print(f"Num-nodes of in test set: {test_labels.shape[0]}")
	print(f"Num-nodes of protected class in test set: {test_idx_sens_node.shape[0]}")
	
	# get all nodes who have been assigned a "negative" label
	test_idx_pred_neg = np.where(test_preds==0)[0]
	print(f"Num-nodes of `negative` label in test set: {test_idx_pred_neg.shape[0]}")
	print(test_idx_pred_neg.shape)
	
	# get all sens nodes who have been assigned "negative" label
	idx_sens_pred_neg = np.intersect1d(test_idx_sens_node, test_idx_pred_neg)

	updated_test_auc_rocs = []
	updated_f1_ss = []
	updated_paritys = []
	updated_equalitys = []
	for i in range(args.trials):
		# randomly select some sens nodes with "negative" labels
		n_flips = int(args.flip_frac*idx_sens_pred_neg.shape[0])
		print(f"num_bad = {idx_sens_pred_neg.shape[0]}, n_flips = {n_flips}")
		np.random.seed(args.seed+i)
		flip_idx = np.random.choice(idx_sens_pred_neg, n_flips, replace=False)
		# print(flip_idx)
	
		# change the "negative" labels to "positive" labels for these nodes
		updated_test_preds = test_preds.copy()
		updated_test_preds[flip_idx] = 1
		# print (np.all(test_preds == updated_test_preds))

		# compute updated f1-scores
		f1_s = f1_score(test_labels, test_preds)
		updated_f1_s = f1_score(test_labels, updated_test_preds)
		updated_f1_ss.append(updated_f1_s)

		# change model output to 0+\epsilon for these nodes
		updated_test_output = test_output.copy()
		updated_test_output[flip_idx] = 100

		# compute updated auc-roc scores
		test_auc_roc = roc_auc_score(test_labels, test_output)
		updated_test_auc_roc = roc_auc_score(test_labels, updated_test_output)
		updated_test_auc_rocs.append(updated_test_auc_roc)
	
		# compute statistical parity and equality of opportunity
		parity, equality = fair_metric(test_preds, test_labels, test_sens)
		updated_parity, updated_equality = fair_metric(updated_test_preds, test_labels, test_sens)
		updated_paritys.append(updated_parity)
		updated_equalitys.append(updated_equality)
	
	updated_test_auc_roc = np.mean(updated_test_auc_rocs)
	updated_f1_s = np.mean(updated_f1_ss)
	updated_parity = np.mean(updated_paritys)
	updated_equality = np.mean(updated_equalitys)
	print(f"AUC-ROC Scores: {test_auc_roc}, {updated_test_auc_roc}")
	print(f"F1-Scores: {f1_s}, {updated_f1_s}")
	print(f"Parity Scores: {parity}, {updated_parity}")
	print(f"Equality Scores: {equality}, {updated_equality}")

	return updated_test_auc_roc, updated_f1_s, updated_parity, updated_equality, updated_test_output, updated_test_preds