import torch
import numpy as np
import pandas as pd
from torch import cdist
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def prepare_X_train(data, debias='X'):
	# get sensitive attributes
	sens = data.X_original[:, data.sens_idx].numpy()
	if isinstance(data.sens_idx, int): # if only one sens_attr
		n_sens = 1
		sens = sens[:, np.newaxis]
	else:
		n_sens = len(data.sens_idx)

	# reorder to place place sens attr at the end
	if debias == 'X':
		X_train = data.X_original.numpy().astype(np.float32)
	elif debias == 'U':
		X_train = np.hstack((data.U,sens))
	return X_train, n_sens


def pw_dists(fpath, param_str, X, recompute=False):
	fname = fpath / (param_str + '_pw_dists.npy')
	print(f"Pairwise Distances fname: {fname}")
	if fname.exists() and not recompute:
		D_pw = np.load(fname)
	else:
		X = torch.FloatTensor(X)
		D_pw = cdist(X, X).numpy()
		with open(fname, 'wb') as fp:
			np.save(fp, D_pw)
	return D_pw


def get_key(s, y):
    return 'S'+str(s)+'Y'+str(y)


def fill_row(sep_SY, sens, Y, D, count_SY, idx_S_same, idx_S_diff, idx_Y_same, idx_Y_diff,j):
	key1 = get_key(int(sens[j]), int(Y[j]))
	count_SY[key1] += 1

	key2 = get_key(1,1)
	intsct_idx = list(set(idx_S_same) & set(idx_Y_same))
	sep_SY[key1+'-'+key2] += np.mean(D[j,intsct_idx])

	key2 = get_key(1,0)
	intsct_idx = list(set(idx_S_same) & set(idx_Y_diff))
	sep_SY[key1+'-'+key2] += np.mean(D[j,intsct_idx])

	key2 = get_key(0,1)
	intsct_idx = list(set(idx_S_diff) & set(idx_Y_same))
	sep_SY[key1+'-'+key2] += np.mean(D[j,intsct_idx])

	key2 = get_key(0,0)
	intsct_idx = list(set(idx_S_diff) & set(idx_Y_diff))
	sep_SY[key1+'-'+key2] += np.mean(D[j,intsct_idx])
	return sep_SY, count_SY


def compute_separabilities(number_of_nodes, data, D):
	# minmax rescaling to bring distances in (0,1) range. 
 	# helpful for comparing averages across original and debiased embeddings.
	D = MinMaxScaler().fit_transform(X=D)
	print(f"Number of nonzeros in pairwise-distances: {np.count_nonzero(D)}")

	# prep label and sens_attr vectors to compare against
	Y = data.Y_original.numpy()
	y_unique = np.unique(Y)
	sens = data.sens.numpy()
	s_unique = np.unique(sens)
	# initialize
	Y_conformities = np.zeros(number_of_nodes)
	S_conformities = np.zeros(number_of_nodes)

	Y_LSI = np.zeros(number_of_nodes)
	S_LSI = np.zeros(number_of_nodes)

	# average distance between each sensitive node to all other sensitive nodes, etc.
	n_sens = 0.0
	n_nonsens = 0.0
	count_SY = {}
	sep_SY = {}
	for s1 in [0,1]:
		for y1 in [0,1]:
			key1 = 'S'+str(s1)+'Y'+str(y1)
			count_SY[key1] = 0.0
			for s2 in [0,1]:
				sep_SY['S'+str(s1)+'-S'+str(s2)] = 0.0
				for y2 in [0,1]:
					key2 = 'S'+str(s2)+'Y'+str(y2)
					key = key1 + '-' + key2
					sep_SY[key] = 0.0
 
	for j in range(number_of_nodes):
		# find indexes where label and sens_attr are same / different
		idx_Y_same = []
		idx_Y_diff = []
		idx_S_same = []
		idx_S_diff = []
		for u in range(D.shape[0]):
			if Y[u] != Y[j]:
				idx_Y_diff.append(u)
			elif Y[u] == Y[j] and u != j:
				idx_Y_same.append(u)
			if sens[u] != sens[j]:
				idx_S_diff.append(u)
			elif sens[u] == sens[j] and u != j:
				idx_S_same.append(u)

		# find 5 closest points to j with same/different label/sens_attr
		D_Y_sort_diff_j = np.sort(D[j,idx_Y_diff])[:5]
		D_Y_sort_same_j = np.sort(D[j,idx_Y_same])[:5]
		D_S_sort_diff_j = np.sort(D[j,idx_S_diff])[:5]
		D_S_sort_same_j = np.sort(D[j,idx_S_same])[:5]

		D_Y_minus = D_Y_sort_diff_j - D_Y_sort_same_j
		D_S_minus = D_S_sort_diff_j - D_S_sort_same_j

		# compute conformity w.r.t. label and sens_attr
		# difference: smaller is better. scores can be negative.
		Y_conformities[j] = D_Y_minus[0]
		S_conformities[j] = D_S_minus[0]

		Y_LSI[j] = np.sum(D_Y_minus)
		S_LSI[j] = np.sum(D_S_minus)

		if sens[j]:
			n_sens += 1
			sep_SY['S1-S1'] += np.mean(D[j,idx_S_same])
			sep_SY['S1-S0'] += np.mean(D[j,idx_S_diff])
			if Y[j]:
				sep_SY, count_SY = fill_row(sep_SY, sens, Y, D, count_SY, idx_S_same, idx_S_diff, idx_Y_same, idx_Y_diff, j)
			else:
				sep_SY, count_SY = fill_row(sep_SY, sens, Y, D, count_SY, idx_S_same, idx_S_diff, idx_Y_same, idx_Y_diff, j)
		else:
			n_nonsens += 1
			sep_SY['S0-S0'] += np.mean(D[j,idx_S_same])
			sep_SY['S0-S1'] += np.mean(D[j,idx_S_diff])
			if Y[j]:
				sep_SY, count_SY = fill_row(sep_SY, sens, Y, D, count_SY, idx_S_same, idx_S_diff, idx_Y_same, idx_Y_diff, j)
			else:
				sep_SY, count_SY = fill_row(sep_SY, sens, Y, D, count_SY, idx_S_same, idx_S_diff, idx_Y_same, idx_Y_diff, j)

  
	assert n_sens == np.sum(sens)
	assert n_nonsens == number_of_nodes - n_sens
	sep_SY['S0-S0'] = sep_SY['S0-S0'] / n_sens
	sep_SY['S0-S1'] = sep_SY['S0-S1'] / n_sens
	sep_SY['S1-S0'] = sep_SY['S1-S0'] / n_nonsens
	sep_SY['S1-S1'] = sep_SY['S1-S1'] / n_nonsens
	Y_conformity = np.round(np.mean(Y_conformities),4)
	S_conformity = np.round(np.mean(S_conformities),4)
	avg_Y_LSI = np.round(np.mean(Y_LSI),4)
	avg_S_LSI = np.round(np.mean(S_LSI),4)

	sep_SY_np = np.zeros((6,6))
	colnames = ['S0', 'S1']
	for s1 in [0,1]:
		for y1 in [0,1]:
			key1 = 'S'+str(s1)+'Y'+str(y1)
			colnames = colnames + [key1]
			for s2 in [0,1]:
				sep_SY_np[s1,s2] = sep_SY['S'+str(s1)+'-S'+str(s2)]
				for y2 in [0,1]:
					key2 = 'S'+str(s2)+'Y'+str(y2)
					key = key1 + '-' + key2
					sep_SY[key] = np.round(sep_SY[key] / count_SY[key1], 4)
					sep_SY_np[2+s1*2+y1, 2+s2*2+y2] = sep_SY[key]
	sep_SY_pd = pd.DataFrame(sep_SY_np, index=colnames, columns=colnames)
	return Y_conformity, S_conformity, avg_Y_LSI, avg_S_LSI, sep_SY_pd
    

def debias_statistics(args, data, debias='U'):
	# create path for saving figures
	args.fig_dirpath = Path("fig") / args.dataset_name / data.data_dir.name
	args.fig_dirpath.mkdir(parents=True, exist_ok=True)

	# save results
	args.res_path = Path("results") / args.dataset_name / data.data_dir.name
	args.res_path.mkdir(parents=True, exist_ok=True)

	# load pairwise distances
	if debias == 'X':
		attr = ' Attributes'
		X_original = data.X_original.numpy().astype(np.float32)
		D_orig = pw_dists(args.X_transf_savepath, 'X_original', X_original, args.recompute_debiasing)
		D_transf = pw_dists(args.X_transf_savepath, args.X_transf_param_str, data.X_transf, args.recompute_debiasing)
		n_dims_orig = data.X_original.shape[1]
		n_dims_transf = data.X_transf.shape[1]
		diff_norm = np.linalg.norm(X_original[:,:-1]-data.X_transf,axis=1).flatten()
	elif debias == 'U':
		attr = ' Embedding'
		D_orig = pw_dists(args.emb_savepath, args.emb_str, data.U, args.recompute_debiasing)
		D_transf = pw_dists(args.U_transf_savepath, args.U_transf_param_str, data.U_debias, args.recompute_debiasing)
		n_dims_orig = data.U.shape[1]
		n_dims_transf = data.U_debias.shape[1]
		diff_norm = np.linalg.norm(data.U-data.U_debias,axis=1).flatten()

	# initialize
	results = dict()
	names = ['Original' + attr, 'Debiased' + attr]
	n_dims = [n_dims_orig, n_dims_transf]
	number_of_nodes = D_orig.shape[0]

	# compute results
	for i, D in enumerate([D_orig, D_transf]):
		results[names[i]] = dict()
		results[names[i]]['Number of Nodes'] = number_of_nodes
		results[names[i]]['Number of Dimensions'] = n_dims[i]

		# compute conformity and local separability index
		Y_conformity, S_conformity, avg_Y_LSI, avg_S_LSI, sep_SY_pd = compute_separabilities(number_of_nodes, data, D)
		results[names[i]]['Avg. Label Conformity'] = np.mean(Y_conformity)
		results[names[i]]['Avg. Sens_Attr Conformity'] = np.mean(S_conformity)
		results[names[i]]['Avg. Label LSI'] = np.mean(avg_Y_LSI)
		results[names[i]]['Avg. Sens_Attr LSI'] = np.mean(avg_S_LSI)

	results[names[0]]['Avg. Distance'] = 0.0
	results[names[1]]['Avg. Distance'] = np.round(np.mean(diff_norm),4)

	results = pd.DataFrame(results)


	if debias == 'X':
		transf_param_str = args.X_transf_param_str
	elif debias == 'U':
		transf_param_str = args.U_transf_param_str
    
	# write stat results to file
	results.to_csv(args.res_path / (transf_param_str + ".csv"))
	results.to_markdown(args.res_path / (transf_param_str + ".md"),tablefmt="grid")
	print(f"Debiasing{attr} Results savefile: {args.res_path / (transf_param_str + '.md')}")

	# write separability results to file
	sep_SY_pd.to_csv(args.res_path / (transf_param_str + "_separability.csv"))
	sep_SY_pd.to_markdown(args.res_path / (transf_param_str + "_separability.md"),tablefmt="grid")
	print(f"Debiasing{attr} Separability Results savefile: {args.res_path / (transf_param_str + '_separability.md')}")

	return results