import numpy as np
import pandas as pd
from pathlib import Path
from torch import cdist, FloatTensor
from sklearn.preprocessing import MinMaxScaler


def pw_dists(fpath, param_str, X, recompute=False):
	fname = fpath / (param_str + '_pw_dists.npy')
	print(f"Pairwise Distances fname: {fname}")
	if fname.exists() and not recompute:
		D_pw = np.load(fname)
	else:
		X = FloatTensor(X)
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


def compute_conformities(number_of_nodes, data, D):
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
	n_neighbors = 20
 
	for j in range(number_of_nodes):
		# find idx of 100 closest points to j
		nearest_neighbors_idx = np.argsort(D[j,:])[1:n_neighbors+1]

		# compute conformity w.r.t. label and sens_attr
		# difference: smaller is better. scores can be negative.
		if Y[nearest_neighbors_idx[0]] == Y[j]:
			Y_conformities[j] = 1
			Y_LSI[j] = 1
		if sens[nearest_neighbors_idx[0]] == sens[j]:
			S_conformities[j] = 1
			S_LSI[j] = 1
  
		for i in nearest_neighbors_idx[1:]:
			if Y[i] == Y[j]:
				Y_LSI[j] += 1
			if sens[i] == sens[j]:
				S_LSI[j] += 1
		Y_LSI[j] /= n_neighbors*1.0
		S_LSI[j] /= n_neighbors*1.0
	return Y_conformities, S_conformities, Y_LSI, S_LSI
    

def compute_debias_statistics(args, data, M_debias, debias='X'):
	if debias == 'X':
		debias_param_str = data.X_debias_param_str
	elif debias == 'U':
		debias_param_str = data.U_debias_param_str

	# save results
	args.res_path = Path("results") / args.dataset_name / args.pretrain_algo
	args.res_path.mkdir(parents=True, exist_ok=True)
	res_f_prefix = f"{args.res_path}/{debias_param_str}_{args.dataset_seed}"

	# load pairwise distances
	if debias == 'X':
		attr = ' Attributes'
		X_original = data.X_original.numpy().astype(np.float32)[:,:-1]
		D_orig = pw_dists(data.X_debias_savepath, 'X_original', X_original, args.recompute_debiasing)
		D_debias = pw_dists(data.X_debias_savepath, data.X_debias_param_str, M_debias, args.recompute_debiasing)
		n_dims_orig = data.X_original.shape[1]
		n_dims_debias = M_debias.shape[1]
		D_orig = MinMaxScaler().fit_transform(X=D_orig)
		D_debias = MinMaxScaler().fit_transform(X=D_debias)
  
	elif debias == 'U':
		attr = ' Embedding'
		D_orig = pw_dists(data.U_savepath, data.U_param_str, data.U, args.recompute_debiasing)
		D_debias = pw_dists(data.U_debias_savepath, data.U_debias_param_str, data.U_debias, args.recompute_debiasing)
		n_dims_orig = data.U.shape[1]
		n_dims_debias = data.U_debias.shape[1]

	# initialize
	results = dict()
	names = ['Original' + attr, 'Debiased' + attr]
	n_dims = [n_dims_orig, n_dims_debias]
	number_of_nodes = D_orig.shape[0]

	# compute results
	for i, D in enumerate([D_orig, D_debias]):
		results[names[i]] = dict()
		results[names[i]]['Number of Nodes'] = number_of_nodes
		results[names[i]]['Number of Dimensions'] = n_dims[i]

		# compute conformity and local separability index
		Y_conformities, S_conformities, Y_LSI, S_LSI = compute_conformities(number_of_nodes, data, D)
		# Y_conformity, S_conformity, avg_Y_LSI, avg_S_LSI, sep_SY_pd = compute_separabilities(number_of_nodes, data, D)
		results[names[i]]['Avg. Label Conformity'] = np.mean(Y_conformities)
		results[names[i]]['Avg. Sens_Attr Conformity'] = np.mean(S_conformities)
		results[names[i]]['Avg. Label LSI'] = np.mean(Y_LSI)
		results[names[i]]['Avg. Sens_Attr LSI'] = np.mean(S_LSI)

		# write separability results to file
		# sep_SY_pd.to_csv(f"{res_f_prefix}_{names[i]}_separability.csv")
		# sep_SY_pd.to_markdown(f"{res_f_prefix}_{names[i]}_separability.md",tablefmt="grid")
		# print(f"Debiasing{attr} Separability Results savefile: {res_f_prefix}_separability.md")

	results = pd.DataFrame(results)
    
	# write stat results to file
	results.to_csv(f"{res_f_prefix}.csv")
	results.to_markdown(f"{res_f_prefix}.md",tablefmt="grid")
	print(f"Debiasing{attr} Results savefile: {res_f_prefix}.md")

	return results