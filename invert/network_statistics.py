import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import scipy.sparse as sp
from numpy.linalg import norm
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.utils import homophily
from torch_geometric.utils import from_scipy_sparse_matrix
from invert.deepwalk_backwards.diameter_estimation import getDiameter


def assortativity(G, w=None):
	return nx.degree_assortativity_coefficient(G, weight=w)


def plot_loglog_dist(names, results, args, data):
	# create path for saving figures
	fig_dirpath = Path("fig") / args.dataset_name / data.data_dir.name
	fig_dirpath.mkdir(parents=True, exist_ok=True)

	colors = ['red', 'green', 'blue']
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i, name in enumerate(names):
		deg_dict = dict(Counter(results[name]['degrees']))
		items = sorted(deg_dict.items())
		ax.plot([k for (k,v) in items], [v for (k,v) in items], color=colors[i],label=name)
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.title("Degree Distribution (log-log)")
	plt.legend()
	plt.tight_layout()
	plt.savefig(fig_dirpath / 'deg_dist.pdf', format='pdf')
	plt.close()
	print(f"Degree distribution figplot fname: {fig_dirpath / 'deg_dist.pdf'}")


def compute_SY_edge_changes(A, sens, Y):
	sep_SY_np = np.zeros((6,6)).astype(np.float32)
	edges_SY = np.zeros((2,2)).astype(np.float32)
	n_nodes = A.shape[0]
	n_edges = A.sum() / 2
	for i in range(n_nodes):
		for j in range(i+1,n_nodes):
			if A[i,j]:
				s1 = sens[i]
				s2 = sens[j]
				y1 = Y[i]
				y2 = Y[j]

				sep_SY_np[s1,s2] += 1
				if s1 != s2:
					sep_SY_np[s2,s1] += 1

				sep_SY_np[2+2*s1+y1, 2+2*s2+y2] += 1
				if y1 == y2:
					sep_SY_np[2+2*s2+y2, 2+2*s1+y1] += 1

				edges_SY[s1,y1] += 1
				edges_SY[s2,y2] += 1

	# sep_SY_np[:3,:3] /= n_edges
	colnames = ['S0', 'S1']
	for s in [0,1]:
		for y in [0,1]:
			key = 'S'+str(s)+'Y'+str(y)
			colnames = colnames + [key]
			# sep_SY_np[2+2*s+y,:] /= edges_SY[s,y]
	sep_SY_pd = pd.DataFrame(sep_SY_np, index=colnames, columns=colnames)
	return sep_SY_pd


def frob_error(A1, A2):
	return norm(A1 - A2) / norm(A1)


def collect_statistics(args, data):
	# save results
	res_dirpath = Path("results") / args.dataset_name / data.data_dir.name
	res_dirpath.mkdir(parents=True, exist_ok=True)

	# compute statistics of original, inverted-graph from embedding, and inverted graph from transformed/debiased embedding
	results = dict()
	names = ['Original', 'Embedding', 'Debiased Embedding']
	n_nodes = data.A.shape[0]
	m = (data.A.sum() + n_nodes)/2

	# count changes in types of edges
	sens = data.sens.numpy().astype(int)
	Y = data.Y_original.numpy().astype(int)
	SY_changes = []

	for i, Adj in enumerate([data.A, data.A_emb, data.A_debias]):
		results[names[i]] = dict()
		G = nx.from_numpy_array(Adj)
		results[names[i]]['Number of Nodes'] = n_nodes
		results[names[i]]['Number of Edges'] = (Adj.sum()+n_nodes) / 2

		# number of differences in edge sets
		n_diff = int(np.sum(np.abs(data.A - Adj))/2)
		results[names[i]]['Num Difference of Edge Sets'] = n_diff

		# jaccard of edge sets
		jac = ((data.A==Adj).sum() + n_nodes)/(2*m)
		results[names[i]]['Jaccard of Edge Sets'] = np.round(jac,4)

		# frobenius norm of difference between adj matrices
		results[names[i]]['Frobenius Error'] = frob_error(data.A, Adj)

		# maximum degree
		results[names[i]]['Maximum Degree'] = max(Adj.sum(axis=1))

		# average clustering coefficient
		cc = nx.average_clustering(G)
		results[names[i]]['Clustering Coefficient'] = np.round(cc,4)

		# compute homophily w.r.t. labels
		edge_index = from_scipy_sparse_matrix(sp.csr_matrix(Adj))[0]
		h_l = np.round(homophily(edge_index,data.Y_original,method="edge"),2)
		results[names[i]]['Label Homophily'] = h_l

		# compute homophily w.r.t. sensitive attribute
		h_s = np.round(homophily(edge_index,data.sens,method="edge"),2)
		results[names[i]]['Sensitive Attribute Homophily'] = h_s

		# compute statistics of types of edges in adj
		SY_changes.append(compute_SY_edge_changes(Adj, sens, Y))

	# create pandas df of separability
	sep_SY_pd = pd.concat(SY_changes)

	sep_SY_pd.to_csv(res_dirpath / (args.full_param_str + "_separability.csv"))
	sep_SY_pd.to_markdown(res_dirpath / (args.full_param_str + "_separability.md"),tablefmt="grid")
	print(f"Inverted Graph Separability Statistics savefile-name: {res_dirpath / (args.full_param_str + '_separability.md')}")

	results_df = pd.DataFrame(results)
	results_df.to_csv(res_dirpath / (args.full_param_str + ".csv"))
	results_df.to_markdown(res_dirpath / (args.full_param_str + ".md"),tablefmt="grid")
	print(f"Inverted Graph Statistics savefile-name: {res_dirpath / (args.full_param_str + '.md')}")

	# plot_loglog_dist(names, results, args, data)

	return results_df