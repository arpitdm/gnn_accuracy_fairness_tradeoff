# basic libraries
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import cdist
from sklearn.preprocessing import MinMaxScaler


# get argument parsers
from parse_args import *
from parse_args import parse_pretrain_args
from utils import create_param_str

# get algorithms
from debias.orig_ifair import *
from debias.iFair_github.iFair import iFair as iFair_gh


class Data:
	pass


def create_data():
	# sample points from isotropic gaussian with unit variance
	np.random.seed(42)
	mean1 = [3.5, 4.0]
	cov2 = [[1.0,0],[0,1.0]]
	D_iso = np.random.multivariate_normal(mean1, cov2, 50)

	# sample points from correlated gaussian with 0.95 covariance and 1.0 variance
	mean2 = [4.0, 3.0]
	cov2 = [[1,0.95],[0.95,1]]
	D_corr = np.random.multivariate_normal(mean2, cov2, 50)
	X = np.vstack((D_iso, D_corr))	
	
	# sensitive attribute set to 1 randomly with probability 0.3
	S_random = np.random.choice([0,1], size=100, p=[0.7,0.3]).astype(np.int32)
	X_random = np.hstack((X.copy(),S_random[:,np.newaxis]))

	# sensitive attribute correlated with X1, set to 1 if X1 <= 3
	S1 = np.zeros(100).astype(np.int32)
	S1[np.where(X[:,0]<=2.8)] = 1
	X1 = np.hstack((X.copy(), S1[:,np.newaxis]))

	# sensitive attribute correlated with X2, set to 1 if X2 <= 3
	S2 = np.zeros(100).astype(np.int32)
	S2[np.where(X[:,1]<=2.8)] = 1
	X2 = np.hstack((X.copy(), S2[:,np.newaxis]))

	Y = np.array([0]*50 + [1]*50).astype(np.int32)

	return X_random, X1, X2, Y


def create_ifair_data(args, X_train):
	ifair = iFair(k=args.k, A_x=args.A_x, A_z=args.A_z, max_iter=args.max_iter, nb_restarts=args.nb_restarts)
	X_transf = ifair.fit_transform(X_train)
	return X_transf


def create_plot(args, data):
	n = 100
	# labels = markers
	markers = np.array(['o']*50 + ['+']*50)

	# sens_attr = colors
	cR = ['tab:orange' if s else 'tab:blue' for s in data.XR[:,-1]]
	c1 = ['tab:orange' if s else 'tab:blue' for s in data.X1[:,-1]]
	c2 = ['tab:orange' if s else 'tab:blue' for s in data.X2[:,-1]]

	fig, ax = plt.subplots(3,2,figsize=(10,15))
	ax = ax.flatten()

	# original + ifair -> random Y
	for _s, c, _x, _y in zip(markers, cR, data.XR[:,0], data.XR[:,1]):
		ax[0].scatter(_x, _y, marker=_s, c=c)
	ax[0].set_title('Sensitive Attribute: Random (Original)')

	s = (np.max(data.XR_hat[:,0]) - np.min(data.XR_hat[:,0]))*0.05
	N = np.random.uniform(-s, s, size=n)
	for _s, c, _x, _y in zip(markers, cR, data.XR_hat[:,0]+N, data.XR_hat[:,1]):
		ax[1].scatter(_x, _y, marker=_s, c=c)
	ax[1].set_title('Sensitive Attribute: Random (iFair)')

	# original + ifair -> correlated with X1
	for _s, c, _x, _y in zip(markers, c2, data.X1[:,0], data.X1[:,1]):
		ax[2].scatter(_x, _y, marker=_s, c=c)
	ax[2].set_title('Sensitive Attribute: Correlated with X1 (Original)')

	s = (np.abs(np.min(data.X1_hat[:,0]))+np.abs(np.max(data.X1_hat[:,0])))*0.05
	N = np.random.uniform(-s, s, size=n)
	for _s, c, _x, _y in zip(markers, c2, data.X1_hat[:,0]+N, data.X1_hat[:,1]):
		ax[3].scatter(_x, _y, marker=_s, c=c)
	ax[3].set_title('Sensitive Attribute: Correlated with X1 (iFair)')

	# original + ifair -> correlated with X2
	for _s, c, _x, _y in zip(markers, c1, data.X2[:,0], data.X2[:,1]):
		ax[4].scatter(_x, _y, marker=_s, c=c)
	ax[4].set_title('Sensitive Attribute: Correlated with X2 (Original)')

	s = (np.abs(np.min(data.X2_hat[:,0]))+np.abs(np.max(data.X2_hat[:,0])))*0.05
	N = np.random.uniform(-s, s, size=n)
	for _s, c, _x, _y in zip(markers, c1, data.X2_hat[:,0]+N, data.X2_hat[:,1]):
		ax[5].scatter(_x, _y, marker=_s, c=c)
	ax[5].set_title('Sensitive Attribute: Correlated with X2 (iFair)')

	path = "fig/ifair_synthetic_experiment/"
	ifair_param_str = create_param_str(args, object='iFair_X')
	fname = path + ifair_param_str + '.pdf'
	print(f"Savefile-name of figure: {fname}")
	plt.savefig(fname)


def compute_conformities(X, X_hat, Y):
	D = cdist(torch.FloatTensor(X), torch.FloatTensor(X)).numpy()
	D_hat = cdist(torch.FloatTensor(X_hat), torch.FloatTensor(X_hat)).numpy()
	D = MinMaxScaler().fit_transform(X=D)
	D_hat = MinMaxScaler().fit_transform(X=D_hat)
	
	print(f"Num-Nonzero (pairwise-distances between original X) D: {np.count_nonzero(D)}, Num-Nonzero  (pairwise-distances between iFair-X)  D_hat: {np.count_nonzero(D_hat)}")
	D_Y_conformities = np.zeros(X.shape[0])
	Dhat_Y_conformities = np.zeros(X.shape[0])

	for j in range(X.shape[0]):
		idx_Y_same = []
		idx_Y_diff = []
		for u in range(X.shape[0]):
			if Y[u] != Y[j]:
				idx_Y_diff.append(u)
			elif Y[u] == Y[j] and u != j:
				idx_Y_same.append(u)

		D_min_diff_j = np.min(D[j,idx_Y_diff])
		D_min_same_j = np.min(D[j,idx_Y_same])
		Dhat_min_diff_j = np.min(D_hat[j,idx_Y_diff])
		Dhat_min_same_j = np.min(D_hat[j,idx_Y_same])

		D_Y_conformities[j] = D_min_diff_j - D_min_same_j
		Dhat_Y_conformities[j] = Dhat_min_diff_j - Dhat_min_same_j
	avg_D_Y_conformity = np.round(np.mean(np.ma.masked_invalid(D_Y_conformities)),4)
	avg_Dhat_Y_conformity = np.round(np.mean(np.ma.masked_invalid(Dhat_Y_conformities)),4)
	print(f"Conformity in original X: {avg_D_Y_conformity} | Conformity in iFair-X: {avg_Dhat_Y_conformity}")
	

def main():
	# set arguments
	parser = argparse.ArgumentParser() 
	parser = parse_experiment_args(parser)
	parser = parse_pretrain_args(parser)
	args = parser.parse_args()
	args.parser = parser
	args.device = 'cpu'

	warnings.filterwarnings("ignore")

	data = Data()
	data.XR, data.X1, data.X2, data.Y = create_data()
	data.XR_hat = create_ifair_data(args, data.XR)
	data.X1_hat = create_ifair_data(args, data.X1)
	data.X2_hat = create_ifair_data(args, data.X2)

	print(f"Conformity = Average Euclidean distance between points of the same/different Label")
	compute_conformities(data.XR, data.XR_hat, data.Y)
	print(f"Conformity = Average Euclidean distance between points of the same/different Sensitive Attribute Value")
	compute_conformities(data.XR, data.XR_hat, data.XR[:,-1])

	create_plot(args, data)


if __name__ == '__main__':
	main()