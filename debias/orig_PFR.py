"""
Adapted implementation of PFR. Original implementation by Preethi Lahoti (plahoti@mpi-inf.mpg.de).
"""
from __future__ import division
import warnings
import numpy as np
import pandas as pd
from numba import jit
from scipy.linalg import eigh
from scipy.sparse import csgraph, csr_matrix
import sklearn.metrics.pairwise as pairwise
from sklearn.preprocessing import MinMaxScaler

# build adjacency matrix of k-nearest neighbor graph over input space X
@jit(parallel=True)
def construct_W_X(S_x, nn_k, t):
    n = S_x.shape[0]
    print(f"num nodes construct_W_X: {n}")
    # W_X = np.zeros((n,n)).astype(float)
    W_X = csr_matrix((n, n), dtype=np.float32)

    # argsort similarity matrix by row
    idx_sorted = np.argsort(S_x,axis=1)

    # for each row, find nn_k nearest neighbors
    idx_nn_k = idx_sorted[:,:nn_k]
    for i in range(n):
        for j in range(n):
            if i in idx_nn_k[j] or j in idx_nn_k[i]:
                W_X[i][j] = np.exp(-S_x[i][j]/t)
    return W_X


# Build between group quantile graph
# In the original PFR paper, they elicit pairwise judgements between diverse and incomparable groups for the COMPAS dataset. They use something called Northpointe's COMPAS decile scores. I couldn't locate this online and to the best of my knowledge, this is unavailable for the datasets we use. So for the German dataset, I use the 'LoanAmount' feature as the Y_s (random variable indicating the ranked position of individuals in X_s where X_s is the set of features of nodes in subgroup s as defined by the sensitive attribute)
@jit(parallel=True)
def construct_W_F(args, data, X_train, quantiles):
    # get node id by membership in sensitive group
    sens = data.X_original[:,-1].numpy()
    sens_idx_0 = np.where(sens == 0)[0]
    sens_idx_1 = np.where(sens == 1)[0]

    # define ranking based on features
    # German:LoanAmount, Recidivism:AGE, Credit:MaxBillAmountOverLast6Months
    Ys = data.Ys
        
    # place into quantile buckets for each sens
    Ys_0_quantiles = pd.qcut(Ys[sens_idx_0],quantiles,labels=False,duplicates='drop')
    Ys_1_quantiles = pd.qcut(Ys[sens_idx_1],quantiles,labels=False,duplicates='drop')
    assert len(sens_idx_0) == len(Ys_0_quantiles)
    assert len(sens_idx_1) == len(Ys_1_quantiles)
    
    # build fairness graph
    n = X_train.shape[0]
    W_F = np.zeros((n,n)).astype(float)
    for i in range(len(sens_idx_0)):
        for j in range(len(sens_idx_1)):
            if Ys_0_quantiles[i] == Ys_1_quantiles[j]:
                W_F[sens_idx_0[i]][sens_idx_1[j]] = 1
                W_F[sens_idx_1[j]][sens_idx_0[i]] = 1
    return W_F

    
# MinMax scale all features. X_train does not contain sensitive attribute
# Compute W_X and W_F.
def prepare_data(args, data, debias):
    if debias == 'X':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(data.X_original[:, :-1].numpy())
        # construct pairwise similarity matrix
        S_x = pairwise.euclidean_distances(X_train, X_train)
        # build W_X
        W_X = construct_W_X(S_x, args.pfr_nn_k, args.pfr_t)
        W_F = construct_W_F(args, data, X_train, args.pfr_quantiles)
        return W_X, W_F, X_train        
    elif debias == 'U':
        X_train = data.U
        # construct pairwise similarity matrix
        S_x = pairwise.euclidean_distances(X_train, X_train)
        # build W_X
        W_X = construct_W_X(S_x, args.pfr_A_nn_k, args.pfr_A_t)
        W_F = construct_W_F(args, data, X_train, args.pfr_A_quantiles)
        return W_X, W_F, X_train        


def create_orig_PFR_data(args, data, debias="X"):

    warnings.filterwarnings("ignore")

    W_X, W_F, X_train = prepare_data(args, data, debias)
    print(f"W_X: {W_X.shape}, W_F: {W_F.shape}, X_train: {X_train.shape}, ")    

    # create PFR-transformed data
    if args.dataset_name == 'Region-Z':
        print("here")
        W_X = csr_matrix(W_X.astype(np.float32))
        W_F = csr_matrix(W_F.astype(np.float32))
    if debias == 'X':
        pfr = PFR(args.pfr_k, W_X, W_F, gamma=args.pfr_gamma)
    elif debias == 'U':
        pfr = PFR(args.pfr_A_k, W_X, W_F, gamma=args.pfr_A_gamma)
    X_hat = pfr.fit_transform(X_train)

    return X_hat


class PFR:
    def __init__(self, k, W_s, W_F, gamma = 1.0, normed = False):
        """
        Initializes the model.
        :param k:       Hyperparam representing the number of latent dimensions.
        :param W_s:     The adjacency matrix of nn-k-nearest neighbour graph over input space X
        :param W_F:     The adjacency matrix of the pairwise fairness graph G associated to the problem.
        :param nn_k:    Hyperparam that controls the number of neighbours considered in the similarity graph.
        :param gamma:   Hyperparam controlling the influence of W^F.
        :param alpha:   Hyperparam controlling the influence of W^X. Always set to 1 - gamma.
        """
        self.k = k
        self.W_F = W_F
        self.W_s = W_s
        self.gamma = gamma
        self.alpha = 1 - self.gamma
        self.normed = normed

    def fit(self, X):
        """
        Learn the model using the training data.
        :param X:     Training data.
        """
        print('Just fitting')
        W = (self.alpha * self.W_s) + (self.gamma * self.W_F)
        L, diag_p = csgraph.laplacian(W, normed=self.normed, return_diag=True)

        # - Formulate the eigenproblem.
        lhs_matrix = (X.T.dot(L.dot(X)))
        rhs_matrix = None

        # - Solve the problem
        eigval, eigvec = eigh(a=lhs_matrix,
                                b=rhs_matrix,
                                overwrite_a=True,
                                overwrite_b=True,
                                check_finite=True)
        eigval = np.real(eigval)

        # - Select eigenvectors based on eigenvalues
        # -- get indices of k smallest eigen values
        print(f"eigval: {eigval.shape}")
        argpart = np.argpartition(eigval, self.k-1)
        print(f"argpart: {argpart.shape}")
        k_eig_ixs = argpart[:self.k]

        # -- columns of eigvec are the eigen vectors corresponding to the eigen values
        # --- select column vectors corresponding to k largest eigen values
        self.V = eigvec[:, k_eig_ixs]

    def transform(self, X):
        print('Transforming...')
        return (self.V.T.dot(X.T)).T

    def fit_transform(self, X):
        """
        Learns the model from the training data and returns the data in the new space.
        :param X:   Training data.
        :return:    Training data in the new space.
        """
        self.fit(X)
        return self.transform(X)