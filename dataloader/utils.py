import random
import numpy as np
import pandas as pd
import os.path as osp
import networkx as nx
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from typing import Optional, Union, Iterable, Callable, List

import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.data import Data, HeteroData, InMemoryDataset, download_url


class CustomInMemoryDataset(InMemoryDataset):
	r"""Custom base class for creating graph datasets which easily fit
	into CPU memory.
	Inherits from :class:`torch_geometric.data.InMemoryDataset`.
	See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
	create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
	tutorial.

	Args:
		- `root` (string, optional): Root directory where the dataset should be
			saved. (default: :obj:`None`)
		- `transform` (callable, optional): A function/transform that takes in 
			an :obj:`torch_geometric.data.Data` object and returns a transformed
			version. The data object will be transformed before every access.
			(default: :obj:`None`)
		- `pre_transform` (callable, optional): A function/transform that takes 
			in an :obj:`torch_geometric.data.Data` object and returns a
			transformed version. The data object will be transformed before
			being saved to disk. (default: :obj:`None`)
		- `pre_filter` (callable, optional): A function that takes in an
			:obj:`torch_geometric.data.Data` object and returns a boolean
			value, indicating whether the data object should be included in the
			final dataset. (default: :obj:`None`)
		- `log` (bool, optional): Whether to print any console output while
			downloading and processing the dataset. (default: :obj:`True`)
	"""
	def __init__(self, root, split: str = "default", transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):       
		super().__init__(root, pre_transform=pre_transform, transform=transform)
		self.data = torch.load(self.processed_paths[0])

	@property
	def raw_dir(self) -> str:
		return osp.join(self.root, 'raw')

	@property
	def raw_paths(self) -> List[str]:
		# The absolute filepaths that must be present in order to skip downloading.
		files = self.raw_file_names
		return [osp.join(self.raw_dir, f) for f in files]

	@property
	def processed_dir(self) -> str:
		return osp.join(self.root, 'processed')

	@property
	def processed_file_names(self) -> str:
		return 'data.pt'

	def download(self):
		for url in self.URLs:
			download_url(url, self.raw_dir)

	def _build_relationship_NIFTY(x: pd.DataFrame, thresh: float = 0.25, seed: int = 912) -> np.array:
		r"""Builds a graph based on feature similarity (copied from NIFTY).

		Args:
			- `x` (pd.DataFrame): Features used for computing similarity 
				scores.
			- `thresh` (float, optional): Threshold similarity score for adding an 
				edge between any pair of nodes.
			- `seed` (int, optional): Random seed for reproducibility.

		Returns:
			- `idx_map` (np.array): Unordered list of edges.
		"""
		random.seed(seed)
		df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
		df_euclid = df_euclid.to_numpy()
		idx_map = []
		for ind in range(df_euclid.shape[0]):
			max_sim = np.sort(df_euclid[ind, :])[-2]
			neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
			random.shuffle(neig_id)
			for neig in neig_id:
				if neig != ind:
					idx_map.append([ind, neig])
		idx_map =  np.array(idx_map)
		
		return idx_map

	def build_edges(self, fname:str, x:pd.DataFrame = None, thresh:float = None, method: str='NIFTY') -> np.ndarray:
		# if edgelist file exists, load using np.genfromtxt
		# otherwise, build relationship if it doesn't exist (as per NIFTY)
		# TODO: generalize to non-edgelist formats and other methods
		edgelist_fname = osp.join(self.raw_dir, fname)
		if osp.exists(edgelist_fname):
			edges = np.genfromtxt(edgelist_fname).astype('int')
		else:
			if method == 'NIFTY':
				edges = self._build_relationship_NIFTY(x=x, thresh=thresh)
			else:
				raise NotImplementedError
			np.savetxt(edgelist_fname, edges)
		return edges

	def unordered_edges_to_adjacency_matrix(self, num_nodes:int, edges_unordered:np.ndarray, idx:np.array = None) -> sp.csr_matrix:
		# converts array of unordered (potentially randomly labelled nodes) to symmetric adjacency matrix with self-loops
		if idx is None:
			idx = np.arange(num_nodes)
		idx_map = {j: i for i, j in enumerate(idx)}
		edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
		adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float32)
		adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
		adj.tocsr().setdiag(1,k=0)
		return adj


class CustomData(Data):
	r"""Custom class for a data object describing a homogeneous graph.

	Args:
		`x` (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
			num_node_features]`. (default: :obj:`None`)
		`edge_index` (LongTensor, optional): Graph connectivity in COO format
			with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
		`edge_attr` (Tensor, optional): Edge feature matrix with shape
			:obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
		`y` (Tensor, optional): Graph-level or node-level ground-truth labels
			with arbitrary shape. (default: :obj:`None`)
		`pos` (Tensor, optional): Node position matrix with shape
			:obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
		**kwargs (optional): Additional attributes.
	"""
	def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
				 edge_attr: OptTensor = None, y: OptTensor = None,
				 pos: OptTensor = None, **kwargs):
		super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
		self.save_pyg_data_to_disk()
		self.save_attributes_labels_to_disk()
		self.save_edgelist_to_disk()

	@property
	def num_nodes(self) -> int:
		return self.X_original.shape[0]

	@property
	def num_attributes(self) -> int:
		return self.X_original.shape[1]

	@property
	def num_edges(self) -> int:
		return int((self.edge_index_original.shape[1]+self.num_nodes)/2)

	def save_pyg_data_to_disk(self):
		# save PyG data to file. `collate` from PyG not used since we don't have PPI style data with multiple graphs or splits.
		self = self if self.pre_transform is None else self.pre_transform(self)
		torch.save(self, self.processed_paths[0])

	def save_attributes_labels_to_disk(self):
		# write processed attributes to disk
		attributes_df = pd.DataFrame(self.X_original.numpy(),columns=self.header,dtype=np.float32)
		attributes_df.to_csv(f'{self.processed_dir}/{self.name}_X_original.csv')

		# write labels to disk
		labels_df = pd.DataFrame(self.Y_original.numpy(),columns=[self.predict_attr],dtype=np.int32)
		labels_df.to_csv(f'{self.processed_dir}/{self.name}_Y_original.csv')

	def save_edgelist_to_disk(self):
		# writes processed graph to disk in edgelist format
		write_to_edgelist(self, self.processed_dir, self.name, self.edge_index_str)


def to_networkx(data, edge_index_str: str = None, node_attrs: Iterable[str] = None, edge_attrs: Iterable[str] = None, to_undirected: Union[bool, str] = False, remove_self_loops: bool = False) -> nx.Graph:
	r"""Converts a :class:`torch_geometric.data.Data` instance to a
	:obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
	a directed :obj:`networkx.DiGraph` otherwise.
	Modified from PyG to allow for choosing which `edge_index` to convert to graph object. Requires that any `edge_index` has exactly `num_nodes` nodes.

	Args:
		- `data` (torch_geometric.data.Data): The data object.
		- `edge_index` (torch.Tensor): Name of edge index attribute of data to be converted to Networkx graph object.
		- `node_attrs` (iterable of str, optional): The node attributes to be
			copied. (default: :obj:`None`)
		- `edge_attrs` (iterable of str, optional): The edge attributes to be
			copied. (default: :obj:`None`)
		- `to_undirected` (bool or str, optional): If set to :obj:`True` or
			"upper", will return a :obj:`networkx.Graph` instead of a
			:obj:`networkx.DiGraph`. The undirected graph will correspond to
			the upper triangle of the corresponding adjacency matrix.
			Similarly, if set to "lower", the undirected graph will correspond
			to the lower triangle of the adjacency matrix. (default:
			:obj:`False`)
		- `remove_self_loops` (bool, optional): If set to :obj:`True`, will not
			include self loops in the resulting graph. (default: :obj:`False`)

	Returns:
		- `G` (nx.Graph): Graph object constructed from edge-index.
	"""
	if to_undirected:
		G = nx.Graph()
	else:
		G = nx.DiGraph()

	G.add_nodes_from(range(data.num_nodes))

	node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

	values = {}
	for key, value in data(*(node_attrs + edge_attrs)):
		if torch.is_tensor(value):
			value = value if value.dim() <= 1 else value.squeeze(-1)
			values[key] = value.tolist()
		else:
			values[key] = value

	to_undirected = "upper" if to_undirected is True else to_undirected
	to_undirected_upper = True if to_undirected == "upper" else False
	to_undirected_lower = True if to_undirected == "lower" else False

	if edge_index_str is not None:
		edge_index = getattr(data,edge_index_str)
	else:
		edge_index = data.edge_index
	for i, (u, v) in enumerate(edge_index.t().tolist()):

		if to_undirected_upper and u > v:
			continue
		elif to_undirected_lower and u < v:
			continue

		if remove_self_loops and u == v:
			continue

		G.add_edge(u, v)

		for key in edge_attrs:
			G[u][v][key] = values[key][i]

	for key in node_attrs:
		for i, feat_dict in G.nodes(data=True):
			feat_dict.update({key: values[key][i]})

	return G


def write_to_edgelist(data, processed_dir: str, name: str, edge_index_str: str = None):
	r"""Writes processed graph data to disk in an edgelist format.

	Args:
		- `data` (torch_geometric.data.Data): The data object.
		processed_dir (str): Path to where edgelist will be stored.
		- `name` (str): Name of the dataset.
		- `edge_index_str` (str): Name of the edge-index attribute in `data`.
	"""
	G = to_networkx(data, to_undirected=True, edge_index_str=edge_index_str)
	edgelist_fname = '_'.join(filter(None, [name, edge_index_str]))
	edgelist_fname = f'{processed_dir}/{edgelist_fname}.edgelist'
	nx.write_edgelist(G, edgelist_fname, data=False)
	return


class AddRemainingSelfLoops(BaseTransform):
	r"""Adds remaining self-loops to the given homogeneous or heterogeneous graph.

	Args:
		- `attr`: (str, optional): The name of the attribute of edge weights
			or multi-dimensional edge features to pass to
			:meth:`torch_geometric.utils.add_self_loops`.
			(default: :obj:`"edge_weight"`)
		- `fill_value` (float or Tensor or str, optional): The way to generate
			edge features of self-loops (in case :obj:`attr != None`).
			If given as :obj:`float` or :class:`torch.Tensor`, edge features of
			self-loops will be directly given by :obj:`fill_value`.
			If given as :obj:`str`, edge features of self-loops are computed by
			aggregating all features of edges that point to the specific node,
			according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
			:obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
	"""
	def __init__(self, attr: Optional[str] = 'edge_weight',
				 fill_value: Union[float, Tensor, str] = None):
		self.attr = attr
		self.fill_value = fill_value

	def __call__(self, data: Union[Data, HeteroData]):
		for store in data.edge_stores:
			if store.is_bipartite() or 'edge_index' not in store:
				continue

			store.edge_index, edge_weight = add_remaining_self_loops(
				store.edge_index, getattr(store, self.attr, None),
				fill_value=self.fill_value, num_nodes=store.size(0))

			setattr(store, self.attr, edge_weight)

		return data

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}()'


class TrainValTestMask(BaseTransform):
	r"""Adds train-val-test mask to the data object. This creates a stratified split such that the label/sensitive attribute balance in the entire dataset is reflected in the training, validation, and test sets.

	Args:
		- `train_size` (float, optional): Fraction of nodes to use for train set (default: `0.2`)
		- `val_size` (float, optional): Fraction of nodes to use for validation set (default: `0.2`)
		- `seed` (int, optional): Random seed for reproducibility (default: `42`)
	"""
	def __init__(self, train_size: float = 0.2, val_size: float = 0.2, seed: int = 42):
		self.seed = seed
		self.train_size = train_size
		self.val_size = val_size
		self.test_size = 1.0 - (self.train_size + self.val_size)
		assert self.train_size + self.val_size + self.test_size == 1

	def __call__(self, data: Union[Data, HeteroData]):
		np.random.seed(self.seed)
		random.seed(self.seed)
		
		label_idx_0 = np.where(data.Y_original==0)[0]
		label_idx_1 = np.where(data.Y_original==1)[0]
		n = data.Y_original.shape[0]
		n0 = len(label_idx_0)
		frac_0 = n0 / n
		frac_1 = 1.0 - frac_0
		random.shuffle(label_idx_0)
		random.shuffle(label_idx_1)

		# (fraction of points in dataset with label 0) * train_size fraction
		f_0_train = int(self.train_size * frac_0 * n)
		f_1_train = int(self.train_size * frac_1 * n)
		idx_train = np.append(label_idx_0[:f_0_train], label_idx_1[:f_1_train])
		
		# next val_size fraction of points as validation set
		f_0_val = int(self.val_size * frac_0 * n)
		f_1_val = int(self.val_size * frac_1 * n)
		idx_val = np.append(label_idx_0[f_0_train:f_0_train+f_0_val], label_idx_1[f_1_train:f_1_train+f_1_val])
		
		# remaining as test
		idx_test = np.append(label_idx_0[f_0_train+f_0_val:], label_idx_1[f_1_train+f_1_val:])

		data.idx_train = idx_train
		data.idx_val = idx_val
		data.idx_test = idx_test
		
		return data

	def __repr__(self) -> str:
		return f'{self.__class__.__name__}()'