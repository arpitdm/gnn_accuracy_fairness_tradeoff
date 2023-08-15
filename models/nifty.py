import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch_geometric.utils import dropout_adj
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.nn import GCNConv, SAGEConv, GINConv

# from utils import fair_metric, eval_model


def train_NIFTY(args, data):
	# setup
	epoch_times = []
 
	# initialize encoder and model
	encoder = Encoder(in_channels=data.n_features, 
				out_channels=args.hidden,
				base_model=args.encoder_name).to(args.device)	
	model = NIFTY(encoder=encoder,
			   num_hidden=args.hidden,
			   num_proj_hidden=args.proj_hidden,
			   sim_coeff=args.sim_coeff,
			   nclass=data.n_classes).to(args.device)

	# initialize nifty params
	val_edge_index_1 = dropout_adj(data.edge_index, p=args.drop_edge_rate_1)[0]
	val_edge_index_2 = dropout_adj(data.edge_index, p=args.drop_edge_rate_2)[0]
	val_x_1 = drop_feature(data.features, args.drop_feature_rate_2, data.sens_idx, sens_flag=False)
	val_x_2 = drop_feature(data.features, args.drop_feature_rate_2, data.sens_idx)

	# initialize optimizers
	par_1 = list(model.encoder.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters()) + list(model.fc4.parameters())
	par_2 = list(model.c1.parameters()) + list(model.encoder.parameters())
	optimizer_1 = optim.Adam(par_1, lr=args.lr, weight_decay=args.weight_decay)
	optimizer_2 = optim.Adam(par_2, lr=args.lr, weight_decay=args.weight_decay)

	# training loop
	for epoch in range(args.epochs+1):
		t = time.time()

		# forward
		model.train()
		optimizer_1.zero_grad()
		optimizer_2.zero_grad()
		edge_index_1 = dropout_adj(data.edge_index, p=args.drop_edge_rate_1)[0]
		edge_index_2 = dropout_adj(data.edge_index, p=args.drop_edge_rate_2)[0]
		x_1 = drop_feature(data.features, args.drop_feature_rate_2, data.sens_idx, sens_flag=False)
		x_2 = drop_feature(data.features, args.drop_feature_rate_2, data.sens_idx)
  
		z1 = model(x_1, edge_index_1)
		z2 = model(x_2, edge_index_2)

		# project
		p1 = model.projection(z1)
		p2 = model.projection(z2)

		# predict
		h1 = model.prediction(p1)
		h2 = model.prediction(p2)

		l1 = model.D(h1[data.idx_train], p2[data.idx_train])/2
		l2 = model.D(h2[data.idx_train], p1[data.idx_train])/2

		sim_loss = args.sim_coeff*(l1+l2)
		sim_loss.backward()
		optimizer_1.step()

		# classifier
		z1 = model(x_1, edge_index_1)
		z2 = model(x_2, edge_index_2)
		c1 = model.classifier(z1)
		c2 = model.classifier(z2)

		# training with binary cross-entropy loss
		l3 = F.binary_cross_entropy_with_logits(c1[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float().to(args.device))/2
		l4 = F.binary_cross_entropy_with_logits(c2[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float().to(args.device))/2

		cl_loss = (1-args.sim_coeff)*(l3+l4)
		cl_loss.backward()
		optimizer_2.step()
		loss = sim_loss + cl_loss
  
		# performance on validation set
		model.eval()
		val_s_loss, val_c_loss = ssf_validation(args, data, model, val_x_1, val_edge_index_1, val_x_2, val_edge_index_2)
		emb = model(val_x_1, val_edge_index_1)
		output = model.predict(emb)
		val_auc_roc = roc_auc_score(data.labels[data.idx_val].cpu().numpy(), output[data.idx_val].detach().cpu().numpy())

		epoch_times.append(time.time() - t)

		if args.verbose:
			print(f"Epoch: {epoch} | train_s_loss: {sim_loss:.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f} | val_auc_roc: {val_auc_roc:.4f} | time: {time.time() - t}")

	# save model
	torch.save(model.state_dict(), data.model_fname)

	# test_auc_roc, f1_s, parity, equality = test_NIFTY(args, data, model)
	avg_epoch_time = round(sum(epoch_times)/len(epoch_times), 4)

	return val_auc_roc, avg_epoch_time, model
	# return val_auc_roc, test_auc_roc, f1_s, parity, equality, avg_epoch_time, model


def test_NIFTY(args, data, model):
	model.eval()
	emb = model(data.features, data.edge_index)
	output = model.predict(emb)
	output_preds = (output.squeeze()>0).type_as(data.labels)
	test_preds = output_preds[data.idx_test].cpu().numpy()
	test_labels = data.labels[data.idx_test].cpu().numpy()
	test_output = output[data.idx_test].detach().cpu().numpy()
	test_sens = data.sens[data.idx_test].numpy()
 
	# compute accuracy scores
	test_auc_roc = roc_auc_score(test_labels, test_output)
	f1_s = f1_score(test_labels, test_preds)
	
	# compute statistical parity and equality of opportunity
	parity, equality = fair_metric(test_preds, test_labels, test_sens)
	
	return test_auc_roc, f1_s, parity, equality


def ssf_validation(args, data, model, x_1, edge_index_1, x_2, edge_index_2):
	z1 = model(x_1, edge_index_1)
	z2 = model(x_2, edge_index_2)

	# projector
	p1 = model.projection(z1)
	p2 = model.projection(z2)

	# predictor
	h1 = model.prediction(p1)
	h2 = model.prediction(p2)

	l1 = model.D(h1[data.idx_val], p2[data.idx_val])/2
	l2 = model.D(h2[data.idx_val], p1[data.idx_val])/2
	sim_loss = args.sim_coeff*(l1+l2)

	# classifier
	c1 = model.classifier(z1)
	c2 = model.classifier(z2)

	# Binary Cross-Entropy
	l3 = F.binary_cross_entropy_with_logits(c1[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float().to(args.device))/2
	l4 = F.binary_cross_entropy_with_logits(c2[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float().to(args.device))/2

	return sim_loss, l3+l4    


class Encoder(torch.nn.Module):
	def __init__(self, in_channels: int, out_channels: int, 
				base_model='GraphSAGE', k: int = 2):
		super(Encoder, self).__init__()
		self.base_model = base_model
		if self.base_model == 'GCN':
			self.conv = GCN(in_channels, out_channels)  
		elif self.base_model == 'GraphSAGE':
			self.conv = SAGE(in_channels, out_channels)
		elif self.base_model == 'GIN':
			self.conv = GIN(in_channels, out_channels)   
		else:
			raise NotImplementedError

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
		x = self.conv(x, edge_index)
		return x


class NIFTY(torch.nn.Module):
	def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, 
				sim_coeff: float = 0.5, nclass: int=1):
		super(NIFTY, self).__init__()
		self.encoder: Encoder = encoder
		self.sim_coeff: float = sim_coeff

		# Projection
		self.fc1 = nn.Sequential(
			spectral_norm(nn.Linear(num_hidden, num_proj_hidden)),
			nn.BatchNorm1d(num_proj_hidden),
			nn.ReLU(inplace=True)
		)
		self.fc2 = nn.Sequential(
			spectral_norm(nn.Linear(num_proj_hidden, num_hidden)),
			nn.BatchNorm1d(num_hidden)
		)

		# Prediction
		self.fc3 = nn.Sequential(
			spectral_norm(nn.Linear(num_hidden, num_hidden)),
			nn.BatchNorm1d(num_hidden),
			nn.ReLU(inplace=True)
		)
		self.fc4 = spectral_norm(nn.Linear(num_hidden, num_hidden))

		# Classifier
		self.c1 = Classifier(ft_in=num_hidden, nb_classes=nclass)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, x: torch.Tensor,
					edge_index: torch.Tensor) -> torch.Tensor:
		return self.encoder(x, edge_index)

	def projection(self, z):
		z = self.fc1(z)
		z = self.fc2(z)
		return z

	def prediction(self, z):
		z = self.fc3(z)
		z = self.fc4(z)
		return z

	def classifier(self, z):
		return self.c1(z)

	def normalize(self, x):
		val = torch.norm(x, p=2, dim=1).detach()
		x = x.div(val.unsqueeze(dim=1).expand_as(x))
		return x

	def D_entropy(self, x1, x2):
		x2 = x2.detach()
		return (-torch.max(F.softmax(x2), dim=1)[0]*torch.log(torch.max(F.softmax(x1), dim=1)[0])).mean()

	def D(self, x1, x2): # negative cosine similarity
		return -F.cosine_similarity(x1, x2.detach(), dim=-1).mean()

	def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, e_1, e_2, idx):

		# projector
		p1 = self.projection(z1)
		p2 = self.projection(z2)

		# predictor
		h1 = self.prediction(p1)
		h2 = self.prediction(p2)

		# classifier
		c1 = self.classifier(z1)

		l1 = self.D(h1[idx], p2[idx])/2
		l2 = self.D(h2[idx], p1[idx])/2
		l3 = F.cross_entropy(c1[idx], z3[idx].squeeze().long().detach())

		return self.sim_coeff*(l1+l2), l3

	def fair_metric(self, pred, labels, sens):
		idx_s0 = sens==0
		idx_s1 = sens==1

		idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
		idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)

		parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
		equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))

		return parity.item(), equality.item()

	def predict(self, emb):

		# projector
		p1 = self.projection(emb)

		# predictor
		h1 = self.prediction(p1)

		# classifier
		c1 = self.classifier(emb)

		return c1

	def linear_eval(self, emb, labels, idx_train, idx_test):
		x = emb.detach()
		classifier = nn.Linear(in_features=x.shape[1], out_features=2, bias=True)
		classifier = classifier.to('cuda')
		optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
		for i in range(1000):
			optimizer.zero_grad()
			preds = classifier(x[idx_train])
			loss = F.cross_entropy(preds, labels[idx_train])
			loss.backward()
			optimizer.step()
			if i%100==0:
				print(loss.item())
		classifier.eval()
		preds = classifier(x[idx_test]).argmax(dim=1)
		correct = (preds == labels[idx_test]).sum().item()
		return preds, correct/preds.shape[0]


class Classifier(nn.Module):
	def __init__(self, ft_in, nb_classes):
		super(Classifier, self).__init__()

		# Classifier projector
		self.fc1 = spectral_norm(nn.Linear(ft_in, nb_classes))

	def forward(self, seq):
		ret = self.fc1(seq)
		return ret


class GCN(nn.Module):
	def __init__(self, nfeat, nhid, dropout=0.5):
		super(GCN, self).__init__()
		self.gc1 = GCNConv(nfeat, nhid)

	def forward(self, x, edge_index):
		x = self.gc1(x, edge_index)
		return x


class SAGE(nn.Module):
	def __init__(self, nfeat, nhid, dropout=0.5):
		super(SAGE, self).__init__()

		# Implemented spectral_norm in the sage main file
		# ~/anaconda3/envs/PYTORCH/lib/python3.7/site-packages/torch_geometric/nn/conv/sage_conv.py
		self.conv1 = SAGEConv(nfeat, nhid, normalize=True)
		self.conv1.aggr = 'mean'
		self.transition = nn.Sequential(
			nn.ReLU(),
			nn.BatchNorm1d(nhid),
			nn.Dropout(p=dropout)
		)
		self.conv2 = SAGEConv(nhid, nhid, normalize=True)
		self.conv2.aggr = 'mean'

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = self.transition(x)
		x = self.conv2(x, edge_index)
		return x


class GIN(nn.Module):
	def __init__(self, nfeat, nhid, dropout=0.5):
		super(GIN, self).__init__()

		self.mlp1 = nn.Sequential(
			spectral_norm(nn.Linear(nfeat, nhid)),
			nn.ReLU(),
			nn.BatchNorm1d(nhid),
			spectral_norm(nn.Linear(nhid, nhid)),
		)
		self.conv1 = GINConv(self.mlp1)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		return x


def drop_feature(x, drop_prob, sens_idx, sens_flag=True):
	drop_mask = torch.empty(
		(x.size(1), ),
		dtype=torch.float32,
		device=x.device).uniform_(0, 1) < drop_prob

	x = x.clone()
	drop_mask[sens_idx] = False

	x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

	# Flip sensitive attribute
	if sens_flag:
		x[:, sens_idx] = 1-x[:, sens_idx]

	return x