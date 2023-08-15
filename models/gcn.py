import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score


def train_GCN(args, data):
    # setup
	epoch_times = []

	# initialize model and optimizer
	model = GCN(nfeat=data.n_features,
				nhid=args.hidden,
				nclass=data.n_classes,
				dropout=args.dropout)
	model = model.to(args.device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	# training loop
	for epoch in range(args.epochs+1):
		t = time.time()

		# training with binary cross-entropy loss
		model.train()
		optimizer.zero_grad()
		output = model(data.features, data.edge_index)

		loss_train = F.binary_cross_entropy_with_logits(output[data.idx_train], data.labels[data.idx_train].unsqueeze(1).float().to(args.device))
		loss_train.backward()
		optimizer.step()
  
		# performance on validation set
		model.eval()
		output = model(data.features, data.edge_index)
		loss_val = F.binary_cross_entropy_with_logits(output[data.idx_val], data.labels[data.idx_val].unsqueeze(1).float().to(args.device))
		val_auc_roc = roc_auc_score(data.labels[data.idx_val].cpu().numpy(), output[data.idx_val].detach().cpu().numpy())

		epoch_times.append(time.time() - t)

		if args.verbose:
			print(f"Epoch {epoch}: train_loss: {loss_train.item():.4f} | val_auc_roc: {val_auc_roc:.4f} | time: {time.time() - t}")

	# save model
	torch.save(model.state_dict(), data.model_fname)
	
	avg_epoch_time = round(sum(epoch_times)/len(epoch_times), 4)

	return val_auc_roc, avg_epoch_time, model


class GCN(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(GCN, self).__init__()
		self.body = GCN_Body(nfeat,nhid,dropout)
		self.fc = nn.Linear(nhid, nclass)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, x, edge_index):
		x = self.body(x, edge_index)
		x = self.fc(x)
		return x


class GCN_Body(nn.Module):
	def __init__(self, nfeat, nhid, dropout):
		super(GCN_Body, self).__init__()
		self.gc1 = GCNConv(nfeat, nhid)

	def forward(self, x, edge_index):
		x = self.gc1(x, edge_index)
		return x