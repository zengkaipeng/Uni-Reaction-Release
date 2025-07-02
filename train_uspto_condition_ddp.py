from model.conditions.pretrain_gnns import PretrainGIN
gnn = PretrainGIN(num_layer=5, emb_dim=300, drop_ratio=0.1)

gnn.requires_grad_(False)

freeze_layers = 2

for x in range(freeze_layers, 5):
	gnn.gnns[x].requires_grad_(True)
	gnn.batch_norms[x].requires_grad_(True)


for k, v in gnn.named_parameters():
	print(k, v.requires_grad)

