from encoder import Encoder

import torch.nn as nn
import torch

class Model(nn.Module):
	def __init__(self, embed = 128):
		super().__init__()
		self.Embed = nn.Linear(3, embed)# x,y,demand
		self.Depot_Embed = nn.Linear(2, embed)# x,y
		self.Encoder = Encoder()
		self.Decoder = Decoder()

	def forward(self, x):
		embed_x = self.init_embed(x)
		graph_embed, node_embed = self.Encoder(embed_x)
		tour, _log_p = self.Decoder(node_embed, graph_embed)

		dist = get_costs(tour, x)


	def init_embed(x):
		depot_embed_x = self.Depot_Embed(x['depot'])
		customer_embed_x = self.Embed(torch.cat((x['xy'], x['demand']), -1))
		return torch.cat((depot_embed_x, customer_embed_x), 1) 
