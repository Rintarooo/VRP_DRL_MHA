import torch
import torch.nn as nn


class Env():
	def __init__(self, x, node_embeddings):
		super().__init__()
		""" depot_xy: (batch, 2)
			customer_xy: (batch, n_nodes-1, 2)
			--> self.xy: (batch, n_nodes, 2)
				Coordinates of depot + customer nodes
			demand: (batch, n_nodes-1)
			
			is_next_depot: (batch, 1), e.g. [[True], [True], ...]
			Nodes that have been visited will be marked with True.
		"""
		self.depot_xy, customer_xy, self.demand = x
		self.depot_xy, customer_xy, self.demand = self.depot_xy, customer_xy, self.demand
		self.xy = torch.cat([self.depot_xy[:, None, :], customer_xy], 1)
		self.batch, self.n_nodes, _ = self.xy.size()
		self.node_embeddings = node_embeddings
		self.embed_dim = node_embeddings.size(-1)

		self.is_next_depot = torch.ones([self.batch, 1], dtype = torch.bool)
		self.visited_customer = torch.zeros((self.batch, self.n_nodes-1, 1), dtype = torch.bool)

	def get_mask_D(self, next_node, visited_mask, D):
		""" next_node: ([[0],[0],[not 0], ...], (batch, 1), dtype = torch.int32), [0] denotes going to depot
			visited_mask **includes depot**: (batch, n_nodes, 1)
			visited_mask[:,1:,:] **excludes depot**: (batch, n_nodes-1, 1)
			customer_idx **excludes depot**: (batch, 1), range[0, n_nodes-1] e.g. [[3],[0],[5],[11], ...], [0] denotes 0th customer, not depot
			self.demand **excludes depot**: (batch, n_nodes-1)
			selected_demand: (batch, 1)
			if next node is depot, do not select demand
			D: (batch, 1), D denotes "remaining vehicle capacity"
			self.capacity_over_customer **excludes depot**: (batch, n_nodes-1)
			visited_customer **excludes depot**: (batch, n_nodes-1, 1)
		 	is_next_depot: (batch, 1), e.g. [[True], [True], ...]
		 	return mask: (batch, n_nodes, 1)		
		"""
		self.is_next_depot = next_node == 0
		D = D.masked_fill(self.is_next_depot == True, 1.0)
		self.visited_customer = self.visited_customer | visited_mask[:,1:,:]
		customer_idx = torch.argmax(visited_mask[:,1:,:].type(torch.long), dim = 1)
		selected_demand = torch.gather(input = self.demand, dim = 1, index = customer_idx)
		D = D - selected_demand * (1.0 - self.is_next_depot.float())
		capacity_over_customer = self.demand > D
		mask_customer = capacity_over_customer[:, :, None] | self.visited_customer

		# print('mask_customer[0]', mask_customer[0])
		mask_depot = self.is_next_depot & (torch.sum((mask_customer == False).type(torch.long), dim = 1) > 0)
		# print('mask_depot', mask_depot[0])

		""" # mask_depot = tf.math.logical_not(tf.reduce_all(mask_customer, axis = 1))
			tf.reduce_all: if there's any False on the specified axis, return False
			# mask_depot = self.is_next_depot | tf.reduce_all(mask_customer, axis = 1)
			We can choose depot if 1) we are not in depot or 2) all nodes are visited
			if the mask for customer nodes are all True, mask_depot should be False so that the vehicle can return back to depot 
			even if some of the mask for customer nodes are False, mask_depot should be False so that vehicle could go back to the depot
			the vechile must not be at the depot in a low but it can stay at the depot when the mask for customer nodes are all True
		"""
		return torch.cat([mask_depot[:, None, :], mask_customer], dim = 1), D
	
	def _get_step(self, next_node, D):
		""" next_node **includes depot** : (batch, 1) tf.int32, range[0, n_nodes-1]
			--> one_hot: (batch, 1, n_nodes)
			prev_node_embedding: (batch, 1, embed_dim)
		"""
		one_hot = torch.eye(self.n_nodes)[next_node]		
		visited_mask = one_hot.type(torch.bool).permute(0,2,1)

		mask, D = self.get_mask_D(next_node, visited_mask, D)
		# self.demand = tf.where(self.visited_customer[:,:,0], tf.zeros_like(self.demand), self.demand)
		
		# prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].repeat(1,1,self.embed_dim))
		prev_node_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = next_node[:,:,None].expand(self.batch,1,self.embed_dim))

		context = torch.cat([prev_node_embedding, D[:,:,None]], dim = -1)
		return mask, context, D

	def _create_t1(self):
		mask_t1 = self.create_mask_t1()
		step_context_t1, D_t1 = self.create_context_D_t1()
		return mask_t1, step_context_t1, D_t1

	def create_mask_t1(self):
		mask_customer = self.visited_customer
		mask_depot = torch.ones([self.batch, 1, 1], dtype = torch.bool)
		return torch.cat([mask_depot, mask_customer], dim = 1)

	def create_context_D_t1(self):
		D_t1 = torch.ones([self.batch, 1], dtype=torch.float)
		depot_idx = torch.zeros([self.batch, 1], dtype = torch.long)# long == int64
		# depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].repeat(1,1,self.embed_dim))
		depot_embedding = torch.gather(input = self.node_embeddings, dim = 1, index = depot_idx[:,:,None].expand(self.batch,1,self.embed_dim))
		# https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
		return torch.cat([depot_embedding, D_t1[:,:,None]], dim = -1), D_t1

	def get_log_likelihood(self, _log_p, pi):
		# Get log_p corresponding to selected actions
		log_p = torch.gather(input = _log_p, dim = -1, index = pi[:,:,None])
		return torch.sum(log_p, 1)

	def get_costs(self, pi):
		""" self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			pi: (batch, decode_step), predicted tour
			d: (batch, decode_step, 2)
			Note: first element of pi is not depot, the first selected node in the path
		"""
		d = torch.gather(input = self.xy, dim = 1, index = pi[:,:,None].repeat(1,1,2))
		# d = torch.gather(input = self.xy, dim = 1, index = pi[:,:,None].expand(self.batch,pi.size(1),2))
		return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p = 2, dim = 2), dim = 1)
				+ (d[:, 0] - self.depot_xy).norm(p = 2, dim = 1)# distance from depot to first selected node
				+ (d[:, -1] - self.depot_xy).norm(p = 2, dim = 1)# distance from last selected node (!=0 for graph with longest path) to depot
				)



class Sampler(nn.Module):
	""" logits: (batch, n_nodes)
			TopKSampler <-- greedy; sample one with biggest probability
			CategoricalSampler <-- sampling; randomly sample one from possible distribution based on probability
	"""
	def __init__(self, n_samples = 1, **kwargs):
		super().__init__(**kwargs)
		self.n_samples = n_samples
		
class TopKSampler(Sampler):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
	
	def forward(self, logits):
		return torch.topk(logits, self.n_samples, dim = 1)[1]

class CategoricalSampler(Sampler):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def forward(self, logits):
		# https://discuss.pytorch.org/t/backpropagate-on-a-stochastic-variable/3496/13
		# from torch.distributions import Categorical
		# m = Categorical(probs = logits)
		# return m.sample(sample_shape = (self.n_samples, )).transpose(-1,-2)
		# return logits.exp().multinomial(self.n_samples, out = logits)
		return torch.multinomial(logits.exp(), self.n_samples)
