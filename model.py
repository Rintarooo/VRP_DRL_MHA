import tensorflow as tf

from data import generate_data
from encoder import GraphAttentionEncoder
from decoder import Sampler, TopKSampler, CategoricalSampler, DecoderCell

class AttentionModel(tf.keras.Model):
	
	def __init__(self, embed_dim = 128, n_encode_layers=3, n_heads = 8, tanh_clipping=10.):
		super().__init__()
		if embed_dim % n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")

		self.encoder = GraphAttentionEncoder(embed_dim = embed_dim, n_heads = n_heads, n_layers = n_encode_layers)
		self.decoder = DecoderCell(n_heads = n_heads, clip = tanh_clipping)

	def get_mask_D(self, next_node, visited_mask, D):
		""" next_node: ([[0],[0],[not 0], ...], (batch, 1), dtype = tf.int32), [0] denotes going to depot
			visited_mask **includes depot**: (batch, n_nodes, 1)
			visited_mask[:,1:,:] **excludes depot**: (batch, n_nodes-1, 1)
			customer_idx **excludes depot**: (batch, 1), range[0, n_nodes-1] e.g. [[3],[0],[5],[11], ...]
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
		D = tf.where(self.is_next_depot, 1.0, D)
		self.visited_customer = self.visited_customer | visited_mask[:,1:,:]
		customer_idx = tf.argmax(tf.cast(visited_mask[:,1:,:], tf.int32), axis = 1)
		selected_demand = tf.gather(params = self.demand, indices = customer_idx, batch_dims = 1)
		D = D - selected_demand * (1.0 - tf.cast(self.is_next_depot, tf.float32))
		capacity_over_customer = self.demand > D
		mask_customer = capacity_over_customer[:, :, None] | self.visited_customer
		mask_depot = tf.math.logical_not(tf.reduce_all(mask_customer, axis = 1))
		""" We can choose depot if 1) we are not in depot or 2) all nodes are visited
			tf.reduce_all: if there's any False on the specified axis, return False
			if all customer nodes are True, mask_depot should be False so that the vehicle returns back to depot 
		"""
		return tf.concat([mask_depot[:, None, :], mask_customer], axis = 1), D
	
	def get_context_mask(self, node_embeddings, graph_embedding, next_node, D):
		""" next_node **includes depot** : (batch, 1) tf.int32, range[0, n_nodes-1]
			--> one_hot: (batch, 1, n_nodes)
			prev_node_embedding: (batch, 1, embed_dim)
		"""
		one_hot = tf.one_hot(indices = next_node, depth = self.n_nodes)		
		visited_mask = tf.transpose(tf.cast(one_hot, dtype = tf.bool), (0,2,1))
		mask, D = self.get_mask_D(next_node, visited_mask, D)
		prev_node_embedding = tf.matmul(one_hot, node_embeddings)
		context = tf.concat([graph_embedding[:,None,:], prev_node_embedding, D[:,:,None]], axis = -1)
		return mask, context, D

	def create_context_mask_t1(self, node_embeddings, graph_embedding):
		mask_t1 = self.create_mask_t1()
		context_t1, D_t1 = self.create_context_t1(node_embeddings, graph_embedding)
		return mask_t1, context_t1, D_t1

	def create_context_t1(self, node_embeddings, graph_embedding):
		D_t1 = tf.ones([self.batch, 1], dtype=tf.float32)
		depot_idx = tf.zeros([self.batch, 1], dtype = tf.int32)
		depot_embedding = tf.gather(node_embeddings, indices = depot_idx, batch_dims = 1)
		return tf.concat([graph_embedding[:,None,:], depot_embedding, D_t1[:,:,None]], axis = -1), D_t1

	def create_mask_t1(self):
		self.visited_customer = tf.zeros((self.batch, self.n_nodes-1, 1), dtype = tf.bool)
		mask_customer = self.visited_customer
		mask_depot = tf.ones([self.batch, 1, 1], dtype = tf.bool)
		return tf.concat([mask_depot, mask_customer], axis = 1)

	def get_log_likelihood(self, _log_p, pi):
		# Get log_p corresponding to selected actions
		log_p = tf.gather_nd(_log_p, tf.expand_dims(pi, axis = -1), batch_dims = 2)
		return tf.reduce_sum(log_p, 1)

	def get_costs(self, pi):
		""" self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			pi: (batch, decode_step), predicted tour
			d: (batch, decode_step, 2)
			Note: first element of pi is not depot, the first selected node in the path
		"""
		d = tf.gather(self.xy, indices = pi, batch_dims = 1)
		return (tf.reduce_sum(tf.norm(d[:, 1:] - d[:, :-1], ord=2, axis=2), axis=1)
					+ tf.norm(d[:, 0] - self.depot_xy, ord=2, axis=1)# distance from depot to first selected node
					+ tf.norm(d[:, -1] - self.depot_xy, ord=2, axis=1))# distance from last selected node (!=0 for graph with longest path) to depot
	
	def setting(self, x, decode_type):
		""" depot_xy: (batch, 2)
			customer_xy: (batch, n_nodes-1, 2)
				--> self.xy: (batch, n_nodes, 2)
					Coordinates of depot + customer nodes
			demand: (batch, n_nodes-1)
			
			is_next_depot: (batch, 1), e.g. [[True], [True], ...]
			Nodes that have been visited will be marked with True.
		"""
		self.depot_xy, customer_xy, self.demand = x
		self.xy = tf.concat([self.depot_xy[:, None, :], customer_xy], 1)
		# self.batch, self.n_nodes, _ = tf.shape(self.xy)
		self.batch = tf.shape(self.xy)[0]
		self.n_nodes = tf.shape(self.xy)[1]
		
		self.is_next_depot = tf.ones([self.batch, 1], dtype = tf.bool)
		self.selecter = {'greedy': TopKSampler(),
						'sampling': CategoricalSampler()}.get(decode_type, None)

	# @tf.function
	def call(self, x, return_pi = False, decode_type = 'greedy'):
		""" node_embeddings: (batch, n_nodes, embed_dim)
			graph_embedding: (batch, embed_dim)
			mask: (batch, n_nodes, 1), dtype = tf.bool, [True] --> [-inf], [False] --> [logits]
			context: (batch, 1, 2*embed_dim+1)

			logits: (batch, 1, n_nodes), logits denotes the value before going into softmax
			next_node: (batch, 1), minval = 0, maxval = n_nodes-1, dtype = tf.int32
			log_p: (batch, n_nodes) <-- logits **squeezed**: (batch, n_nodes), log(exp(x_i) / exp(x).sum())
		"""
		self.setting(x, decode_type)
		node_embeddings, graph_embedding = self.encoder(x)

		mask, context, D = self.create_context_mask_t1(node_embeddings, graph_embedding) 
		log_ps = tf.TensorArray(dtype = self.decoder.dtype, size = 0, dynamic_size = True, element_shape = (self.batch, self.n_nodes))
		# tf.float32
		tours = tf.TensorArray(dtype = tf.int32, size = 0, dynamic_size = True, element_shape = (self.batch,))
		
		#tf.while_loop
		for i in tf.range(self.n_nodes*2):
			logits = self.decoder([context, node_embeddings], mask) 
			next_node = self.selecter(tf.squeeze(logits, axis = 1))
			mask, context, D = self.get_context_mask(node_embeddings, graph_embedding, next_node, D)
	
			tours = tours.write(i, tf.squeeze(next_node, axis = 1))
			log_p = tf.nn.log_softmax(tf.squeeze(logits, axis = 1), axis = -1)
			log_ps = log_ps.write(i, log_p)

			if tf.reduce_all(self.visited_customer):
				break

		pi = tf.transpose(tours.stack(), perm = (1,0))
		ll = self.get_log_likelihood(tf.transpose(log_ps.stack(), perm = (1,0,2)), pi)
		cost = self.get_costs(pi)
		if return_pi:
			return cost, ll, pi
		return cost, ll
		
if __name__ == '__main__':
	# tf.config.experimental_run_functions_eagerly(True)
	model = AttentionModel()
	dataset = generate_data(seed = 123)
	for i, data in enumerate(dataset.batch(10)):
		output = model(data, decode_type = 'sampling', return_pi = True)
		print(output[0])# cost: (batch,)
		print(output[1])# ll: (batch,)
		print(output[2])# pi: (batch, decode_step) # tour
		if i == 0:
			break

	# print('model.trainable_weights')
	# for w in model.trainable_weights:
	# 	print(w.name)
	model.summary()

