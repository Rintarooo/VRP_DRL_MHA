import tensorflow as tf

from encoder_tf import GraphAttentionEncoder, get_data_onthefly
from decoder_tf import Sampler, TopKSampler, CategoricalSampler, DecoderCell
from env import AgentVRP

class AttentionModel(tf.keras.Model):

	def __init__(self, embed_dim = 128, n_encode_layers=3,
				 n_heads = 8, tanh_clipping=10., decode_type = 'greedy'):

		super().__init__()
		head_depth = embed_dim // n_heads
		if embed_dim % n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")

		# attributes for VRP problem
		self.AgentClass = AgentVRP
		
		self.encoder = GraphAttentionEncoder(embed_dim = embed_dim,
											  n_heads = n_heads,
											  n_layers = n_encode_layers)
		
		self.decoder = DecoderCell(n_heads = n_heads,
									clip = tanh_clipping)
		
		self.selecter = {'greedy': TopKSampler(),
						'categorical': CategoricalSampler()}.get(decode_type, None)
		assert self.selecter is not None, 'decode_type: greedy or categorical'

	# def get_log_likelihood(self, _log_p, pi):

	# 	# Get log_p corresponding to selected actions
	# 	log_p = tf.gather_nd(_log_p, tf.cast(tf.expand_dims(pi, axis=-1), tf.int32), batch_dims=2)
	# 	# tf.matmul

	# 	# Calculate log_likelihood
	# 	print('log_p:', log_p)
	# 	return tf.reduce_sum(log_p,1)

	def get_log_likelihood(self, _log_p, pi):
		# _log_p: (batch, n_nodes, decode_step)
		# pi: (batch, decode_step) # tour
		transpose_log_p = tf.transpose(_log_p, perm = (0,2,1))
		print('transpose_log_p[0]:', transpose_log_p[0])
		# one_hot = tf.one_hot(indices = pi, depth = self.n_nodes)
		# one_hot: (batch, decode_step, n_nodes)
		# log_p = tf.matmul(one_hot, _log_p)
		log_p = tf.gather(_log_p, indices = pi, batch_dims = 1)
		print('log_p[0]:', log_p[0])
		return tf.reduce_sum(log_p, 1)

	def update_context_and_mask(self, next_node, node_embeddings, graph_embedding):
		# next_node: (batch, 1), minval = 0, maxval = n_nodes-1, dtype = tf.int32
		# mask: (batch, n_nodes, 1)
		visited_mask = tf.zeros_like(node_embeddings[:,:,:1], dtype=tf.uint8)
		one_hot = tf.one_hot(indices = next_node, depth = self.n_nodes)
		# one_hot: (batch, 1, n_nodes)
		visited_mask += tf.transpose(tf.cast(one_hot, dtype = tf.uint8), (0,2,1))

		mask, D = self.env.think_capacity(next_node, visited_mask)
		
		prev_node_embedding = tf.matmul(one_hot, node_embeddings)
		# prev_node_embedding: (batch, 1, embed_dim)
		 
		context = tf.concat((graph_embedding[:,None,:], prev_node_embedding, D[:,:,None]), axis=-1)
		return mask, context

	def create_context(self, node_embeddings, graph_embbedding):
		D_t1 = tf.ones([self.batch, 1], dtype = tf.float32)
		depot_idx = tf.zeros([self.batch, 1], dtype = tf.int32)
		one_hot = tf.one_hot(indices = depot_idx, depth = self.n_nodes)
		depot_embedding = tf.matmul(one_hot, node_embeddings)
		# depot_embedding = tf.gather(node_embeddings, indices = depot_idx, batch_dims = 1)
		return tf.concat([graph_embbedding[:,None,:], depot_embedding, D_t1[:,:,None]], axis = -1)

	def create_mask(self):
		mask_customer = tf.zeros([self.batch, self.n_nodes-1, 1], dtype=tf.bool)
		mask_depot = tf.ones([self.batch, 1, 1], dtype = tf.bool)
		# [True] --> np.inf, [False] --> logits
		return tf.concat([mask_depot, mask_customer], axis = 1)

	def call(self, x, return_pi=False):
		""" node_embeddings: (batch, n_nodes, embed_dim)
			graph_embedding: (batch, embed_dim)
		"""
		node_embeddings, graph_embedding = self.encoder(x)
		self.batch, self.n_nodes, _ = tf.shape(node_embeddings)
		mask = self.create_mask()
		context = self.create_context(node_embeddings, graph_embedding)
		""" mask: (batch, n_nodes, 1)
			context: (batch, 1, 2*embed_dim+1)
		"""
		log_ps, tours = [], []
		self.env = self.AgentClass(x)

		# Perform decoding steps
		i = 0
		#tf.while_loop
		while not self.env.all_visited():

			if i > 0:
				self.env.reset()
				node_embeddings, graph_embedding = self.encoder(x, mask)
				context = self.create_context(node_embeddings, graph_embedding)

			while not self.env.partial_visited():

				# compute MHA decoder vectors for current mask
				logits = self.decoder([context, node_embeddings], mask)# context: (batch, 1, 3*embed_dim), node_embeddings: (batch, 1, embed_dim)
				# logits: (batch, 1, n_nodes), logits denotes the value before going into softmax 
				next_node = self.selecter(tf.squeeze(logits, axis = 1))
				# next_node: (batch, 1), minval = 0, maxval = n_nodes, dtype = tf.int32
				mask, context = self.update_context_and_mask(next_node, node_embeddings, graph_embedding)
				
				log_p = tf.nn.log_softmax(logits, axis = -1)
				# log_p: (batch, 1, n_nodes) <-- logits: (batch, 1, n_nodes)
				# log(exp(x_i) / exp(x).sum())
				
				# log_ps.append(tf.squeeze(log_p, axis = 1))
				tours.append(tf.squeeze(next_node, axis = 1))
				# print(next_node)
				
				log_ps.append(tf.gather(tf.squeeze(log_p, axis = 1), indices = next_node, batch_dims = 1))

			print(i)
			i += 1
		
		# _log_p, pi = tf.stack(log_ps, 2), tf.stack(tours, 1)
		pi = tf.stack(tours, 1)
		
		cost = self.env.get_costs(pi)

		# ll = self.get_log_likelihood(_log_p, pi)
		# _log_p: (batch, n_nodes, decode_step)
		# pi: (batch, decode_step) # tour
		ll = tf.reduce_sum(tf.stack(log_ps, 0), 0)
		
		if return_pi:
			return cost, ll, pi

		return cost, ll
		
if __name__ == '__main__':
	model = AttentionModel()
	dataset = get_data_onthefly()
	for i, data in enumerate(dataset.batch(4)):
		output = model(data, return_pi = True)
		print(output[0])
		print(output[1])
		print(output[2])
		if i == 0:
			break

