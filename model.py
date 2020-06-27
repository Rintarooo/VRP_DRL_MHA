import tensorflow as tf

from data import generate_data
from encoder import GraphAttentionEncoder
from decoder import Sampler, TopKSampler, CategoricalSampler, DecoderCell
from env import AgentVRP

class AttentionModel(tf.keras.Model):

	def __init__(self, embed_dim = 128, n_encode_layers=3,
				 n_heads = 8, tanh_clipping=10., decode_type = 'greedy'):

		super().__init__()
		head_depth = embed_dim // n_heads
		if embed_dim % n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")

		self.AgentClass = AgentVRP
		self.encoder = GraphAttentionEncoder(embed_dim = embed_dim, n_heads = n_heads, n_layers = n_encode_layers)
		self.decoder = DecoderCell(n_heads = n_heads, clip = tanh_clipping)
		self.selecter = {'greedy': TopKSampler(),
						'sampling': CategoricalSampler()}.get(decode_type, None)
		assert self.selecter is not None, 'decode_type: greedy or sampling'

	def update_context_and_mask(self, next_node, node_embeddings, graph_embedding):
		one_hot = tf.one_hot(indices = next_node, depth = self.n_nodes)
		# next_node: (batch, 1) tf.int32, range[0, n_nodes-1]--> one_hot: (batch, 1, n_nodes)		
		visited_mask = tf.transpose(tf.cast(one_hot, dtype = tf.bool), (0,2,1))
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
			mask: (batch, n_nodes, 1)
			context: (batch, 1, 2*embed_dim+1)
		"""
		node_embeddings, graph_embedding = self.encoder(x)
		self.batch, self.n_nodes, _ = tf.shape(node_embeddings)
		mask = self.create_mask()
		context = self.create_context(node_embeddings, graph_embedding)
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
				logits = self.decoder([context, node_embeddings], mask)
				# logits: (batch, 1, n_nodes), logits denotes the value before going into softmax 
				next_node = self.selecter(tf.squeeze(logits, axis = 1))
				# next_node: (batch, 1), minval = 0, maxval = n_nodes-1, dtype = tf.int32
				mask, context = self.update_context_and_mask(next_node, node_embeddings, graph_embedding)
				
				log_p = tf.nn.log_softmax(logits, axis = -1)# log(exp(x_i) / exp(x).sum())
				# log_p: (batch, 1, n_nodes) <-- logits: (batch, 1, n_nodes)
				tours.append(tf.squeeze(next_node, axis = 1))
				log_ps.append(tf.gather(tf.squeeze(log_p, axis = 1), indices = next_node, batch_dims = 1))
			# print(i)
			i += 1

		pi = tf.stack(tours, 1)
		ll = tf.squeeze(tf.reduce_sum(tf.stack(log_ps, 0), 0), axis = 1)
		cost = self.env.get_costs(pi)
		if return_pi:
			return cost, ll, pi
		return cost, ll
		
if __name__ == '__main__':
	model = AttentionModel()
	model.decode_type = 'sampling'
	dataset = generate_data()
	for i, data in enumerate(dataset.batch(4)):
		output = model(data, return_pi = True)
		print(output[0])# cost: (batch,)
		print(output[1])# ll: (batch,)
		print(output[2])# pi: (batch, decode_step) # tour
		if i == 0:
			break
	# print('model.trainable_weights')
	# for w in model.trainable_weights:
	# 	print(w.name)
	model.summary()

