import tensorflow as tf
from layers import MultiHeadAttention, DotProductAttention
from decoder_utils import TopKSampler, CategoricalSampler, Env
from data import generate_data


class DecoderCell(tf.keras.models.Model):
	def __init__(self, embed_dim = 128, n_heads = 8, clip = 10., **kwargs):
		super().__init__(**kwargs)
		if embed_dim % n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")
		self.n_heads = n_heads
		self.embed_dim = embed_dim
		self.clip = clip

		self.Wk1 = tf.keras.layers.Dense(self.embed_dim, use_bias = False, name='wk')# (embed_dim, embed_dim)
		self.Wv = tf.keras.layers.Dense(self.embed_dim, use_bias = False, name='wv')# (embed_dim, embed_dim)
		self.Wk2 = tf.keras.layers.Dense(self.embed_dim, use_bias = False, name='wk_tanh')# (embed_dim, embed_dim)
		self.Wq_fixed = tf.keras.layers.Dense(self.embed_dim, use_bias = False, name='wq_context')# torch.nn.Linear(embed_dim, embed_dim)

		self.Wout = tf.keras.layers.Dense(self.embed_dim, use_bias = False, name='w_out')# (embed_dim, embed_dim)
		self.Wq_step = tf.keras.layers.Dense(self.embed_dim, use_bias = False, name='wq_step_context')# torch.nn.Linear(embed_dim, embed_dim)
		
		self.MHA1 = MultiHeadAttention(n_heads = self.n_heads, embed_dim = embed_dim, not_need_W = True)
		# self.MHA2 = MultiHeadAttention(n_heads = 1, embed_dim = embed_dim, clip = self.clip, not_need_W = True, return_logits = True)
		self.MHA2 = DotProductAttention(clip = clip, return_logits = True, head_depth = self.embed_dim)# because n_heads = 1
		self.env = Env
	
	# @tf.function
	def compute_static(self, node_embeddings, graph_embedding):
		Q_fixed = self.Wq_fixed(graph_embedding[:,None,:])
		K1 = self.Wk1(node_embeddings)
		V = self.Wv(node_embeddings)
		K2 = self.Wk2(node_embeddings)
		return Q_fixed, K1, V, K2

	# @tf.function
	def _compute_mha(self, Q_fixed, step_context, K1, V, K2, mask):
		Q_step = self.Wq_step(step_context)
		Q = Q_fixed + Q_step
		query = self.MHA1([Q, K1, V], mask = mask)
		query = self.Wout(query)
		logits = self.MHA2([query, K2, None], mask = mask)
		return tf.squeeze(logits, axis = 1)
		
	
	def call(self, x, encoder_output, return_pi = False, decode_type = 'sampling'):
		""" context: (batch, 1, 2*embed_dim+1)
			context = tf.concat([], axis = -1)
			tf.concat([graph embedding[:,None,:], previous node embedding, remaining vehicle capacity[:,:,None]], axis = -1)
			graph embedding: (batch, embed_dim)
			previous node embedding: (batch, n_nodes, embed_dim)
			remaining vehicle capacity(= D): (batch, 1) 
			node_embeddings: (batch, n_nodes, embed_dim)
		"""

		""" node_embeddings: (batch, n_nodes, embed_dim)
			graph_embedding: (batch, embed_dim)
			mask: (batch, n_nodes, 1), dtype = tf.bool, [True] --> [-inf], [False] --> [logits]
			context: (batch, 1, 2*embed_dim+1)

			logits: (batch, 1, n_nodes), logits denotes the value before going into softmax
			next_node: (batch, 1), minval = 0, maxval = n_nodes-1, dtype = tf.int32
			log_p: (batch, n_nodes) <-- logits **squeezed**: (batch, n_nodes), log(exp(x_i) / exp(x).sum())
		"""
		node_embeddings, graph_embedding = encoder_output
		Q_fixed, K1, V, K2 = self.compute_static(node_embeddings, graph_embedding)

		env = Env(x, node_embeddings)
		mask, step_context, D = env._create_t1()

		selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
 
		log_ps = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True, element_shape = (env.batch, env.n_nodes))	
		tours = tf.TensorArray(dtype = tf.int32, size = 0, dynamic_size = True, element_shape = (env.batch,))
		
		for i in tf.range(env.n_nodes*2):
			logits = self._compute_mha(Q_fixed, step_context, K1, V, K2, mask) 
			log_p = tf.nn.log_softmax(logits, axis = -1)
			# print(log_p[0])
			next_node = selecter(log_p)
			mask, step_context, D = env._get_step(next_node, D)
	
			tours = tours.write(i, tf.squeeze(next_node, axis = 1))
			log_ps = log_ps.write(i, log_p)

			if tf.reduce_all(env.visited_customer):
				break

		pi = tf.transpose(tours.stack(), perm = (1,0))
		ll = env.get_log_likelihood(tf.transpose(log_ps.stack(), perm = (1,0,2)), pi)
		cost = env.get_costs(pi)
		if return_pi:
			return cost, ll, pi
		return cost, ll

if __name__ == '__main__':
	batch, n_nodes, embed_dim = 10, 21, 128
	dataset = generate_data()
	decoder = DecoderCell(embed_dim, n_heads = 8, clip = 10.)
	for i, data in enumerate(dataset.batch(batch)):
		node_embeddings = tf.ones((batch, n_nodes, embed_dim), dtype = tf.float32)
		graph_embedding = tf.ones((batch, embed_dim), dtype = tf.float32)
		encoder_output = (node_embeddings, graph_embedding)
		cost, ll, pi = decoder(data, encoder_output, return_pi = True, decode_type = 'sampling')
		print('cost', cost)
		print('ll', ll)
		print('pi', pi)	
		if i == 0:
			break
	
	decoder.summary()

	for w in decoder.trainable_weights:# non_trainable_weights:
		print(w.name, w.shape)
