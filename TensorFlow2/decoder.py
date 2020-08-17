import tensorflow as tf
from layers import MultiHeadAttention, DotProductAttention
from decoder_utils import TopKSampler, CategoricalSampler, Env
from data import generate_data


class DecoderCell(tf.keras.models.Model):
	def __init__(self, embed_dim = 128, n_heads = 8, clip = 10., **kwargs):
		super().__init__(**kwargs)
		
		self.Wk1 = tf.keras.layers.Dense(embed_dim, use_bias = False)# torch.nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wv = tf.keras.layers.Dense(embed_dim, use_bias = False)
		self.Wk2 = tf.keras.layers.Dense(embed_dim, use_bias = False)
		self.Wq_fixed = tf.keras.layers.Dense(embed_dim, use_bias = False)
		self.Wout = tf.keras.layers.Dense(embed_dim, use_bias = False)
		self.Wq_step = tf.keras.layers.Dense(embed_dim, use_bias = False)
		
		self.MHA = MultiHeadAttention(n_heads = n_heads, embed_dim = embed_dim, need_W = False)
		self.SHA = DotProductAttention(clip = clip, return_logits = True, head_depth = embed_dim)
		# SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
		self.env = Env
	
	# @tf.function
	def compute_static(self, node_embeddings, graph_embedding):
		Q_fixed = self.Wq_fixed(graph_embedding[:,None,:])
		K1 = self.Wk1(node_embeddings)
		V = self.Wv(node_embeddings)
		K2 = self.Wk2(node_embeddings)
		return Q_fixed, K1, V, K2

	@tf.function
	def _compute_mha(self, Q_fixed, step_context, K1, V, K2, mask):
		Q_step = self.Wq_step(step_context)
		Q1 = Q_fixed + Q_step
		Q2 = self.MHA([Q1, K1, V], mask = mask)
		Q2 = self.Wout(Q2)
		logits = self.SHA([Q2, K2, None], mask = mask)
		return tf.squeeze(logits, axis = 1)
		
	# @tf.function
	def call(self, x, encoder_output, return_pi = False, decode_type = 'sampling'):
		""" context: (batch, 1, 2*embed_dim+1)
			tf.concat([graph embedding[:,None,:], previous node embedding, remaining vehicle capacity[:,:,None]], axis = -1)
			encoder output 
			==> graph embedding: (batch, embed_dim) 
			==> node_embeddings: (batch, n_nodes, embed_dim)
			previous node embedding: (batch, n_nodes, embed_dim)
			remaining vehicle capacity(= D): (batch, 1)
			
			mask: (batch, n_nodes, 1), dtype = tf.bool, [True] --> [-inf], [False] --> [logits]
			context: (batch, 1, 2*embed_dim+1)

			squeezed logits: (batch, n_nodes), logits denotes the value before going into softmax
			next_node: (batch, 1), minval = 0, maxval = n_nodes-1, dtype = tf.int32
			log_p: (batch, n_nodes) <-- squeezed logits: (batch, n_nodes), log(exp(x_i) / exp(x).sum())
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
			next_node = selecter(log_p)
			mask, step_context, D = env._get_step(next_node, D)
	
			tours = tours.write(i, tf.squeeze(next_node, axis = 1))
			log_ps = log_ps.write(i, log_p)
			# tf.print(type(env.visited_customer))

			# if tf.reduce_all(env.visited_customer):
			# 	break

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
