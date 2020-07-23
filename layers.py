import tensorflow as tf
import numpy as np

class DotProductAttention(tf.keras.layers.Layer):
	def __init__(self, clip = None, return_logits = False, head_depth = 16, inf = 1e+10, **kwargs):
		super().__init__(**kwargs)
		self.clip = clip
		self.return_logits = return_logits
		self.inf = inf
		dk = tf.cast(head_depth, tf.float32)
		self.scale = tf.math.sqrt(dk) 

	def call(self, x, mask = None):
		Q, K, V = x
		logits = tf.matmul(Q, K, transpose_b = True) / self.scale
		""" Q: (batch, n_heads, q_seq(=n_nodes or =1), head_depth)
			K: (batch, n_heads, k_seq(=n_nodes), head_depth)
			logits: (batch, n_heads, q_seq(this could be 1), k_seq)
		"""
		if self.clip is not None:
			logits = self.clip * tf.math.tanh(logits)
			
		if self.return_logits:
			if mask is not None:
				logits = tf.where(tf.transpose(mask, perm=(0, 2, 1)), tf.ones_like(logits) * (-np.inf), logits)
			return logits

		if mask is not None:
			logits = tf.where(mask[:,None,None,:,0], tf.ones_like(logits) * (-np.inf), logits)
			""" mask: (batch, n_nodes, 1), tf.Tensor([[ True], [ True], [False]])
				mask[:,None,None,:,0]: (batch, 1, 1, n_nodes) ==> broadcast depending on logits shape
				[True] -> [1 * -np.inf], [False] -> [logits]
			"""
		probs = tf.nn.softmax(logits, axis = -1)
		return tf.matmul(probs, V)

class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, n_heads = 8, embed_dim = 128, clip = None, return_logits = None, spilt_Wq = None, **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.embed_dim = embed_dim
		self.head_depth = self.embed_dim // self.n_heads
		if self.embed_dim % self.n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")
		
		self.return_logits = return_logits
		self.spilt_Wq = spilt_Wq 
		stdv = 1./tf.math.sqrt(tf.cast(embed_dim, tf.float32))
		init = tf.keras.initializers.RandomUniform(minval = -stdv, maxval = stdv)# init = tf.random_uniform_initializer(minval = -stdv, maxval= stdv)

		self.attention = DotProductAttention(clip = clip, return_logits = return_logits, head_depth = self.head_depth)
		self.Wk = tf.keras.layers.Dense(self.embed_dim, use_bias = False, kernel_initializer = init)# (embed_dim, embed_dim)
		
		if self.return_logits is None:
			self.Wv = tf.keras.layers.Dense(self.embed_dim, use_bias = False, kernel_initializer = init)# (embed_dim, embed_dim)
			self.Wout = tf.keras.layers.Dense(self.embed_dim, use_bias = False, kernel_initializer = init)# (embed_dim, embed_dim)
			if self.spilt_Wq:
				self.Wq_fixed = tf.keras.layers.Dense(self.embed_dim, use_bias = False, kernel_initializer = init, name='wq_fixed')# torch.nn.Linear(embed_dim, embed_dim)
				self.Wq_step = tf.keras.layers.Dense(self.embed_dim, use_bias = False, kernel_initializer = init, name='wq_step')# torch.nn.Linear(embed_dim, embed_dim)
			else:
				self.Wq = tf.keras.layers.Dense(self.embed_dim, use_bias = False, kernel_initializer = init)# torch.nn.Linear(embed_dim, embed_dim)
			
	def split_heads(self, T, batch):
		""" https://qiita.com/halhorn/items/c91497522be27bde17ce
			T: (batch, n_nodes, self.embed_dim)
			T reshaped: (batch, n_nodes, self.n_heads, self.head_depth)
			return: (batch, self.n_heads, n_nodes, self.head_depth)
		"""
		T = tf.reshape(T, (batch, -1, self.n_heads, self.head_depth))
		return tf.transpose(T, perm=(0,2,1,3))

	def combine_heads(self, T, batch):
		""" T: (batch, self.n_heads, n_nodes, self.head_depth)
			T transposed: (batch, n_nodes, self.n_heads, self.head_depth)
			return: (batch, n_nodes, self.embed_dim)
		"""
		T = tf.transpose(T, perm=(0,2,1,3))
		return tf.reshape(T, (batch, -1, self.embed_dim))	
	
	def call(self, x, mask = None):
		"""	q, k, v = x
			encoder arg x: [x, x, x]
			shape of q: (batch, n_nodes, embed_dim)
			output[0] - output[h_heads-1]: (batch, n_nodes, head_depth)
			--> concat output: (batch, n_nodes, head_depth * h_heads)
			return output: (batch, n_nodes, embed_dim)
		"""
		q, k, v = x
		batch = k.shape[0]

		if self.return_logits:
			return self.attention([q, self.Wk(k), None], mask = mask)
		
		if self.spilt_Wq:
			Q = self.Wq_fixed(q[:,:,:self.embed_dim]) + self.Wq_step(q[:,:,self.embed_dim:])
		else:
			Q = self.Wq(q)
			
		K, V = self.Wk(k), self.Wv(v)	
		output = self.attention([self.split_heads(T, batch) for T in [Q, K, V]], mask = mask)
		output = self.combine_heads(output, batch)
		return self.Wout(output)

if __name__ == '__main__':
	mha = MultiHeadAttention(n_heads = 8, embed_dim = 128, name = 'MHA')
	batch, n_nodes, embed_dim = 5, 21, 128
	x = tf.random.uniform((batch, n_nodes, embed_dim), dtype = tf.float32)
	output = mha([x,x,x])
	print(output.shape)

	# for w in mha.trainable_variables:# non_trainable_weights:
	# 	print(w.name, w.numpy())
	

	