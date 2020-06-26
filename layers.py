import tensorflow as tf
import numpy as np

class DotProductAttention(tf.keras.layers.Layer):
	def __init__(self, clip = None, return_logits = False, inf = 1e+10, **kwargs):
		super().__init__(**kwargs)
		self.clip = clip
		self.return_logits = return_logits
		self.inf = inf

	def call(self, x, mask = None):
		Q, K, V = x
		# print('Q.shape:', Q.shape)
		# print('K.shape:', K.shape)
		# print('V.shape:', V.shape)
		d_k = tf.cast(tf.shape(K)[-1], tf.float32)
		logits = tf.matmul(Q, K, transpose_b = True) / tf.math.sqrt(d_k)
		# (batch, seq_len, head_depth) * (batch, head_depth, seq_len)
		# = (batch, seq_len, seq_len)
		if self.clip is not None:
			logits = self.clip * tf.tanh(logits)
		if mask is not None:
			logits = tf.where(tf.transpose(mask, perm=(0, 2, 1)), tf.ones_like(logits) * (-np.inf), logits)
			""" mask: tf.Tensor([[ True], [ True], [False]])
				[True] -> [1 * -np.inf], [False] -> [logits]
			"""
		if self.return_logits:
			return logits
		probs = tf.nn.softmax(logits, axis = -1)
		return tf.matmul(probs, V)

class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, n_heads = 8, embed_dim = 128, **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.embed_dim = embed_dim
		self.head_depth = self.embed_dim // self.n_heads

		if self.embed_dim % self.n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")

		self.Wq = tf.keras.layers.Dense(self.head_depth, use_bias=False)  # torch.nn.Linear(embed_dim, d_q(=head_depth))
		self.Wk = tf.keras.layers.Dense(self.head_depth, use_bias=False)  # (embed_dim, d_k)
		self.Wv = tf.keras.layers.Dense(self.head_depth, use_bias=False)  # (embed_dim, d_v)		
		self.Wq_layers = [self.Wq for _ in range(n_heads)]
		self.Wk_layers = [self.Wk for _ in range(n_heads)]
		self.Wv_layers = [self.Wv for _ in range(n_heads)]
		self.attentions = [DotProductAttention() for _ in range(n_heads)]
		self.concat = tf.keras.layers.Concatenate(axis = -1)
		self.Wo = tf.keras.layers.Dense(self.embed_dim, use_bias=False)# (embed_dim, embed_dim)
	
	def call(self, x, mask = None):
		"""	q, k, v = x
			encoder x: [x, x, x]
			shape of q: (batch, n_nodes, embed_dim)
			output: (batch, n_nodes, embed_dim)
		"""
		q, k, v = x
		output = [attention([Wq(q), Wk(k), Wv(v)], mask = mask)
			for attention, Wq, Wk, Wv in zip(self.attentions, self.Wq_layers, self.Wk_layers, self.Wv_layers)]
		output = self.concat(output)
		output = self.Wo(output)
		return output

if __name__ == '__main__':
	mha = MultiHeadAttention(n_heads = 8, embed_dim = 128, name = 'MHA')# input_shape[2] = embed_dim = 128
	batch, n_nodes, embed_dim = 5, 21, 128
	x = tf.random.uniform((batch, n_nodes, embed_dim), dtype = tf.float32)
	output = mha([x,x,x])
	print(output.shape)


	