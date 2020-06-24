import tensorflow as tf
import numpy as np

class MultiHeadAttention(tf.keras.layers.Layer):
	""" Attention Layer - multi-head scaled dot product attention (for encoder and decoder)
		Args:
			num_heads: number of attention heads which will be computed in parallel
			embed_dim: embedding size of output features
		Call arguments:
			q: query, shape (..., seq_len_q, depth_q)
			k: key, shape == (..., seq_len_k, depth_k)
			v: value, shape == (..., seq_len_v, depth_v)
			mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k) or None.
			Since we use scaled-product attention, we assume seq_len_k = seq_len_v
		Returns:
			  attention outputs of shape (batch_size, seq_len_q, embed_dim)
	"""

	def __init__(self, n_heads = 8, embed_dim = 128, **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.embed_dim = embed_dim
		self.head_depth = self.embed_dim // self.n_heads

		if self.embed_dim % self.n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")

		# define weight matrices
		self.Wq = tf.keras.layers.Dense(self.embed_dim, use_bias=False)  # (d_q, embed_dim)
		self.Wk = tf.keras.layers.Dense(self.embed_dim, use_bias=False)  # (d_k, embed_dim)
		self.Wv = tf.keras.layers.Dense(self.embed_dim, use_bias=False)  # (d_v, embed_dim)

		self.W_out = tf.keras.layers.Dense(self.embed_dim, use_bias=False)  # (embed_dim, embed_dim)

	def split_heads(self, tensor, batch_size):
		"""Function for computing attention on several heads simultaneously
		Splits last dimension of a tensor into (num_heads, head_depth).
		Then we transpose it as (batch_size, num_heads, ..., head_depth) so that we can use broadcast
		"""
		tensor = tf.reshape(tensor, (batch_size, -1, self.n_heads, self.head_depth))
		return tf.transpose(tensor, perm=[0, 2, 1, 3])

	# treats first parameter q as input, and  k, v as parameters, so input_shape=q.shape
	def call(self, x, mask=None):
		# shape of q: (batch_size, seq_len_q, d_q)
		# encoder q = k = v
		q, k, v = x
		batch_size = tf.shape(q)[0]

		# compute Q = q * w_q, ...
		Q = self.Wq(q)  # (batch_size, seq_len_q, d_q) x (d_q, embed_dim) --> (batch_size, seq_len_q, embed_dim)
		K = self.Wk(k)  # ... --> (batch_size, seq_len_k, embed_dim)
		V = self.Wv(v)  # ... --> (batch_size, seq_len_v, embed_dim)

		# split heads: embed_dim = num_heads * head_depth + reshape
		Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, head_depth)
		K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_k, head_depth)
		V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_v, head_depth)

		# rescaling
		K_dim = tf.cast(tf.shape(K)[-1], tf.float32)
		logits = logits = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(K_dim)
		# similarity between context vector Q and key K // self-similarity in case of self-attention
	 	# (batch_size, num_heads, seq_len_q, seq_len_k)
	    # seq_len_q = n_nodes for encoder self-attention
	    # seq_len_q = 1 for decoder context-vector attention
	    # seq_len_k = n_nodes for both encoder & decoder

		if mask is not None:
			# we need to reshape mask:
			# (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
			# so that we will be able to do a broadcast:
			# (batch_size, num_heads, seq_len_q, seq_len_k) + (batch_size, 1, seq_len_q, seq_len_k)
			mask = mask[:, tf.newaxis, :, :]

			# we use tf.where since 0*-np.inf returns nan, but not -np.inf
			# logits = tf.where(
			#                     tf.broadcast_to(mask, logits.shape), tf.ones_like(logits) * (-np.inf),
			#                     logits
			#                      )

			logits = tf.where(mask,
									tf.ones_like(logits) * (-np.inf),
									logits)

		probs = tf.nn.softmax(logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

		# Replace NaN by zeros (tf.nn.softmax returns NaNs for masked rows)
		probs = tf.where(tf.math.is_nan(probs), tf.zeros_like(probs), probs)

		# seq_len_k = seq_len_v
		attention = tf.matmul(probs, V)  # (batch_size, num_heads, seq_len_q, head_depth)

		# transpose back to (batch_size, seq_len_q, num_heads, head_depth)
		attention = tf.transpose(attention, perm=[0, 2, 1, 3])

		# concatenate heads (last 2 dimensions)
		attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len_q, embed_dim)

		# project output to the same dimension
		# this is equiv. to sum in the article (project heads with W_o and sum), beacuse of block-matrix multiplication
		#e.g. https://math.stackexchange.com/questions/2961550/matrix-block-multiplication-definition-properties-and-applications

		output = self.W_out(attention)  # (batch_size, seq_len_q, embed_dim)

		return output

if __name__ == '__main__':
	mha = MultiHeadAttention(n_heads = 8, embed_dim = 128, name='MHA')# input_shape[2] = embed_dim = 128
	x = tf.ones((2,21,128), dtype = tf.float32)
	output = mha([x,x,x])
	print(output.shape)


	