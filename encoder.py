import tensorflow as tf
import numpy as np
from data import generate_data

class MultiHeadAttention(tf.keras.layers.Layer):
	""" Attention Layer - multi-head scaled dot product attention (for encoder and decoder)

		Args:
			num_heads: number of attention heads which will be computed in parallel
			d_model: embedding size of output features

		Call arguments:
			q: query, shape (..., seq_len_q, depth_q)
			k: key, shape == (..., seq_len_k, depth_k)
			v: value, shape == (..., seq_len_v, depth_v)
			mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k) or None.

			Since we use scaled-product attention, we assume seq_len_k = seq_len_v

		Returns:
			  attention outputs of shape (batch_size, seq_len_q, d_model)
	"""

	def __init__(self, n_heads, d_model, **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.d_model = d_model
		self.head_depth = self.d_model // self.n_heads

		if self.d_model % self.n_heads != 0:
			raise ValueError("number of heads must divide d_model")

		# define weight matrices
		self.wq = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_q, d_model)
		self.wk = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_k, d_model)
		self.wv = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_v, d_model)

		self.w_out = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_model, d_model)

	def split_heads(self, tensor, batch_size):
		"""Function for computing attention on several heads simultaneously
		Splits last dimension of a tensor into (num_heads, head_depth).
		Then we transpose it as (batch_size, num_heads, ..., head_depth) so that we can use broadcast
		"""
		tensor = tf.reshape(tensor, (batch_size, -1, self.n_heads, self.head_depth))
		return tf.transpose(tensor, perm=[0, 2, 1, 3])

	# treats first parameter q as input, and  k, v as parameters, so input_shape=q.shape
	def call(self, q, k, v, mask=None):
		# shape of q: (batch_size, seq_len_q, d_q)
		batch_size = tf.shape(q)[0]

		# compute Q = q * w_q, ...
		Q = self.wq(q)  # (batch_size, seq_len_q, d_q) x (d_q, d_model) --> (batch_size, seq_len_q, d_model)
		K = self.wk(k)  # ... --> (batch_size, seq_len_k, d_model)
		V = self.wv(v)  # ... --> (batch_size, seq_len_v, d_model)

		# split heads: d_model = num_heads * head_depth + reshape
		Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, head_depth)
		K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_k, head_depth)
		V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_v, head_depth)

		# similarity between context vector Q and key K // self-similarity in case of self-attention
		compatibility = tf.matmul(Q, K, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
														   # seq_len_q = n_nodes for encoder self-attention
														   # seq_len_q = 1 for decoder context-vector attention
														   # seq_len_k = n_nodes for both encoder & decoder
		# rescaling
		dk = tf.cast(tf.shape(K)[-1], tf.float32)
		compatibility = compatibility / tf.math.sqrt(dk)

		if mask is not None:
			# we need to reshape mask:
			# (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
			# so that we will be able to do a broadcast:
			# (batch_size, num_heads, seq_len_q, seq_len_k) + (batch_size, 1, seq_len_q, seq_len_k)
			mask = mask[:, tf.newaxis, :, :]

			# we use tf.where since 0*-np.inf returns nan, but not -np.inf
			compatibility = tf.where(mask,
									 tf.ones_like(compatibility) * (-np.inf),
									 compatibility
									 )

		compatibility = tf.nn.softmax(compatibility, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

		# seq_len_k = seq_len_v
		attention = tf.matmul(compatibility, V)  # (batch_size, num_heads, seq_len_q, head_depth)

		# transpose back to (batch_size, seq_len_q, num_heads, head_depth)
		attention = tf.transpose(attention, perm=[0, 2, 1, 3])

		# concatenate heads (last 2 dimensions)
		attention = tf.reshape(attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		# project output to the same dimension
		# this is equiv. to sum in the article (project heads with W_o and sum), beacuse of block-matrix multiplication
		#e.g. https://math.stackexchange.com/questions/2961550/matrix-block-multiplication-definition-properties-and-applications

		output = self.w_out(attention)  # (batch_size, seq_len_q, d_model)

		return output


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
	"""Feed-Forward Sublayer: fully-connected Feed-Forward network,
	built based on MHA vectors from MultiHeadAttention layer with skip-connections

		Args:
			num_heads: number of attention heads in MHA layers.
			input_dim: embedding size that will be used as d_model in MHA layers.
			feed_forward_hidden: number of neuron units in each FF layer.

		Call arguments:
			x: batch of shape (batch_size, n_nodes, node_embedding_size).
			mask: mask for MHA layer

		Returns:
			   outputs of shape (batch_size, n_nodes, input_dim)

	"""

	def __init__(self, input_dim, num_heads, feed_forward_hidden=512, **kwargs):
		super().__init__(**kwargs)
		self.mha = MultiHeadAttention(n_heads=num_heads, d_model=input_dim, name='MHA')
		self.bn1 = tf.keras.layers.BatchNormalization(name='bn1', trainable=True)
		self.bn2 = tf.keras.layers.BatchNormalization(name='bn2', trainable=True)
		self.ff1 = tf.keras.layers.Dense(feed_forward_hidden, name='ff1')
		self.ff2 = tf.keras.layers.Dense(input_dim, name='ff2')

	def call(self, x, mask=None):
		mha_out = self.mha(x, x, x, mask)
		sc1_out = tf.keras.layers.Add()([x, mha_out])
		bn1_out = self.bn1(sc1_out, training=True)

		ff1_out = self.ff1(bn1_out)
		relu1_out = tf.keras.activations.relu(ff1_out)
		ff2_out = self.ff2(relu1_out)
		sc2_out = tf.keras.layers.Add()([bn1_out, ff2_out])
		bn2_out = self.bn2(sc2_out, training=True)

		return bn2_out

class GraphAttentionEncoder(tf.keras.models.Model):
	"""Graph Encoder, which uses MultiHeadAttentionLayer sublayer.

		Args:
			input_dim: embedding size that will be used as d_model in MHA layers.
			num_heads: number of attention heads in MHA layers.
			num_layers: number of attention layers that will be used in encoder.
			feed_forward_hidden: number of neuron units in each FF layer.

		Call arguments:
			x: tuples of 3 tensors:  (batch_size, 2), (batch_size, n_nodes-1, 2), (batch_size, n_nodes-1)
			First tensor contains coordinates for depot, second one is for coordinates of other nodes,
			Last tensor is for normalized demands for nodes except depot

			mask: mask for MHA layer

		Returns:
			   Embedding for all nodes + mean embedding for graph.
			   Tuples ((batch_size, n_nodes, input_dim), (batch_size, input_dim))
	"""

	def __init__(self, input_dim, num_heads, num_layers, feed_forward_hidden=512):
		super().__init__()

		self.input_dim = input_dim
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.feed_forward_hidden = feed_forward_hidden

		# initial embeddings (batch_size, n_nodes-1, 2) --> (batch-size, input_dim), separate for depot and other nodes
		self.init_embed_depot = tf.keras.layers.Dense(self.input_dim, name='init_embed_depot')  # nn.Linear(2, embedding_dim)
		self.init_embed = tf.keras.layers.Dense(self.input_dim, name='init_embed')

		self.mha_layers = [MultiHeadAttentionLayer(self.input_dim, self.num_heads, self.feed_forward_hidden)
							for _ in range(self.num_layers)]

	def call(self, x, mask=None):

		assert mask is None, "TODO mask not yet supported!"

		x = tf.concat((self.init_embed_depot(x[0])[:, None, :],  # (batch_size, 2) --> (batch_size, 1, 2)
					   self.init_embed(tf.concat((x[1], x[2][:, :, None]), axis=-1))  # (batch_size, n_nodes-1, 2) + (batch_size, n_nodes-1)
					   ), axis=1)  # (batch_size, n_nodes, input_dim)

		# stack attention layers
		for i in range(self.num_layers):
			x = self.mha_layers[i](x)

		output = (x, tf.reduce_mean(x, axis=1))
		return output # (embeds of nodes, avg graph embed)=((batch_size, n_nodes, input), (batch_size, input_dim))


if __name__ == '__main__':
	model = GraphAttentionEncoder(input_dim = 128, num_heads = 8, num_layers = 3)
	dataset = generate_data(n_samples=10, n_customer = 20)
	for i, data in enumerate(dataset.batch(10)):
		output = model(data)
		print(output[0].shape)
		print(output[1].shape)
		if i == 0:
			break

	for w in model.trainable_weights:# non_trainable_weights:
		print(w.name)
		print(w.shape)
	model.summary()