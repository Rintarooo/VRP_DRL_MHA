import tensorflow as tf
from Environment import VRPproblem
import numpy as np
from data import generate_data

class GraphAttentionDecoder(tf.keras.models.Model):

	def __init__(self,
				 output_dim,
				 num_heads,
				 tanh_clipping=10,
				 decode_type=None):

		super().__init__()

		self.output_dim = output_dim
		self.num_heads = num_heads

		self.head_depth = self.output_dim // self.num_heads
		self.dk_mha_decoder = tf.cast(self.head_depth, tf.float32) #for decoding in mha_decoder
		self.dk_get_loc_p = tf.cast(self.output_dim, tf.float32)  # for decoding in mha_decoder

		if self.output_dim % self.num_heads != 0:
			raise ValueError("number of heads must divide d_model=output_dim")

		self.tanh_clipping = tanh_clipping
		self.decode_type = decode_type

		# we split projection matrix Wq into 2 matrices: Wq*[h_c, h_N, D] = Wq_context*h_c + Wq_step_context[h_N, D]
		self.wq_context = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wq_context')  # (d_q_context, output_dim)
		self.wq_step_context = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wq_step_context')  # (d_q_step_context, output_dim)

		# we need two Wk projections since there is MHA followed by 1-head attention - they have different keys K
		self.wk = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wk')  # (d_k, output_dim)
		self.wk_tanh = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wk_tanh')  # (d_k_tanh, output_dim)

		# we dont need Wv projection for 1-head attention: only need attention weights as outputs
		self.wv = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='wv')  # (d_v, output_dim)

		# we dont need wq for 1-head tanh attention, since we can absorb it into w_out
		self.w_out = tf.keras.layers.Dense(self.output_dim, use_bias=False, name='w_out')  # (d_model, d_model)

		self.problem = VRPproblem

	def set_decode_type(self, decode_type):
		self.decode_type = decode_type

	def split_heads(self, tensor, batch_size):
		"""Function for computing attention on several heads simultaneously
		Splits last dimension of a tensor into (num_heads, head_depth).
		Then we transpose it as (batch_size, num_heads, ..., head_depth) so that we can use broadcast
		"""
		tensor = tf.reshape(tensor, (batch_size, -1, self.num_heads, self.head_depth))
		return tf.transpose(tensor, perm=[0, 2, 1, 3])

	def _select_node(self, logits):
		"""Select next node based on decoding type.
		"""

		#assert tf.reduce_all(logits == logits), "Probs should not contain any nans"

		if self.decode_type == "greedy":
			#probs = tf.exp(logits)
			#selected = tf.math.argmax(probs, axis=-1) # (batch_size, 1)
			selected = tf.math.argmax(logits, axis=-1) # (batch_size, 1)

		elif self.decode_type == "sampling":
			# logits has a shape of (batch_size, 1, n_nodes), we have to squeeze it
			# to (batch_size, n_nodes) since tf.random.categorical requires matrix
			selected = tf.random.categorical(logits[:, 0, :], 1) #(bach_size,1)
		else:
			assert False, "Unknown decode type"

		return tf.squeeze(selected, axis=-1) # (bach_size,)

	def get_step_context(self, state, embeddings):
		"""Takes a state and graph embeddings,
		   Returns a part [h_N, D] of context vector [h_c, h_N, D],
		   that is related to RL Agent last step.
		"""
		# index of previous node
		prev_node = state.prev_a  # (batch_size, 1)

		# from embeddings=(batch_size, n_nodes, input_dim) select embeddings of previous nodes
		cur_embedded_node = tf.gather(embeddings, tf.cast(prev_node, tf.int32), batch_dims=1)  # (batch_size, 1, input_dim)

		# add remaining capacity
		step_context = tf.concat([cur_embedded_node, self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]], axis=-1)

		return step_context  # (batch_size, 1, input_dim + 1)

	def decoder_mha(self, Q, K, V, mask=None):
		""" Computes Multi-Head Attention part of decoder
		Basically, its a part of MHA sublayer, but we cant construct a layer since Q changes in a decoding loop.

		Args:
			mask: a mask for visited nodes,
				has shape (batch_size, seq_len_q, seq_len_k), seq_len_q = 1 for context vector attention in decoder
			Q: query (context vector for decoder)
					has shape (..., seq_len_q, head_depth) with seq_len_q = 1 for context_vector attention in decoder
			K, V: key, value (projections of nodes embeddings)
				have shape (..., seq_len_k, head_depth), (..., seq_len_v, head_depth),
																with seq_len_k = seq_len_v = n_nodes for decoder
		"""

		#batch_size = tf.shape(Q)[0]

		compatibility = tf.matmul(Q, K, transpose_b=True)/tf.math.sqrt(self.dk_mha_decoder)  # (batch_size, num_heads, seq_len_q, seq_len_k)

		#dk = tf.cast(tf.shape(K)[-1], tf.float32)
		#compatibility = compatibility / tf.math.sqrt(dk)
		#compatibility = compatibility / tf.math.sqrt(self.dk_mha_decoder)

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
		attention = tf.matmul(compatibility, V)  # (batch_size, num_heads, seq_len_q, head_depth)

		# transpose back to (batch_size, seq_len_q, num_heads, depth)
		attention = tf.transpose(attention, perm=[0, 2, 1, 3])

		# concatenate heads (last 2 dimensions)
		attention = tf.reshape(attention, (self.batch_size, -1, self.output_dim))  # (batch_size, seq_len_q, output_dim)

		output = self.w_out(attention)  # (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context att in decoder

		return output

	def get_log_p(self, Q, K, mask=None):
		"""Single-Head attention sublayer in decoder,
		computes log-probabilities for node selection.

		Args:
			mask: mask for nodes
			Q: query (output of mha layer)
					has shape (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context attention in decoder
			K: key (projection of node embeddings)
					has shape  (batch_size, seq_len_k, output_dim), seq_len_k = n_nodes for decoder
		"""

		compatibility = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(self.dk_get_loc_p)

		#dk = tf.cast(tf.shape(K)[-1], tf.float32)
		#compatibility = compatibility / tf.math.sqrt(dk)
		#compatibility = compatibility / tf.math.sqrt(self.dk_get_loc_p)

		compatibility = tf.math.tanh(compatibility) * self.tanh_clipping

		if mask is not None:

			# we dont need to reshape mask like we did in multi-head version:
			# (batch_size, seq_len_q, seq_len_k) --> (batch_size, num_heads, seq_len_q, seq_len_k)
			# since we dont have multiple heads

			compatibility = tf.where(mask,
									 tf.ones_like(compatibility) * (-np.inf),
									 compatibility
									 )

		log_p = tf.nn.log_softmax(compatibility, axis=-1)  # (batch_size, seq_len_q, seq_len_k)

		return log_p


	def _calc_log_likelihood(self, _log_p, a):

		# Get log_p corresponding to selected actions
		log_p = tf.gather_nd(_log_p, tf.cast(tf.expand_dims(a, axis=-1), tf.int32), batch_dims=2)

		# Calculate log_likelihood
		return tf.reduce_sum(log_p,1)


	def call(self, inputs, embeddings, context_vectors, decode_type = 'sampling'):
		# embeddings shape = (batch_size, n_nodes, input_dim)
		# context vectors shape = (batch_size, input_dim)
		self.embeddings = embeddings
		self.batch_size = tf.shape(self.embeddings)[0]
		self.decode_type = decode_type

		outputs = []
		sequences = []

		state = self.problem(inputs)

		# we compute some projections (common for each policy step) before decoding loop for efficiency
		K = self.wk(self.embeddings)  # (batch_size, n_nodes, output_dim)
		K_tanh = self.wk_tanh(self.embeddings)  # (batch_size, n_nodes, output_dim)
		V = self.wv(self.embeddings)  # (batch_size, n_nodes, output_dim)
		Q_context = self.wq_context(context_vectors[:, tf.newaxis, :])  # (batch_size, 1, output_dim)


		# we dont need to split K_tanh since there is only 1 head; Q will be split in decoding loop
		K = self.split_heads(K, self.batch_size)  # (batch_size, num_heads, n_nodes, head_depth)
		V = self.split_heads(V, self.batch_size)  # (batch_size, num_heads, n_nodes, head_depth)

		# Perform decoding steps
		i = 0

		while not state.all_finished():
			step_context = self.get_step_context(state, self.embeddings)  # (batch_size, 1, input_dim + 1)
			Q_step_context = self.wq_step_context(step_context)  # (batch_size, 1, output_dim)
			Q = Q_context + Q_step_context

			# split heads for Q
			Q = self.split_heads(Q, self.batch_size)  # (batch_size, num_heads, 1, head_depth)

			# get current mask
			mask = state.get_mask()  # (batch_size, 1, n_nodes) with True/False indicating where agent can go

			# compute MHA decoder vectors for current mask
			mha = self.decoder_mha(Q, K, V, mask)  # (batch_size, 1, output_dim)

			# compute probabilities
			log_p = self.get_log_p(mha, K_tanh, mask)  # (batch_size, 1, n_nodes)

			# next step is to select node
			selected = self._select_node(log_p)

			state.step(selected)

			outputs.append(log_p[:, 0, :])
			sequences.append(selected)

			i += 1

		# Collected lists, return Tensor
		log_ps, pi = tf.stack(outputs, 1), tf.cast(tf.stack(sequences, 1), tf.float32)
		cost = self.problem.get_costs(inputs, pi)
		ll = self._calc_log_likelihood(log_ps, pi)
		return cost, ll

if __name__ == '__main__':
	model = GraphAttentionDecoder(output_dim = 128, num_heads = 8)
	model.decode_type = 'sampling'
	batch = 10
	n_nodes = 21
	embed_dim = 128
	embeddings, context_vectors = tf.ones((batch, n_nodes, embed_dim)), tf.ones((batch, embed_dim))
	data = generate_data(10, 20)
	for i, data in enumerate(data.batch(10)):
		output = model(data, embeddings, context_vectors)
		print(output[0])
		print(output[1])
		if i == 0:
			break

	for w in model.trainable_weights:# non_trainable_weights:
		print(w.name)
		print(w.shape)
	model.summary()



