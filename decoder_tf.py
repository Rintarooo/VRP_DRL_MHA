import tensorflow as tf
from layers_tf import MultiHeadAttention, DotProductAttention

class DecoderCell(tf.keras.layers.Layer):
	def __init__(self, n_heads = 8, clip=10., **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.clip = clip

	def build(self, input_shape):
		context_shape, node_shape = input_shape
		self.prep_attention_layer = MultiHeadAttention(n_heads=self.n_heads, embed_dim = node_shape[2])
		self.final_attention_layer = DotProductAttention(return_logits = True, clip = self.clip)
		super().build(input_shape)

	def call(self, inputs, mask=None):
		"""
		Arguments:
			inputs (List[tf.Tensor]): List of tf.Tensor represents context (batch_size, 1, 2*embed_dim+1) and tf.Tensor of nodes (batch_size, n_nodes, embed_dim)

		Returns:
			tf.Tensor with shape (batch_size, 1, n_nodes)
		"""
		context, nodes = inputs
		query = self.prep_attention_layer([context, nodes, nodes], mask=mask)
		logits = self.final_attention_layer([query, nodes, nodes], mask=mask)
		return logits

class Sampler(tf.keras.layers.Layer):
	def __init__(self, n_samples=1, **kwargs):
		super().__init__(**kwargs)
		self.n_samples = n_samples

class TopKSampler(Sampler):# greedy
	def call(self, logits):
		return tf.math.top_k(logits, self.n_samples).indices

class CategoricalSampler(Sampler):# categorial
	def call(self, logits):
		return tf.random.categorical(logits, self.n_samples, dtype=tf.int32)

if __name__ == '__main__':
	decoder = DecoderCell()
	batch, n_nodes, embed_dim = 5, 21, 128
	input1 = tf.ones((batch, 1, 2*embed_dim + 1), dtype = tf.float32)
	input2 = tf.ones((batch, n_nodes, embed_dim), dtype = tf.float32)
	# mask = tf.random.uniform((batch, n_nodes, 1), dtype = tf.float32)
	mask = tf.zeros((batch, n_nodes, 1), dtype = tf.float32)
	logits = decoder([input1, input2], mask)
	print(logits.shape)
	# logits: (batch, 1, n_nodes), logits denotes the value before going into softmax
	sampler = TopKSampler() 
	next_node = sampler(tf.squeeze(logits, axis=1))
	print(next_node.shape)
	# next node: (batch, 1)

