import tensorflow as tf
from layers import MultiHeadAttention, DotProductAttention

# class DecoderCell(tf.keras.layers.Layer):
class DecoderCell(tf.keras.models.Model):
	def __init__(self, n_heads = 8, clip = 10., **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.clip = clip

	def build(self, input_shape):
		context_shape, node_embeddings_shape = input_shape
		self.prep_attention_layer = MultiHeadAttention(n_heads = self.n_heads, embed_dim = node_embeddings_shape[2], spilt_Wq = True)
		self.final_attention_layer = MultiHeadAttention(n_heads = 1, embed_dim = node_embeddings_shape[2], clip = self.clip, return_logits = True)
		super().build(input_shape)

	@tf.function
	def call(self, inputs, mask = None):
		""" context: (batch, 1, 2*embed_dim+1)
			context = tf.concat([], axis = -1)
			tf.concat([graph embedding[:,None,:], previous node embedding, remaining vehicle capacity[:,:,None]], axis = -1)
			graph embedding: (batch, embed_dim)
			previous node embedding: (batch, n_nodes, embed_dim)
			remaining vehicle capacity(= D): (batch, 1) 
			node_embeddings: (batch, n_nodes, embed_dim)
		"""
		context, node_embeddings = inputs
		query = self.prep_attention_layer([context, node_embeddings, node_embeddings], mask = mask)
		logits = self.final_attention_layer([query, node_embeddings, None], mask = mask)
		return logits

class Sampler(tf.keras.layers.Layer):
	def __init__(self, n_samples = 1, **kwargs):
		super().__init__(**kwargs)
		self.n_samples = n_samples
		""" logits: (batch, n_classes)
			TopKSampler <-- greedy; sample ones with biggest probability
			CategoricalSampler <-- sampling; randomly sample ones from possible distribution based on probability
		"""

class TopKSampler(Sampler):
	def call(self, logits):
		return tf.math.top_k(logits, self.n_samples).indices

class CategoricalSampler(Sampler):
	def call(self, logits):
		return tf.random.categorical(logits, self.n_samples, dtype = tf.int32)

if __name__ == '__main__':
	batch, n_nodes, embed_dim = 5, 21, 128
	context = tf.ones((batch, 1, 2*embed_dim + 1), dtype = tf.float32)
	nodes = tf.ones((batch, n_nodes, embed_dim), dtype = tf.float32)
	mask = tf.zeros((batch, n_nodes, 1), dtype = tf.bool)
	decoder = DecoderCell()
	logits = decoder([context, nodes], mask)
	print(logits.shape)# logits: (batch, 1, n_nodes), logits denotes the value before going into softmax
	sampler = CategoricalSampler() 
	next_node = sampler(tf.squeeze(logits, axis = 1))
	print(next_node.shape)# next node: (batch, 1)
	decoder.summary()

	for w in decoder.trainable_weights:# non_trainable_weights:
		print(w.name, w.shape)
