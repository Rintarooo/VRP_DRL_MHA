import tensorflow as tf
from layers import MultiHeadAttention
from data import generate_data

class ResidualBlock_BN(tf.keras.layers.Layer):
	def __init__(self, MHA, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA
		self.BN = tf.keras.layers.BatchNormalization()

	def call(self, x, mask = None):
		if mask is None:
			return self.BN(x + self.MHA(x))# training = True
		else:
			return self.BN(x + self.MHA(x, mask))

class SelfAttention(tf.keras.layers.Layer):
	def __init__(self, MHA, **kwargs):
		super().__init__(**kwargs)
		self.MHA = MHA

	def call(self, x, mask=None):
		return self.MHA([x, x, x], mask=mask)

class AttentionLayer(tf.keras.layers.Layer):
	def __init__(self, n_heads = 8, FF_hidden = 512, activation = 'relu', **kwargs):
		super().__init__(**kwargs)
		self.n_heads = n_heads
		self.FF_hidden = FF_hidden
		self.activation = activation
		
	def build(self, input_shape):
		self.MHA_sublayer = ResidualBlock_BN(
			SelfAttention(
					MultiHeadAttention(n_heads = self.n_heads, embed_dim = input_shape[2])# input_shape[2] = embed_dim = 128	
			)
		)
		self.FF_sublayer = ResidualBlock_BN(
			tf.keras.models.Sequential([
					tf.keras.layers.Dense(self.FF_hidden, activation = self.activation),
					tf.keras.layers.Dense(input_shape[2])
			])
		)
		super().build(input_shape)
	
	"""	def call
		args: (batch_size, n_nodes, embed_dim).
		return: (batch_size, n_nodes, embed_dim)
	"""
	def call(self, x, mask=None):
		return self.FF_sublayer(self.MHA_sublayer(x, mask=mask))

class GraphAttentionEncoder(tf.keras.models.Model):
	def __init__(self, embed_dim = 128, n_heads = 8, n_layers = 3, FF_hidden=512):
		super().__init__()
		# initial embeddings (batch_size, n_nodes-1, 2) --> (batch-size, embed_dim), separate for depot and customer nodes
		self.init_embed_depot = tf.keras.layers.Dense(embed_dim)# torch.nn.Linear(2, embedding_dim)
		self.init_embed = tf.keras.layers.Dense(embed_dim)
		self.attention_layers = [AttentionLayer(n_heads, FF_hidden)
							for _ in range(n_layers)]
	# @tf.function	
	def call(self, x, mask=None):
		x = tf.concat((self.init_embed_depot(x[0])[:, None, :],  # depot xy(batch_size, 2) --> (batch_size, 1, 2)
					   self.init_embed(tf.concat((x[1], x[2][:, :, None]), axis=-1))  # customer xy(batch_size, n_nodes-1, 2) + customer demand(batch_size, n_nodes-1)
					   ), axis = 1)  # embed_x(batch_size, n_nodes, embed_dim)

		for layer in self.attention_layers:
			x = layer(x, mask)# stack attention layers

		return (x, tf.reduce_mean(x, axis = 1))
		"""	(node embeddings(= embedding for all nodes), graph embedding(= mean of node embeddings for graph))
				=((batch_size, n_nodes, embed_dim), (batch_size, embed_dim))
		"""


if __name__ == '__main__':
	encoder = GraphAttentionEncoder()
	dataset = generate_data()
	for i, data in enumerate(dataset.batch(5)):
		output = encoder(data)
		print(output[0].shape)
		print(output[1].shape)
		if i == 0:
			break
	encoder.summary()
	for w in encoder.non_trainable_weights:
		print(w.name)
	