import tensorflow as tf

from data import generate_data
from encoder import GraphAttentionEncoder
from decoder import DecoderCell

class AttentionModel(tf.keras.models.Model):
	
	def __init__(self, embed_dim = 128, n_encode_layers = 3, n_heads = 8, tanh_clipping = 10., FF_hidden = 512):
		super().__init__()
		
		self.Encoder = GraphAttentionEncoder(embed_dim, n_heads, n_encode_layers, FF_hidden)
		self.Decoder = DecoderCell(embed_dim, n_heads, tanh_clipping)

	def call(self, x, training = True, return_pi = False, decode_type = 'greedy'):
		encoder_output = self.Encoder(x, training = training)
		decoder_output = self.Decoder(x, encoder_output, return_pi = return_pi, decode_type = decode_type)
		if return_pi:
			cost, ll, pi = decoder_output
			return cost, ll, pi
		cost, ll = decoder_output
		return cost, ll
		
if __name__ == '__main__':
	# tf.config.experimental_run_functions_eagerly(True)
	model = AttentionModel()
	dataset = generate_data(seed = 123)
	return_pi = False
	for i, data in enumerate(dataset.batch(6)):
		output = model(data, decode_type = 'sampling', return_pi = return_pi)
		if return_pi:
			print(output[0])# cost: (batch,)
			print(output[1])# ll: (batch,)
			print(output[2])# pi: (batch, decode_step) # tour
		else:
			print(output[0])# cost: (batch,)
			print(output[1])# ll: (batch,)
		if i == 0:
			break

	# print('model.trainable_weights')
	# for w in model.trainable_weights:
	# 	print(w.name)
	# 	print(w.numpy())

	model.summary()

