# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def generate_data(n_samples = 10000, n_customer = 20):
	CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
	depot, graphs, demand = (tf.random.uniform((n_samples, 2), minval = 0, maxval = 1),
							tf.random.uniform((n_samples, n_customer, 2), minval = 0, maxval = 1),
							tf.cast(tf.random.uniform((n_samples, n_customer), minval = 1, maxval = 10, 
								dtype = tf.int32), tf.float32) / tf.cast(CAPACITIES[n_customer], tf.float32)
							)
	return tf.data.Dataset.from_tensor_slices((list(depot), list(graphs), list(demand)))

if __name__ == '__main__':
	dataset = generate_data()
	# data = next(iter(dataset))
	# print(data[0].shape)
	# print(data[1].shape)
	# print(data[2].shape)

	for i, data in enumerate(dataset.batch(5)):
		print(data[0].shape)
		print(data[1].shape)
		print(data[2].shape)
		if i == 0:
			break
	
