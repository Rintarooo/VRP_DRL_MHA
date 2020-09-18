import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

# def generate_data_VerySlow(n_samples = 10000, n_customer = 20):
# 	CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
# 	depot, graphs, demand = (tf.random.uniform((n_samples, 2), minval = 0, maxval = 1),
# 							tf.random.uniform((n_samples, n_customer, 2), minval = 0, maxval = 1),
# 							tf.cast(tf.random.uniform((n_samples, n_customer), minval = 1, maxval = 10, 
# 								dtype = tf.int32), tf.float32) / tf.cast(CAPACITIES[n_customer], tf.float32))
# 	return tf.data.Dataset.from_tensor_slices((list(depot), list(graphs), list(demand)))

def generate_data(n_samples = 1000, n_customer = 20, seed = None):
	g = tf.random.experimental.Generator.from_non_deterministic_state()
	if seed is not None:
		g = tf.random.experimental.Generator.from_seed(seed)

	CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
	
	@tf.function	
	def tf_rand():
		return [g.uniform(shape=[n_samples, 2], minval = 0, maxval = 1),
				g.uniform(shape = [n_samples, n_customer, 2], minval = 0, maxval = 1),
				tf.cast(g.uniform(shape = [n_samples, n_customer], minval = 1, maxval = 10, 
					dtype = tf.int32), tf.float32) / tf.cast(CAPACITIES[n_customer], tf.float32)]
	# print(np.array(g.uniform(shape=[n_samples, 2], minval = 0, maxval = 1)).shape)
	# print(np.array(g.uniform(shape = [n_samples, n_customer, 2], minval = 0, maxval = 1)).shape)
	# print(np.array(tf.cast(g.uniform(shape = [n_samples, n_customer], minval = 1, maxval = 10, 
	# 				dtype = tf.int32), tf.float32) / tf.cast(CAPACITIES[n_customer], tf.float32)).shape)
	return tf.data.Dataset.from_tensor_slices(tuple(tf_rand()))

def data_from_txt(path):
	if not os.path.isfile(path):
		raise FileNotFoundError	
	with open(path, 'r') as f:
		lines = list(map(lambda s: s.strip(), f.readlines()))
		customer_xy, demand = [], []
		ZERO, DEPOT, CUSTO, DEMAND = [False for i in range(4)]
		ZERO = True
		for line in lines:
			if(ZERO):
				if(line == 'NODE_COORD_SECTION'):
					ZERO = False
					DEPOT = True

			elif(DEPOT):
				depot_xy = list(map(lambda k: float(k)/100., line.split()))[1:]# depot_xy.append(list(map(int, line.split()))[1:])
				DEPOT = False
				CUSTO = True
				
			elif(CUSTO):
				if(line == 'DEMAND_SECTION'):
					DEMAND = True
					CUSTO = False
					continue
				customer_xy.append(list(map(lambda k: float(k)/100., line.split()))[1:])
			elif(DEMAND):
				if(line == '1 0'):
					continue
				elif(line == 'DEPOT_SECTION'):
					break
				else:
					demand.append(list(map(lambda k: float(k)/100., line.split()))[1])# demand.append(list(map(int, line.split()))[1])
	
	# print(np.array(depot_xy).shape)
	# print(np.array(customer_xy).shape)
	# print(np.array(demand).shape)
	tf_rand = [np.expand_dims(np.array(depot_xy), axis = 0), 
				np.expand_dims(np.array(customer_xy), axis = 0), 
				np.expand_dims(np.array(demand), axis = 0)]
	return tf.data.Dataset.from_tensor_slices(tuple(tf_rand))


# def data_from_txt(path):
# 	if not os.path.isfile(path):
# 		raise FileNotFoundError	
# 	with open(path, 'r') as f:
# 		lines = list(map(lambda s: s.strip(), f.readlines()))
# 		customer_xy, demand = [], []
# 		for i, line in enumerate(lines, 1):
# 			if(i == 8):
# 				depot_xy = list(map(lambda k: float(k)/100., line.split()))[1:]# depot_xy.append(list(map(int, line.split()))[1:])
# 			elif(9 <= i & i <= 52):
# 				customer_xy.append(list(map(lambda k: float(k)/100., line.split()))[1:])
# 			elif(55 <= i & i <= 98):
# 				demand.append(list(map(lambda k: float(k)/100., line.split()))[1])# demand.append(list(map(int, line.split()))[1])
	
	# print(np.array(depot_xy).shape)
	# print(np.array(customer_xy).shape)
	# print(np.array(demand).shape)
	
	# tf_rand = [np.expand_dims(np.array(depot_xy), axis = 0), 
	# 			np.expand_dims(np.array(customer_xy), axis = 0), 
	# 			np.expand_dims(np.array(demand), axis = 0)]
	# return tf.data.Dataset.from_tensor_slices(tuple(tf_rand))


if __name__ == '__main__':
	dataset = generate_data(n_samples = 1280, n_customer = 100, seed = 123)
	
	# path = '../OpenData/A-n53-k7.txt'
	# dataset = data_from_txt(path)
	# data = next(iter(dataset))
	
	for i, data in enumerate(dataset.batch(1)):
		print(data[0])
		print(data[1])
		print(data[2])
		if i == 0:
			break