import pickle
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import time


def create_data_on_disk(graph_size, num_samples, is_save=True, filename=None, is_return=False, seed=1234):
	"""Generate validation dataset (with SEED) and save
	"""

	CAPACITIES = {
		10: 20.,
		20: 30.,
		50: 40.,
		100: 50.
	}
	depo, graphs, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, 2), seed=seed),
							tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2), seed=seed),
							tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
													  dtype=tf.int32, seed=seed), tf.float32) / tf.cast(CAPACITIES[graph_size], tf.float32)
							)
	if is_save:
		save_to_pickle('Validation_dataset_{}.pkl'.format(filename), (depo, graphs, demand))

	if is_return:
		return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))


def save_to_pickle(filename, item):
	"""Save to pickle
	"""
	with open(filename, 'wb') as handle:
		pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_from_pickle(path, return_tf_data_set=True, num_samples=None):
	"""Read dataset from file (pickle)
	"""

	objects = []
	with (open(path, "rb")) as openfile:
		while True:
			try:
				objects.append(pickle.load(openfile))
			except EOFError:
				break
	objects = objects[0]
	if return_tf_data_set:
		depo, graphs, demand = objects
		if num_samples is not None:
			return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand))).take(num_samples)
		else:
			return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))
	else:
		return objects


def generate_data_onfly(num_samples=10000, graph_size=20):
	"""Generate temp dataset in memory
	"""

	CAPACITIES = {
		10: 20.,
		20: 30.,
		50: 40.,
		100: 50.
	}
	depo, graphs, demand = (tf.random.uniform(minval=0, maxval=1, shape=(num_samples, 2)),
							tf.random.uniform(minval=0, maxval=1, shape=(num_samples, graph_size, 2)),
							tf.cast(tf.random.uniform(minval=1, maxval=10, shape=(num_samples, graph_size),
													  dtype=tf.int32), tf.float32)/tf.cast(CAPACITIES[graph_size], tf.float32)
							)

	return tf.data.Dataset.from_tensor_slices((list(depo), list(graphs), list(demand)))

if __name__ == '__main__':
	dataset = generate_data_onfly()
	it = iter(dataset)
	print(next(it)[0].shape)
	print(next(it)[1].shape)
	print(next(it)[2].shape)
	
