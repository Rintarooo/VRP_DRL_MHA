import tensorflow as tf

class AgentVRP():

	VEHICLE_CAPACITY = 1.0

	def __init__(self, dataset):

		self.depot_xy, customer_xy, self.demand = dataset
		""" depot_xy: (batch, 2)
			customer_xy: (batch, n_nodes-1, 2)
			demand: (batch, n_nodes-1)

			self.xy: (batch, n_nodes, 2) # Coordinates of depot + customer nodes
		"""
		self.xy = tf.concat((self.depot_xy[:, None, :], customer_xy), 1)
		# self.demand = tf.cast(demand, tf.float32)
		
		self.batch = tf.shape(self.depot_xy)[0]
		self.n_nodes = tf.shape(self.xy)[1]

		
		# Nodes that have been visited will be marked with 1
		self.visited_nodes = tf.zeros((self.batch, self.n_nodes-1, 1), dtype=tf.uint8)
		self.is_depot = tf.zeros((self.batch,1), dtype = tf.bool)
		# is_depot: ([[ True], [ True], [False], ...], (batch, 1), dtype = tf.bool)
		self.used_capacity = tf.zeros((self.batch, 1), dtype=tf.float32)

		# Step counter
		self.t = tf.ones((1), dtype=tf.int64)

	
	def all_visited(self):
		"""Checks if all games are visited
			return tf.Tensor(False, shape=(), dtype=bool)
			0 -> False, not 0 -> True
			if any 0 exists, return False
		"""
		return tf.reduce_all(tf.cast(self.visited_nodes, tf.bool))


	def partial_visited(self):
		"""Checks if partial solution for all graphs has been built, i.e. all agents in batch came back to depot
		"""
		return tf.reduce_all(self.is_depot) and self.t != 1

	def reset_t1(self):
		self.t = tf.ones((1), dtype=tf.int64)
		self.used_capacity = tf.zeros((self.batch, 1), dtype=tf.float32)

	def think_capacity(self, next_node, visited_mask):
		# next_node: ([[0],[0],[not 0], ...], (batch, 1), dtype = tf.int32)
		self.is_depot = next_node == 0

		self.visited_nodes += visited_mask[:,1:,:]# exclude depot, mask: (batch, n_nodes-1, 1)
		
		# indice prev_a includes depot but demand includes only cutomer so slide by 1 node backward
		selected_demand = tf.gather(params = self.demand, indices = tf.clip_by_value(next_node - 1, clip_value_min = 0, clip_value_max = tf.shape(self.demand)[-1] - 1), batch_dims = -1)
		# tf.matmul
		# selected_demand: (batch, 1)
		self.used_capacity += selected_demand * (1.0 - tf.cast(self.is_depot, tf.float32))
		# self.used_capacity: (batch, 1)
		D = self.VEHICLE_CAPACITY - self.used_capacity
		# D: (batch, 1)
		# D[:,:,None]: (batch, 1, 1)
		# D denotes "remaining vehicle capacity"
		capacity_over_nodes = self.demand > D
		# self.demand: (batch, n_nodes-1)
		# self.used_capacity = tf.zeros((batch, 1), dtype=tf.float32)
		# is_depot: ([[ True], [ True], [False]], (batch, 1), dtype = tf.bool)
		mask_capacity = capacity_over_nodes[:, :, None] | ((self.t > 0) & self.is_depot[:, None, :])
		mask = tf.cast(mask_capacity, tf.uint8) + visited_mask[:,1:,:]
		
		self.t += 1

		# We can choose depot if 1) we are not in depot OR 2) all nodes are visited
		# mask_depot = self.is_depot & (tf.reduce_sum(mask, axis = 1) > 0)
		mask_depot = (tf.reduce_sum(mask, axis = 1) == self.nodes - 1)
		# mask_depot: (batch, 1)
		return tf.concat([tf.cast(mask_depot[:, None, :], dtype = tf.uint8), mask], axis = 1), D

	# @staticmethod
	def get_costs(self, pi):
		d = tf.gather(self.xy, tf.cast(pi, tf.int32), batch_dims = 1)
		# Calculation of total distance
		# Note: first element of pi is not depot, but the first selected node in the path
		return (tf.reduce_sum(tf.norm(d[:, 1:] - d[:, :-1], ord=2, axis=2), axis=1)
				+ tf.norm(d[:, 0] - self.depot_xy, ord=2, axis=1) # Distance from depot to first selected node
				+ tf.norm(d[:, -1] - self.depot_xy, ord=2, axis=1))  # Distance from last selected node (!=0 for graph with longest path) to depot
