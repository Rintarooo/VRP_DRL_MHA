import tensorflow as tf
from data import generate_data

class AgentVRP():

	def __init__(self, dataset):
		""" depot_xy: (batch, 2)
			customer_xy: (batch, n_nodes-1, 2)
				--> self.xy: (batch, n_nodes, 2)
					Coordinates of depot + customer nodes
			demand: (batch, n_nodes-1)
			
			is_depot: (batch, 1), e.g. [[True], [True], ...]
			Nodes that have been visited will be marked with True.
		"""
		self.depot_xy, customer_xy, self.demand = dataset
		self.xy = tf.concat([self.depot_xy[:, None, :], customer_xy], 1)
		self.batch, self.n_nodes, _ = tf.shape(self.xy)
		self.visited_customer = tf.zeros((self.batch, self.n_nodes-1, 1), dtype = tf.bool)
		self.is_depot = tf.ones((self.batch, 1), dtype = tf.bool)
		self.reset()

	def reset(self):
		self.t = tf.ones((1), dtype=tf.int64)
		self.D = tf.ones((self.batch, 1), dtype=tf.float32)
	
	def all_visited(self):
		""" tf.reduce_all: if there's any False on the specified axis, return False
		"""
		return tf.reduce_all(self.visited_customer)

	def partial_visited(self):
		"""Checks if partial solution for all graphs has been built, i.e. all agents in batch came back to depot
		"""
		return tf.reduce_all(self.is_depot) and self.t != 1

	def get_mask_D(self, next_node, visited_mask):
		""" next_node: ([[0],[0],[not 0], ...], (batch, 1), dtype = tf.int32), [0] denotes going to depot
			visited_mask **includes depot**: (batch, n_nodes, 1)
			visited_mask[:,1:,:] **excludes depot**: (batch, n_nodes-1, 1)
			customer_idx **excludes depot**: (batch, 1), range[0, n_nodes-1] e.g. [[3],[0],[5],[11], ...]
			self.demand **excludes depot**: (batch, n_nodes-1)
			selected_demand: (batch, 1)
			if next node is depot, do not select demand
			D: (batch, 1), D denotes "remaining vehicle capacity"
			self.capacity_over_customer **excludes depot**: (batch, n_nodes-1)
			visited_customer **excludes depot**: (batch, n_nodes-1, 1)
		 	is_depot: (batch, 1), e.g. [[True], [True], ...]
		 	return mask: (batch, n_nodes, 1)		
		"""
		self.is_depot = next_node == 0
		self.visited_customer = self.visited_customer | visited_mask[:,1:,:]
		customer_idx = tf.argmax(tf.cast(visited_mask[:,1:,:], tf.int32), axis = 1)
		selected_demand = tf.gather(params = self.demand, indices = customer_idx, batch_dims = 1)
		self.D -= selected_demand * (1.0 - tf.cast(self.is_depot, tf.float32))
		capacity_over_customer = self.demand > self.D
		mask_customer = capacity_over_customer[:, :, None] | self.visited_customer
		mask_depot = tf.math.logical_not(tf.reduce_all(mask_customer, axis = 1))
		""" We can choose depot if 1) we are not in depot or 2) all nodes are visited
			tf.reduce_all: if there's any False on the specified axis, return False
			if all customer nodes are True, mask_depot should be False so that the vehicle returns back to depot 
		"""
		self.t += 1
		return tf.concat([mask_depot[:, None, :], mask_customer], axis = 1), self.D

	def get_costs(self, pi):
		""" self.xy: (batch, n_nodes, 2) # Coordinates of depot + customer nodes
			pi: (batch, decode_step) # predicted path
			d: (batch, decode_step, 2)
		"""
		d = tf.gather(self.xy, indices = pi, batch_dims = 1)
		# Note: first element of pi is not depot, but the first selected node in the path
		return (tf.reduce_sum(tf.norm(d[:, 1:] - d[:, :-1], ord=2, axis=2), axis=1)
				+ tf.norm(d[:, 0] - self.depot_xy, ord=2, axis=1)# distance from depot to first selected node
				+ tf.norm(d[:, -1] - self.depot_xy, ord=2, axis=1))# distance from last selected node (!=0 for graph with longest path) to depot

if __name__ == '__main__':
	dataset = generate_data(n_samples = 10)
	for i, data in enumerate(dataset.batch(1)):
		print(data[0])
		print(data[1])
		env = AgentVRP(data)
		pi = [[1,3,2,4]]
		print(env.get_costs(pi))
		if i == 0:
			break

