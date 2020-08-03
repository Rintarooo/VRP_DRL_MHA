import tensorflow as tf

class Env():
	def __init__(self, x, node_embeddings):
		super().__init__()
		""" depot_xy: (batch, 2)
			customer_xy: (batch, n_nodes-1, 2)
			--> self.xy: (batch, n_nodes, 2)
				Coordinates of depot + customer nodes
			demand: (batch, n_nodes-1)
			
			is_next_depot: (batch, 1), e.g. [[True], [True], ...]
			Nodes that have been visited will be marked with True.

			self.batch, self.n_nodes, _ = tf.shape(self.xy)
			~.shape --> return static shape, tf.shape() --> return dynamic shape
			https://pgaleone.eu/tensorflow/2018/07/28/understanding-tensorflow-tensors-shape-static-dynamic/
		"""
		self.depot_xy, customer_xy, self.demand = x
		self.xy = tf.concat([self.depot_xy[:, None, :], customer_xy], 1)
		self.batch, self.n_nodes, _ = self.xy.shape
		self.node_embeddings = node_embeddings

		self.is_next_depot = tf.ones([self.batch, 1], dtype = tf.bool)
		self.visited_customer = tf.zeros((self.batch, self.n_nodes-1, 1), dtype = tf.bool)

	def get_mask_D(self, next_node, visited_mask, D):
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
		 	is_next_depot: (batch, 1), e.g. [[True], [True], ...]
		 	return mask: (batch, n_nodes, 1)		
		"""
		self.is_next_depot = next_node == 0
		D = tf.where(self.is_next_depot, 1.0, D)
		self.visited_customer = self.visited_customer | visited_mask[:,1:,:]
		customer_idx = tf.argmax(tf.cast(visited_mask[:,1:,:], tf.int32), axis = 1)
		selected_demand = tf.gather(params = self.demand, indices = customer_idx, batch_dims = 1)
		D = D - selected_demand * (1.0 - tf.cast(self.is_next_depot, tf.float32))
		capacity_over_customer = self.demand > D
		mask_customer = capacity_over_customer[:, :, None] | self.visited_customer

		# print('mask_customer[0]', mask_customer[0])
		mask_depot = self.is_next_depot & (tf.reduce_sum(tf.cast(mask_customer == False, tf.int32), axis = 1) > 0)
		# print('mask_depot', mask_depot[0])

		""" # mask_depot = tf.math.logical_not(tf.reduce_all(mask_customer, axis = 1))
			tf.reduce_all: if there's any False on the specified axis, return False
			# mask_depot = self.is_next_depot | tf.reduce_all(mask_customer, axis = 1)
			We can choose depot if 1) we are not in depot or 2) all nodes are visited
			if the mask for customer nodes are all True, mask_depot should be False so that the vehicle can return back to depot 
			even if some of the mask for customer nodes are False, mask_depot should be False so that vehicle could go back to the depot
			the vechile must not be at the depot in a low but it can stay at the depot when the mask for customer nodes are all True
		"""
		return tf.concat([mask_depot[:, None, :], mask_customer], axis = 1), D
	
	def _get_step(self, next_node, D):
		""" next_node **includes depot** : (batch, 1) tf.int32, range[0, n_nodes-1]
			--> one_hot: (batch, 1, n_nodes)
			prev_node_embedding: (batch, 1, embed_dim)
		"""
		one_hot = tf.one_hot(indices = next_node, depth = self.n_nodes)		
		visited_mask = tf.transpose(tf.cast(one_hot, dtype = tf.bool), (0,2,1))
		mask, D = self.get_mask_D(next_node, visited_mask, D)
		self.demand = tf.where(self.visited_customer[:,:,0], 0.0, self.demand)
		# prev_node_embedding = tf.matmul(one_hot, self.node_embeddings)
		prev_node_embedding = tf.gather(self.node_embeddings, indices = next_node, batch_dims = 1)

		context = tf.concat([prev_node_embedding, D[:,:,None]], axis = -1)
		return mask, context, D

	def _create_t1(self):
		mask_t1 = self.create_mask_t1()
		step_context_t1, D_t1 = self.create_context_D_t1()
		return mask_t1, step_context_t1, D_t1

	def create_mask_t1(self):
		mask_customer = self.visited_customer
		mask_depot = tf.ones([self.batch, 1, 1], dtype = tf.bool)
		return tf.concat([mask_depot, mask_customer], axis = 1)

	def create_context_D_t1(self):
		D_t1 = tf.ones([self.batch, 1], dtype=tf.float32)
		depot_idx = tf.zeros([self.batch, 1], dtype = tf.int32)
		depot_embedding = tf.gather(self.node_embeddings, indices = depot_idx, batch_dims = 1)
		return tf.concat([depot_embedding, D_t1[:,:,None]], axis = -1), D_t1

	def get_log_likelihood(self, _log_p, pi):
		# Get log_p corresponding to selected actions
		log_p = tf.gather_nd(_log_p, tf.expand_dims(pi, axis = -1), batch_dims = 2)
		return tf.reduce_sum(log_p, 1)

	def get_costs(self, pi):
		""" self.xy: (batch, n_nodes, 2), Coordinates of depot + customer nodes
			pi: (batch, decode_step), predicted tour
			d: (batch, decode_step, 2)
			Note: first element of pi is not depot, the first selected node in the path
		"""
		d = tf.gather(self.xy, indices = pi, batch_dims = 1)
		return (tf.reduce_sum(tf.norm(d[:, 1:] - d[:, :-1], ord=2, axis=2), axis=1)
					+ tf.norm(d[:, 0] - self.depot_xy, ord=2, axis=1)# distance from depot to first selected node
					+ tf.norm(d[:, -1] - self.depot_xy, ord=2, axis=1))# distance from last selected node (!=0 for graph with longest path) to depot



class Sampler(tf.keras.layers.Layer):
	""" logits: (batch, n_nodes)
			TopKSampler <-- greedy; sample one with biggest probability
			CategoricalSampler <-- sampling; randomly sample one from possible distribution based on probability
	"""
	def __init__(self, n_samples = 1, **kwargs):
		super().__init__(**kwargs)
		self.n_samples = n_samples
		
class TopKSampler(Sampler):
	def call(self, logits):
		return tf.math.top_k(logits, self.n_samples).indices

class CategoricalSampler(Sampler):
	def call(self, logits):
		return tf.random.categorical(logits, self.n_samples, dtype = tf.int32)

