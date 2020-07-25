import tensorflow as tf

class VRPproblem():

    VEHICLE_CAPACITY = 1.0

    def __init__(self, input):

        depot = input[0]
        loc = input[1]

        self.batch_size, self.n_loc, _ = loc.shape  # (batch_size, n_nodes, 2)

        # Coordinates of depot + other nodes
        self.coords = tf.concat((depot[:, None, :], loc), -2)
        self.demand = tf.cast(input[2], tf.float32)

        # Indices of graphs in batch
        self.ids = tf.range(self.batch_size, dtype=tf.int64)[:, None]

        # State
        self.prev_a = tf.zeros((self.batch_size, 1), dtype=tf.float32)
        self.used_capacity = tf.zeros((self.batch_size, 1), dtype=tf.float32)
        self.from_depot = self.prev_a == 0

        # Nodes that have been visited will be marked with 1
        self.visited = tf.zeros((self.batch_size, 1, self.n_loc + 1), dtype=tf.uint8)

        # Step counter
        self.i = tf.zeros(1, dtype=tf.int64)

        # Constant tensors for scatter update (in step method)
        self.step_updates = tf.ones((self.batch_size, 1), dtype=tf.uint8)  # (batch_size, 1)
        self.scatter_zeros = tf.zeros((self.batch_size, 1), dtype=tf.int64)  # (batch_size, 1)

    def all_finished(self):
        return self.i.numpy() >= self.demand.shape[-1] and tf.reduce_all(tf.cast(self.visited, tf.bool))

    def get_mask(self):
        """ Returns a mask (batch_size, 1, n_nodes) with available actions.
            Impossible nodes are masked.
        """

        # Exclude depot
        visited_loc = self.visited[:, :, 1:]

        # Mark nodes which exceed vehicle capacity
        exceeds_cap = self.demand + self.used_capacity > self.VEHICLE_CAPACITY

        # We mask nodes that are already visited or have too much demand
        mask_loc = tf.cast(visited_loc, tf.bool) | exceeds_cap[:, None, :]

        # We can choose depot if 1) we are not in depot OR 2) all nodes are visited
        mask_depot = self.from_depot & (tf.reduce_sum(tf.cast(mask_loc == False, tf.int32), axis=-1) > 0)

        return tf.concat([mask_depot[:, :, None], mask_loc], axis=-1)

    def step(self, action):

        # Update current state
        selected = action[:, None]

        self.prev_a = selected
        self.from_depot = self.prev_a == 0

        # We have to shift indices by 1 since demand doesn't include depot
        selected_demand = tf.gather_nd(self.demand,
                                       tf.concat([self.ids, tf.clip_by_value(self.prev_a - 1, 0, self.n_loc - 1)], axis=1)
                                       )[:, None]  # (batch_size, 1)

        # We add current node capacity to used capacity and set it to zero if we return to the depot
        self.used_capacity = (self.used_capacity + selected_demand) * (1.0 - tf.cast(self.from_depot, tf.float32))

        # Update visited nodes (set 1 to visited nodes)
        idx = tf.cast(tf.concat((self.ids, self.scatter_zeros, self.prev_a), axis=-1), tf.int32)[:, None, :]  # (batch_size, 1, 3)
        self.visited = tf.tensor_scatter_nd_update(self.visited, idx, self.step_updates)  # (batch_size, 1, n_nodes)

        self.i = self.i + 1

    @staticmethod
    def get_costs(dataset, pi):

        # Place nodes with coordinates in order of decoder tour
        loc_with_depot = tf.concat([dataset[0][:, None, :], dataset[1]], axis=1)  # (batch_size, n_nodes, 2)
        d = tf.gather(loc_with_depot, tf.cast(pi, tf.int32), batch_dims=1)

        # Calculation of total distance
        # Note: first element of pi is not depot, but the first selected node in the path
        return (tf.reduce_sum(tf.norm(d[:, 1:] - d[:, :-1], ord=2, axis=2), axis=1)
                + tf.norm(d[:, 0] - dataset[0], ord=2, axis=1) # Distance from depot to first selected node
                + tf.norm(d[:, -1] - dataset[0], ord=2, axis=1))  # Distance from last selected node (!=0 for graph with longest path) to depot
