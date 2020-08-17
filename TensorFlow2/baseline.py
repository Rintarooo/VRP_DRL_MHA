import tensorflow as tf
from scipy.stats import ttest_rel
from tqdm import tqdm

from data import generate_data
from model import AttentionModel

def copy_model(model, embed_dim = 128, n_customer = 20):
	""" Copy model weights to new model
		https://stackoverflow.com/questions/56841736/how-to-copy-a-network-in-tensorflow-2-0
	"""
	small_dataset = generate_data(n_samples = 5, n_customer = n_customer)
	new_model = AttentionModel(embed_dim)
	for data in (small_dataset.batch(5)):
		# _, _ = model(data, decode_type = 'sampling')
		cost, _ = new_model(data, decode_type = 'sampling')
		
	for a, b in zip(new_model.variables, model.variables):
		a.assign(b)# copies the weigths variables of model_b into model_a
	return new_model

def load_model(path, embed_dim = 128, n_customer = 20, n_encode_layers = 3):
	""" Load model weights from hd5 file
		https://stackoverflow.com/questions/51806852/cant-save-custom-subclassed-model
	"""
	small_dataset = generate_data(n_samples = 5, n_customer = n_customer)
	model_loaded = AttentionModel(embed_dim, n_encode_layers = n_encode_layers)
	for data in (small_dataset.batch(5)):
		_, _ = model_loaded(data, decode_type = 'greedy')

	model_loaded.load_weights(path)
	return model_loaded

def rollout(model, dataset, batch = 1000, disable_tqdm = False):
	costs_list = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True, element_shape = (batch,))
	for i, inputs in tqdm(enumerate(dataset.batch(batch)), disable = disable_tqdm, desc = 'Rollout greedy execution'):
		cost, _ = model(inputs, decode_type = 'greedy')
		costs_list = costs_list.write(i, cost)
	return tf.reshape(costs_list.stack(), (-1,))

# def validate(dataset, model, batch = 1000):
# 	"""Validates model on given dataset in greedy mode
# 	"""
# 	val_costs = rollout(model, dataset, batch = batch)
# 	mean_cost = tf.reduce_mean(val_costs)
# 	print(f"Validation score: {np.round(mean_cost, 4)}")
# 	return mean_cost

class RolloutBaseline:

	def __init__(self, model, task, weight_dir, n_rollout_samples = 10000, 
				embed_dim = 128, n_customer = 20, warmup_beta = 0.8, wp_epochs = 1, 
				from_checkpoint = False, path_to_checkpoint = None, epoch = 0,
				):
		"""
		Args:
			model: current model
			task: suffix for baseline checkpoint task
			from_checkpoint: start from checkpoint flag
			path_to_checkpoint: path to baseline model weights
			wp_epochs: until when epoch reaches wp_n_epocohs do we warm-up
			epoch: current epoch number
			n_rollout_samples: number of samples to be generated for baseline dataset
			warmup_beta: warmup mixing parameter (exp. exponential moving average parameter)
		"""

		self.n_rollout_samples = n_rollout_samples
		self.cur_epoch = epoch
		self.wp_epochs = wp_epochs
		self.beta = warmup_beta

		# controls the amount of warmup
		self.alpha = 0.0

		self.M = None

		# Checkpoint params
		self.task = task
		self.from_checkpoint = from_checkpoint
		self.path_to_checkpoint = path_to_checkpoint

		# Problem params
		self.embed_dim = embed_dim
		self.n_customer = n_customer
		self.weight_dir = weight_dir

		# create and evaluate initial baseline
		self._update_baseline(model, epoch)


	def _update_baseline(self, model, epoch):

		# Load or copy baseline model based on self.from_checkpoint condition
		if self.from_checkpoint and self.alpha == 0:
			print('Baseline model loaded')
			self.model = load_model(self.path_to_checkpoint, embed_dim = self.embed_dim, n_customer = self.n_customer)
		else:
			print('Baseline model copied')
			self.model = copy_model(model, embed_dim = self.embed_dim, n_customer = self.n_customer)
			# For checkpoint
			self.model.save_weights('%s%s_baseline_epoch%s.h5'%(self.weight_dir, self.task, epoch), save_format = 'h5')
		# We generate a new dataset for baseline model on each baseline update to prevent possible overfitting
		self.dataset = generate_data(n_samples = self.n_rollout_samples, n_customer = self.n_customer)
		
		print(f'Evaluating baseline model on baseline dataset (epoch = {epoch})')
		self.bl_vals = rollout(self.model, self.dataset)
		self.mean = tf.reduce_mean(self.bl_vals)
		self.cur_epoch = epoch

	def ema_eval(self, cost):# def eval
		"""exponential moving average (only for warm-up epochs)
		"""
		if self.M is None:# first iteration
			self.M = tf.reduce_mean(cost)
		else:
			self.M = self.beta * self.M + (1. - self.beta) * tf.reduce_mean(cost)
		return self.M

	def eval(self, batch, cost):
		"""Evaluates current baseline model on single training batch
		"""
		if self.alpha == 0:
			return self.ema_eval(cost)

		if self.alpha < 1:
			v_ema = self.ema_eval(cost)
		else:
			v_ema = 0.0

		v_b, _ = self.model(batch, decode_type = 'greedy')

		v_b = tf.stop_gradient(v_b)
		v_ema = tf.stop_gradient(v_ema)

		# Combination of baseline cost and exp. moving average cost
		return self.alpha * v_b + (1 - self.alpha) * v_ema

	def eval_all(self, dataset):
		"""Evaluates current baseline model on the whole dataset only for non warm-up epochs
		"""
		if self.alpha < 1:
			return None

		val_costs = rollout(self.model, dataset, batch = 2048)

		return val_costs

	def epoch_callback(self, model, epoch):
		"""Compares current baseline model with the training model and updates baseline if it is improved
		"""
		self.cur_epoch = epoch

		print(f'Evaluating candidate model on baseline dataset (callback epoch = {self.cur_epoch})')
		candidate_vals = rollout(model, self.dataset)# costs for training model on baseline dataset
		candidate_mean = tf.reduce_mean(candidate_vals)

		print(f'Epoch {self.cur_epoch} candidate mean {candidate_mean}, baseline mean {self.mean}')

		if candidate_mean < self.mean:
			t, p = ttest_rel(candidate_vals, self.bl_vals)# scipy.stats.ttest_rel

			p_val = p / 2
			print(f'p-value: {p_val}')

			if p_val < 0.05:
				print('Update baseline')
				self._update_baseline(model, self.cur_epoch)

		# alpha controls the amount of warmup
		if self.alpha < 1.0:
			self.alpha = (self.cur_epoch + 1) / float(self.wp_epochs)
			print(f'alpha was updated to {self.alpha}')

