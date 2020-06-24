import tensorflow as tf
from tqdm import tqdm
from model import AttentionModel
from _ import generate_inputs

def REINFORCE():
	with tf.GradientTape() as g:
		inputs = generate_inputs()
		L, logp = model(inputs)
		b = baseline.eval(inputs)
		adv = L - tf.stop_gradient(b)
		grad_target, L_mean = tf.reduce_mean(adv * logp), tf.reduce_mean(L)
	grad = g.gradient(zip(grad_target, model.trainable_variables))
	return grad_target, grad, L_mean

def train_epoch(model):
	for epoch in tqdm(range(cfg.epochs)):
		grad_target, grad, L_mean = REINFORCE()
		optimizer.apply_gradients(zip(grad, model.trainable_variables))

		act_loss.update_state(grad_target)

	
def train(cfg):
	act_loss = tf.keras.metrics.Mean()
	optimizer = tf.keras.optimizers.Adam(learning_rate = lr)


	train_epoch()
