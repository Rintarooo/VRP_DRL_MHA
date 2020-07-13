import tensorflow as tf
from tqdm import tqdm
from time import time

from model import AttentionModel
from rollout import RolloutBaseline
from data import generate_data
from config import Config, load_pkl, file_parser

def train(cfg, log_path = None):

	def allocate_memory():
	# https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0
		physical_devices = tf.config.experimental.list_physical_devices('GPU')
		if len(physical_devices) > 0:
			for k in range(len(physical_devices)):
				tf.config.experimental.set_memory_growth(physical_devices[k], True)
				print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
		else:
			print("Not enough GPU hardware devices available")

	
	allocate_memory()
	model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping)
	baseline = RolloutBaseline(model, cfg.task, cfg.weight_dir, cfg.n_rollout_samples, 
								cfg.embed_dim, cfg.n_customer, cfg.warmup_beta, cfg.wp_epochs)
	optimizer = tf.keras.optimizers.Adam(learning_rate = cfg.lr)
	ave_loss = tf.keras.metrics.Mean()
	ave_L = tf.keras.metrics.Mean()
	
	for epoch in range(cfg.epochs):
		dataset = generate_data(cfg.n_samples, cfg.n_customer)
		
		bs = baseline.eval_all(dataset)
		bs = tf.reshape(bs, (-1, cfg.batch)) if bs is not None else None # bs: (cfg.batch_steps, cfg.batch) or None
		
		t1 = time()
		for t, inputs in enumerate(dataset.batch(cfg.batch)):
			with tf.GradientTape() as tape:
				L, ll = model(inputs, decode_type = 'sampling', training = True)
				b = bs[t] if bs is not None else baseline.eval(inputs, L)
				b = tf.stop_gradient(b)
				loss = tf.reduce_mean((L - b) * ll)
				L_mean = tf.reduce_mean(L)
			grads = tape.gradient(loss, model.trainable_weights)# model.trainable_weights == thita
			
			grads, _ = tf.clip_by_global_norm(grads, 1.0)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))# optimizer.step
			
			ave_loss.update_state(loss)
			ave_L.update_state(L_mean)
			
			if t%(cfg.batch_verbose) == 0:
				print('Epoch %d (batch = %d): Loss: %1.2f L: %1.2f'%(epoch, t, ave_loss.result().numpy(), ave_L.result().numpy()))
				t2 = time()
				print('%dmin%dsec'%((t2-t1)//60, (t2-t1)%60))
				if cfg.islogger:
					if log_path is None:
						log_path = '%s%s_%s.csv'%(cfg.log_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
						with open(log_path, 'w') as f:
							f.write('time,epoch,batch,loss,cost\n')
					with open(log_path, 'a') as f:
						f.write('%dmin%dsec,%d,%d,%1.2f,%1.2f\n'%((t2-t1)//60, (t2-t1)%60, epoch, t, ave_loss.result().numpy(), ave_L.result().numpy()))
				t1 = time()
			ave_loss.reset_states()
			ave_L.reset_states()

		baseline.epoch_callback(model, epoch)
		model.save_weights('%s%s_epoch%s.h5'%(cfg.weight_dir, cfg.task, epoch), save_format = 'h5')

if __name__ == '__main__':
	cfg = load_pkl(file_parser().path)
	train(cfg)


	