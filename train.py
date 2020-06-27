import tensorflow as tf
from tqdm import tqdm
from time import time

from model import AttentionModel
from rollout import RolloutBaseline
from data import generate_data
from config import Config, load_pkl, file_parser
	
def train(cfg, log_path = None):
	
	model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, 
						cfg.tanh_clipping, 'sampling')
	baseline = RolloutBaseline(model, cfg.task, cfg.weight_dir, cfg.n_rollout_samples, 
								cfg.embed_dim, cfg.n_customer, cfg.warmup_beta, cfg.wp_epochs)
	optimizer = tf.keras.optimizers.Adam(learning_rate = cfg.lr)
	ave_loss = tf.keras.metrics.Mean()
	ave_L = tf.keras.metrics.Mean()
	
	for epoch in tqdm(range(cfg.epochs), desc = 'epoch'):
		t1 = time()
		dataset = generate_data(cfg.n_samples, cfg.n_customer)
		bs = baseline.eval_all(dataset)
		bs = tf.reshape(bs, (-1, cfg.batch)) if bs is not None else None # bs: (cfg.batch_steps, cfg.batch) or None
		
		for t, inputs in enumerate(dataset.batch(cfg.batch)):
			with tf.GradientTape() as tape:
				L, logp = model(inputs)
				b = bs[t] if bs is not None else baseline.eval(inputs, L)
				b = tf.stop_gradient(b)
				loss = tf.reduce_mean((L - b) * logp)
				L_mean = tf.reduce_mean(L)
			grad = tape.gradient(loss, model.trainable_weights)# model.trainable_weights == thita
			grad, _ = tf.clip_by_global_norm(grad, 1.0)
			optimizer.apply_gradients(zip(grad, model.trainable_weights))# optimizer.step

			ave_loss.update_state(loss)
			ave_L.update_state(L_mean)
			if t%(cfg.batch_steps*0.1) == 0:
				print('epoch%d, %d/%dsamples: loss %1.2f, average L %1.2f, baseline average b %1.2f\n'%(
						epoch, t*cfg.batch, cfg.n_samples, ave_loss.result().numpy(), ave_L.result().numpy(), tf.reduce_mean(b)))

		baseline.epoch_callback(model, epoch)
		model.decode_type = 'sampling'
		model.save_weights('%s%s_epoch%s.h5'%(cfg.weight_dir, cfg.task, epoch), save_format = 'h5')
		
		if cfg.islogger:
				if log_path is None:
					log_path = '%s%s_%s.csv'%(cfg.log_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
					with open(log_path, 'w') as f:
						f.write('time,epoch,loss,average length\n')
				with open(log_path, 'a') as f:
					t2 = time()
					f.write('%dmin%dsec,%d,%1.2f,%1.2f\n'%((t2-t1)//60, (t2-t1)%60, epoch, ave_loss.result().numpy(), ave_L.result().numpy()))

		ave_loss.reset_states()
		ave_L.reset_states()

if __name__ == '__main__':
	cfg = load_pkl(file_parser().path)
	train(cfg)


	