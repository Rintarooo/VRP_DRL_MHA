import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

from model import AttentionModel
from baseline import RolloutBaseline
from data import generate_data, Generator
from config import Config, load_pkl, file_parser

def train(cfg, log_path = None):

	def rein_loss(model, inputs, bs, t, device):
		inputs = list(map(lambda x: x.to(device), inputs))
		L, ll = model(inputs, decode_type = 'sampling')
		b = bs[t] if bs is not None else baseline.eval(inputs, L)
		return ((L - b.detach().to(device)) * ll).mean(), L.mean()
	
	model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping)
	model.train()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	baseline = RolloutBaseline(model, cfg.task, cfg.weight_dir, cfg.n_rollout_samples, 
								cfg.embed_dim, cfg.n_customer, cfg.warmup_beta, cfg.wp_epochs, device)
	optimizer = optim.Adam(model.parameters(), lr = cfg.lr)
	
	t1 = time()
	for epoch in range(cfg.epochs):
		ave_loss, ave_L = 0.0, 0.0
		dataset = Generator(cfg.batch*cfg.batch_steps, cfg.n_customer)
		
		bs = baseline.eval_all(dataset)
		bs = bs.view(-1, cfg.batch) if bs is not None else None# bs: (cfg.batch_steps, cfg.batch) or None
		
		dataloader = DataLoader(dataset, batch_size = cfg.batch, shuffle = True)
		for t, inputs in enumerate(dataloader):
			loss, L_mean = rein_loss(model, inputs, bs, t, device)

			optimizer.zero_grad()
			loss.backward()

			print('grad: ', model.Decoder.Wk1.weight.data.grad)
			# https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py
			# https://github.com/Rintarooo/TSP_DRL_PointerNet/blob/master/train.py
			nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type = 2)
			optimizer.step()
			
			ave_loss = loss.item()
			ave_L = L_mean.item()
			
			if t%(cfg.batch_verbose) == 0:
				t2 = time()
				print('Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec'%(
					epoch, t, ave_loss, ave_L, (t2-t1)//60, (t2-t1)%60))
				if cfg.islogger:
					if log_path is None:
						log_path = '%s%s_%s.csv'%(cfg.log_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
						with open(log_path, 'w') as f:
							f.write('time,epoch,batch,loss,cost\n')
					with open(log_path, 'a') as f:
						f.write('%dmin%dsec,%d,%d,%1.3f,%1.3f\n'%(
							(t2-t1)//60, (t2-t1)%60, epoch, t, ave_loss, ave_L))
				t1 = time()

		baseline.epoch_callback(model, epoch)
		torch.save(model.state_dict(), '%s%s_epoch%s.h5'%(cfg.weight_dir, cfg.task, epoch))

if __name__ == '__main__':
	cfg = load_pkl(file_parser().path)
	train(cfg)	