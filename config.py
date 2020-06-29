import pickle
import os
import argparse
from datetime import datetime

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', metavar = 'M', type = str, default = 'train', choices = ['train', 'test'], help = 'train or test')
	parser.add_argument('--seed', metavar = 'SEED', type = int, default = 123, help = 'random seed number for inference, reproducibility')
	parser.add_argument('-nc', '--n_customer', metavar = 'N', type = int, default = 20, help = 'number of customer nodes, time sequence')

	# train config
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 256, help = 'batch size')
	parser.add_argument('-bs', '--batch_steps', metavar = 'S', type = int, default = 10, help = 'number of samples = batch * batch_steps')
	parser.add_argument('-nr', '--n_rollout_samples', metavar = 'R', type = int, default = 1000, help = 'baseline rollout number of samples')
	parser.add_argument('-e', '--epochs', metavar = 'E', type = int, default = 1000, help = 'total number of samples = epochs * number of samples')
	parser.add_argument('-em', '--embed_dim', metavar = 'EM', type = int, default = 128, help = 'embedding size')
	parser.add_argument('-nh', '--n_heads', metavar = 'NH', type = int, default = 8, help = 'number of heads in MHA')
	parser.add_argument('-c', '--tanh_clipping', metavar = 'C', type = float, default = 10., help = 'improve exploration; clipping logits')
	parser.add_argument('-ne', '--n_encode_layers', metavar = 'NE', type = int, default = 3, help = 'number of MHA encoder layers')
	
	parser.add_argument('--lr', metavar = 'LR', type = float, default = 1e-4, help = 'initial learning rate')
	parser.add_argument('-wb', '--warmup_beta', metavar = 'WB', type = float, default = 0.8, help = 'exponential moving average, warmup')
	parser.add_argument('-we', '--wp_epochs', metavar = 'WE', type = int, default = 10, help = 'warmup epochs')
	parser.add_argument('-minv', '--init_min', metavar = 'MINV', type = float, default = -0.08, help = 'initialize weight minimun value -0.08~')
	parser.add_argument('-maxv', '--init_max', metavar = 'MAXV', type = float, default = 0.08, help = 'initialize weight ~0.08 maximum value')
	
	parser.add_argument('--islogger', action = 'store_false', help = 'flag csv logger default true')
	parser.add_argument('-ls', '--log_step', metavar = 'LOGS', type = int, default = 10, help = 'logger timing')
	parser.add_argument('-ld', '--log_dir', metavar = 'LD', type = str, default = './Csv/', help = 'csv logger dir')
	parser.add_argument('-wd', '--weight_dir', metavar = 'MD', type = str, default = './Weights/', help = 'model weight save dir')
	parser.add_argument('-pd', '--pkl_dir', metavar = 'PD', type = str, default = './Pkl/', help = 'pkl save dir')
	parser.add_argument('-cd', '--cuda_dv', metavar = 'CD', type = str, default = '0', help = 'os CUDA_VISIBLE_DEVICE')
	args = parser.parse_args()
	return args

class Config():
	def __init__(self, **kwargs):	
		for k, v in kwargs.items():
			self.__dict__[k] = v
		self.task = 'VRP%d_%s'%(self.n_customer, self.mode)
		self.dump_date = datetime.now().strftime('%m%d_%H_%M')
		for x in [self.log_dir, self.weight_dir, self.pkl_dir]:
			os.makedirs(x, exist_ok = True)
		self.pkl_path = self.pkl_dir + self.task + '.pkl'
		self.n_samples = self.batch * self.batch_steps
		
def dump_pkl(args, verbose = True, param_log = True):
	cfg = Config(**vars(args))
	with open(cfg.pkl_path, 'wb') as f:
		pickle.dump(cfg, f)
		print('--- save pickle file in %s ---\n'%cfg.pkl_path)
		if verbose:
			print(''.join('%s: %s\n'%item for item in vars(cfg).items()))
		if param_log:
			path = '%sparam_%s_%s.csv'%(cfg.log_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
			with open(path, 'w') as f:
				f.write(''.join('%s,%s\n'%item for item in vars(cfg).items())) 
	
def load_pkl(pkl_path, verbose = True):
	if not os.path.isfile(pkl_path):
		raise FileNotFoundError('pkl_path')
	with open(pkl_path, 'rb') as f:
		cfg = pickle.load(f)
		if verbose:
			print(''.join('%s: %s\n'%item for item in vars(cfg).items()))
		os.environ['CUDA_VISIBLE_DEVICE'] = cfg.cuda_dv
	return cfg

def file_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, 
						default = './Pkl/VRP20_train.pkl', help = 'file path, pkl or h5 only')
	args = parser.parse_args()
	return args
	
if __name__ == '__main__':
	args = arg_parser()
	dump_pkl(args)
	# cfg = load_pkl(file_parser().path)
	# for k, v in vars(cfg).items():
	# 	print(k, v)
	# 	print(vars(cfg)[k])#==v
