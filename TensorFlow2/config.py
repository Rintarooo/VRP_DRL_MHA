import pickle
import os
import argparse
from datetime import datetime

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', metavar = 'M', type = str, default = 'train', choices = ['train', 'test'], help = 'train or test')
	parser.add_argument('--seed', metavar = 'SE', type = int, default = 123, help = 'random seed number for inference, reproducibility')
	parser.add_argument('-n', '--n_customer', metavar = 'N', type = int, default = 20, help = 'number of customer nodes, time sequence')

	# train config
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 512, help = 'batch size')
	parser.add_argument('-bs', '--batch_steps', metavar = 'BS', type = int, default = 2500, help = 'number of samples = batch * batch_steps')
	parser.add_argument('-bv', '--batch_verbose', metavar = 'BV', type = int, default = 10, help = 'print and logging during training process')
	parser.add_argument('-nr', '--n_rollout_samples', metavar = 'R', type = int, default = 10000, help = 'baseline rollout number of samples')
	parser.add_argument('-e', '--epochs', metavar = 'E', type = int, default = 20, help = 'total number of samples = epochs * number of samples')
	parser.add_argument('-em', '--embed_dim', metavar = 'EM', type = int, default = 128, help = 'embedding size')
	parser.add_argument('-nh', '--n_heads', metavar = 'NH', type = int, default = 8, help = 'number of heads in MHA')
	parser.add_argument('-c', '--tanh_clipping', metavar = 'C', type = float, default = 10., help = 'improve exploration; clipping logits')
	parser.add_argument('-ne', '--n_encode_layers', metavar = 'NE', type = int, default = 3, help = 'number of MHA encoder layers')
	
	parser.add_argument('--lr', metavar = 'LR', type = float, default = 1e-4, help = 'initial learning rate')
	parser.add_argument('-wb', '--warmup_beta', metavar = 'WB', type = float, default = 0.8, help = 'exponential moving average, warmup')
	parser.add_argument('-we', '--wp_epochs', metavar = 'WE', type = int, default = 1, help = 'warmup epochs')
	
	parser.add_argument('--islogger', action = 'store_false', help = 'flag csv logger default true')
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
						default = 'Pkl/VRP20_train.pkl', help = 'file path, pkl or h5 only')
	args = parser.parse_args()
	return args

def test_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, required = True,  
						help = 'Weights/VRP***_train_epoch***.h5, h5 file required')
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 2, help = 'batch size')
	parser.add_argument('-n', '--n_customer', metavar = 'N', type = int, default = 20, help = 'number of customer nodes, time sequence')
	parser.add_argument('-s', '--seed', metavar = 'S', type = int, default = 123, help = 'random seed number for inference, reproducibility')
	parser.add_argument('-t', '--txt', metavar = 'T', type = str, help = 'if you wanna test out on text file, example: ../OpenData/A-n53-k7.txt')
	parser.add_argument('-d', '--decode_type', metavar = 'D', type = str, required = True, choices = ['greedy', 'sampling'], help = 'greedy or sampling required')
	args = parser.parse_args()
	return args
	
if __name__ == '__main__':
	args = arg_parser()
	dump_pkl(args)
	# cfg = load_pkl(file_parser().path)
	# for k, v in vars(cfg).items():
	# 	print(k, v)
	# 	print(vars(cfg)[k])#==v
