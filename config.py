import pickle
import os
import argparse
import torch

def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', metavar = 'M', type = str, default = 'test', choices = ['train', 'test'], help = 'train or test')
	parser.add_argument('--seed', metavar = 'SEED', type = int, default = 1, help = 'random seed number for inference, reproducibility')
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 128, help = 'batch size')
	parser.add_argument('--xy', metavar = 'XY', type = int, default = 2, help = 'xy coordinate size input size')
	parser.add_argument('-ct', '--city_t', metavar = 'CT', type = int, default = 20, help = 'number of cities, time sequence')
	parser.add_argument('-e', '--embed', metavar = 'EM', type = int, default = 128, help = 'embedding size')
	parser.add_argument('-hi', '--hidden', metavar = 'HI', type = int, default = 128, help = 'hidden size')
	parser.add_argument('-c', '--clip_logits', metavar = 'C', type = int, default = 10, help = 'improve exploration; clipping logits')
	parser.add_argument('-st', '--softmax_T', metavar = 'T', type = float, default = 1.0, help = 'improve exploration; softmax temperature, 2.0, 2.2 and 1.5 to yield the best results')
	parser.add_argument('-s', '--steps', metavar = 'S', type = int, default = 100, help = '100 training steps(epochs)')
	parser.add_argument('-o', '--optim', metavar = 'O', type = str, default = 'Adam', help = 'torch optimizer')
	parser.add_argument('--lr', metavar = 'LR', type = float, default = 1e-3, help = 'initial learning rate')
	parser.add_argument('--lr_decay', metavar = 'LRD', type = float, default = 0.96, help = 'learning rate scheduler, decay by a factor of 0.96 ')
	parser.add_argument('--lr_decay_step', metavar = 'LRDS', type = int, default = 5e3, help = 'learning rate scheduler, decay every 5000 steps')
	parser.add_argument('-al', '--alpha', metavar = 'ALP', type = float, default = 0.99, help = 'alpha decay in active search')
	parser.add_argument('-minv', '--init_min', metavar = 'MINV', type = float, default = -0.08, help = 'initialize weight minimun value -0.08~')
	parser.add_argument('-maxv', '--init_max', metavar = 'MAXV', type = float, default = 0.08, help = 'initialize weight ~0.08 maximum value')
	
	parser.add_argument('--islogger', action = 'store_false', help = 'flag csv logger default true')
	parser.add_argument('--issaver', action = 'store_false', help = 'flag model saver default true')
	parser.add_argument('-ls', '--log_step', metavar = 'LOGS', type = int, default = 10, help = 'logger timing')
	parser.add_argument('-ld', '--log_dir', metavar = 'LD', type = str, default = './Csv/', help = 'csv logger dir')
	parser.add_argument('-md', '--model_dir', metavar = 'MD', type = str, default = './Pt/', help = 'model save dir')
	parser.add_argument('-pd', '--pkl_dir', metavar = 'PD', type = str, default = './Pkl/', help = 'pkl save dir')
	parser.add_argument('-ap', '--act_model_path', metavar = 'AMP', type = str, default = './Pt/0510_act.pt', help = 'load actor model path')
	parser.add_argument('-cd', '--cuda_dv', metavar = 'CD', type = str, default = '0', help = 'os CUDA_VISIBLE_DEVICE')
	args = parser.parse_args()
	return args

class Config():
	def __init__(self, **kwargs):	
		for k, v in kwargs.items():
			self.__dict__[k] = v
		self.task = 'VRP%d'%self.city_t
		self.pkl_path = self.pkl_dir + self.task + '_%s.pkl'%self.mode
		
def dump_pkl(args):
	cfg = Config(**vars(args))
	with open(cfg.pkl_path, 'wb') as f:
		pickle.dump(cfg, f)
		print('--- save pickle file in %s ---\n'%cfg.pkl_path)
	
def load_pkl(pkl_path, verbose = True):
	if not os.path.isfile(pkl_path):
		raise FileNotFoundError('pkl_path')
	with open(pkl_path, 'rb') as f:
		cfg = pickle.load(f)
		if verbose:
			print(''.join('%s: %s\n'%item for item in vars(cfg).items()))
		os.environ['CUDA_VISIBLE_DEVICE'] = cfg.cuda_dv
	return cfg

def pkl_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, 
						default = './Pkl/test.pkl', help = 'pkl file name')
	args = parser.parse_args()
	return args
	
if __name__ == '__main__':
	args = argparser()
	dump_pkl(args)
	# cfg = load_pkl('./Pkl/test.pkl')
	# for k, v in vars(cfg).items():
	# 	print(k, v)
	# 	print(vars(cfg)[k])#==v
