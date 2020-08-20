import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

def generate_data(n_samples = 10, n_customer = 20, seed = None):
	""" https://pytorch.org/docs/master/torch.html?highlight=rand#torch.randn
	"""
	if seed is not None:
		torch.manual_seed(seed)
	CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	if torch.cuda.is_available():
		if n_samples == 1:# squeeze(0)
			return (torch.cuda.FloatTensor(n_samples, 2).uniform_(0, 1).squeeze(0),
					torch.cuda.FloatTensor(n_samples, n_customer, 2).uniform_(0, 1).squeeze(0),
					torch.cuda.FloatTensor(n_samples, n_customer).random_(1, 10).squeeze(0) / CAPACITIES[n_customer])

		return (torch.cuda.FloatTensor(n_samples, 2).uniform_(0, 1),
				torch.cuda.FloatTensor(n_samples, n_customer, 2).uniform_(0, 1),
				torch.cuda.FloatTensor(n_samples, n_customer).random_(1, 10) / CAPACITIES[n_customer])
	
	else:
		if n_samples == 1:# squeeze(0)
			return (torch.FloatTensor(n_samples, 2).uniform_(0, 1).squeeze(0),
					torch.FloatTensor(n_samples, n_customer, 2).uniform_(0, 1).squeeze(0),
					torch.FloatTensor(n_samples, n_customer).random_(1, 10).squeeze(0) / CAPACITIES[n_customer])

		return (torch.FloatTensor(n_samples, 2).uniform_(0, 1),
				torch.FloatTensor(n_samples, n_customer, 2).uniform_(0, 1),
				torch.FloatTensor(n_samples, n_customer).random_(1, 10) / CAPACITIES[n_customer])

	# if n_samples == 1:# squeeze(0)
	# # np.random.rand --> generate uniformly distributed over [0, 1) 
	# # https://docs.scipy.org/doc//numpy-1.15.0/reference/generated/numpy.random.uniform.html
	# 	return (torch.tensor(np.random.rand(n_samples, 2), dtype = torch.float, requires_grad = True).squeeze(0),
	# 			torch.tensor(np.random.rand(n_samples, n_customer, 2), dtype = torch.float, requires_grad = True).squeeze(0),
	# 			torch.tensor(np.random.randint(1, 10, size = (n_samples, n_customer))/ CAPACITIES[n_customer], dtype = torch.float, requires_grad = True).squeeze(0))# 1 - 9

	# return (torch.tensor(np.random.rand(n_samples, 2), dtype = torch.float, requires_grad = True),# np.random.rand --> generate uniformly distributed over [0, 1) 
	# 			torch.tensor(np.random.rand(n_samples, n_customer, 2), dtype = torch.float, requires_grad = True),
	# 			torch.tensor(np.random.randint(1, 10, size = (n_samples, n_customer))/ CAPACITIES[n_customer], dtype = torch.float, requires_grad = True))# 1 - 9


class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
		https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
	"""
	def __init__(self, n_samples = 5120, n_customer = 20, seed = None):
		# n_samples = batch * batch_steps
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		if seed is not None:
			# self.data_list = [generate_data(1, n_customer, seed+i) for i in range(n_samples)]
			self.data_list = [generate_data(1, n_customer, seed+i) for i in tqdm(range(n_samples), disable = False, desc = 'Generate input data')]
		else:
			# self.data_list = [generate_data(1, n_customer) for i in range(n_samples)]
			self.data_list = [generate_data(1, n_customer) for i in tqdm(range(n_samples), disable = False, desc = 'Generate input data')]

	def __getitem__(self, idx):
		return self.data_list[idx]

	def __len__(self):
		return len(self.data_list)

def _print(data):
	print(data[0].size())
	print(data[1].size())
	print(data[2].size())
	
if __name__ == '__main__':
	data = generate_data(n_samples = 128, n_customer = 20, seed = 123)
	for i in range(3):
		print(data[i].dtype)
	_print(data)

	batch = 512
	batch_steps = 10
	dataset = Generator(n_samples = batch*batch_steps)
	
	# data = next(iter(dataset))
	# _print(data)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)
	for i, data in enumerate(dataloader):
		# for da in data:
		# 	print(da)
		new_input = list(map(lambda x: x.to(device), data))
		if i == 0:
			break