import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

def generate_data(n_samples = 10, n_customer = 20, seed = None):
	""" https://pytorch.org/docs/master/torch.html?highlight=rand#torch.randn
		x[0] -- depot_xy: (batch, 2)
		x[1] -- customer_xy: (batch, n_nodes-1, 2)
		x[2] -- demand: (batch, n_nodes-1)
	"""
	if seed is not None:
		torch.manual_seed(seed)
	CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
	# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

	"""
	if n_samples == 1:# squeeze(0)
	# np.random.rand --> generate uniformly distributed over [0, 1) 
	# https://docs.scipy.org/doc//numpy-1.15.0/reference/generated/numpy.random.uniform.html
		return (torch.tensor(np.random.rand(n_samples, 2), dtype = torch.float, requires_grad = True).squeeze(0),
				torch.tensor(np.random.rand(n_samples, n_customer, 2), dtype = torch.float, requires_grad = True).squeeze(0),
				torch.tensor(np.random.randint(1, 10, size = (n_samples, n_customer))/ CAPACITIES[n_customer], dtype = torch.float, requires_grad = True).squeeze(0))# 1 - 9

	return (torch.tensor(np.random.rand(n_samples, 2), dtype = torch.float, requires_grad = True),# np.random.rand --> generate uniformly distributed over [0, 1) 
				torch.tensor(np.random.rand(n_samples, n_customer, 2), dtype = torch.float, requires_grad = True),
				torch.tensor(np.random.randint(1, 10, size = (n_samples, n_customer))/ CAPACITIES[n_customer], dtype = torch.float, requires_grad = True))# 1 - 9
	"""


def data_from_txt(path):
	if not os.path.isfile(path):
		raise FileNotFoundError	
	with open(path, 'r') as f:
		lines = list(map(lambda s: s.strip(), f.readlines()))
		customer_xy, demand = [], []
		ZERO, DEPOT, CUSTO, DEMAND = [False for i in range(4)]
		ZERO = True
		for line in lines:
			if(ZERO):
				if(line == 'NODE_COORD_SECTION'):
					ZERO = False
					DEPOT = True

			elif(DEPOT):
				depot_xy = list(map(lambda k: float(k)/100., line.split()))[1:]# depot_xy.append(list(map(int, line.split()))[1:])
				DEPOT = False
				CUSTO = True
				
			elif(CUSTO):
				if(line == 'DEMAND_SECTION'):
					DEMAND = True
					CUSTO = False
					continue
				customer_xy.append(list(map(lambda k: float(k)/100., line.split()))[1:])
			elif(DEMAND):
				if(line == '1 0'):
					continue
				elif(line == 'DEPOT_SECTION'):
					break
				else:
					demand.append(list(map(lambda k: float(k)/100., line.split()))[1])# demand.append(list(map(int, line.split()))[1])
	
	# print(np.array(depot_xy).shape)
	# print(np.array(customer_xy).shape)
	# print(np.array(demand).shape)
	
	return (torch.tensor(np.expand_dims(np.array(depot_xy), axis = 0), dtype = torch.float), 
			torch.tensor(np.expand_dims(np.array(customer_xy), axis = 0), dtype = torch.float), 
			torch.tensor(np.expand_dims(np.array(demand), axis = 0), dtype = torch.float))

class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
		https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
	"""
	def __init__(self, n_samples = 5120, n_customer = 20, seed = None):
		# n_samples = batch * batch_steps
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		if seed is not None:
			self.data_list = [generate_data(1, n_customer, seed+i) for i in tqdm(range(n_samples), disable = False, desc = 'Generate input data')]
		else:
			self.data_list = [generate_data(1, n_customer) for i in tqdm(range(n_samples), disable = False, desc = 'Generate input data')]

	def __getitem__(self, idx):
		return self.data_list[idx]

	def __len__(self):
		return len(self.data_list)
	
if __name__ == '__main__':
	data = generate_data(n_samples = 128, n_customer = 20, seed = 123)
	for i in range(3):
		print(data[i].dtype)# torch.float32
		print(data[i].size())

	# batch, batch_steps = 512, 10
	# dataset = Generator(n_samples = batch*batch_steps)
	# # data = next(iter(dataset))	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)
	# for i, data in enumerate(dataloader):
	# 	new_input = list(map(lambda x: x.to(device), data))
	# 	if i == 0:
	# 		break

	path = './OpenData/A-n53-k7.txt'
	data = data_from_txt(path)
	data = list(map(lambda x: x.to(device), data))
	for da in data:
		print(da.size())
