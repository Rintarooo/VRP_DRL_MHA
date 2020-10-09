import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}

def generate_data(device, n_samples = 10, n_customer = 20, seed = None):
	""" https://pytorch.org/docs/master/torch.html?highlight=rand#torch.randn
		x[0] -- depot_xy: (batch, 2)
		x[1] -- customer_xy: (batch, n_nodes-1, 2)
		x[2] -- demand: (batch, n_nodes-1)
	"""
	if seed is not None:
		torch.manual_seed(seed)
	
	return (torch.rand((n_samples, 2), device = device),
			torch.rand((n_samples, n_customer, 2), device = device),
			torch.randint(size = (n_samples, n_customer), low = 1, high = 10, device = device) / CAPACITIES[n_customer])

class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
		https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
		https://github.com/nperlmut31/Vehicle-Routing-Problem/blob/master/dataloader.py
	"""
	def __init__(self, device, n_samples = 5120, n_customer = 20, seed = None):
		self.tuple = generate_data(device, n_samples, n_customer)

	def __getitem__(self, idx):
		return (self.tuple[0][idx], self.tuple[1][idx], self.tuple[2][idx])

	def __len__(self):
		return self.tuple[0].size(0)

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

if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device-->', device)
	
	data = generate_data(device, n_samples = 128, n_customer = 20, seed = 123)
	for i in range(3):
	 	print(data[i].dtype)# torch.float32
	 	print(data[i].size())
	
	
	batch, batch_steps, n_customer = 128, 100000, 20
	dataset = Generator(device, n_samples = batch*batch_steps, n_customer = n_customer)
	data = next(iter(dataset))	
	
	dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)
	print('use datalodaer ...')
	for i, data in enumerate(dataloader):
		for j in range(len(data)):
			print(data[j].dtype)# torch.float32
			print(data[j].size())	
		if i == 0:
			break

	path = '../OpenData/A-n53-k7.txt'
	data = data_from_txt(path)
	print('read file ...')
	data = list(map(lambda x: x.to(device), data))
	for da in data:
		print(data[j].dtype)# torch.float32
		print(da.size())
