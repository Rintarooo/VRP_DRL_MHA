import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

def generate_data(n_samples = 10, n_customer = 20, seed = None):
	""" https://pytorch.org/docs/master/torch.html?highlight=rand#torch.randn
	"""
	if seed is not None:
		torch.manual_seed(seed)
	CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}

	if n_samples == 1:# squeeze(0)
		return (torch.FloatTensor(n_samples, 2).uniform_(0, 1).squeeze(0),
				torch.FloatTensor(n_samples, n_customer, 2).uniform_(0, 1).squeeze(0),
				torch.FloatTensor(n_samples, n_customer).random_(1, 10).squeeze(0) / CAPACITIES[n_customer])

	return (torch.FloatTensor(n_samples, 2).uniform_(0, 1),
			torch.FloatTensor(n_samples, n_customer, 2).uniform_(0, 1),
			torch.FloatTensor(n_samples, n_customer).random_(1, 10) / CAPACITIES[n_customer])

class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
		https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
	"""
	# def __init__(self, batch = 512, batch_steps = 10, n_customer = 20, seed = None):
	def __init__(self, n_samples = 5120, n_customer = 20, seed = None):
		# n_samples = batch * batch_steps
		if seed is not None:
			self.data_list = [generate_data(1, n_customer, seed+i) for i in range(n_samples)]
		else:
			self.data_list = [generate_data(1, n_customer) for i in range(n_samples)]

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
	_print(data)

	batch = 512
	batch_steps = 10
	# dataset = Generator(batch = batch, batch_steps = batch_steps)
	dataset = Generator(n_samples = batch*batch_steps)
	
	# data = next(iter(dataset))
	# _print(data)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	dataloader = DataLoader(dataset, batch_size = 128, shuffle = True)
	for i, data in enumerate(dataloader):
		for da in data:
			print(da.size())
		new_input = list(map(lambda x: x.to(device), data))
		if i == 0:
			break