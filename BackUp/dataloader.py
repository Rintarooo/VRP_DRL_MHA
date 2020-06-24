from torch.utils.data import Dataset, DataLoader
import torch

class VRP_dataset(Dataset):
	def __init__(self, num_samples = 100, num_nodes = 20):
		super().__init__()
		self.num_samples = num_samples
		self.num_nodes = num_nodes
		self.CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}
		self.data = []

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		for i in range(self.num_samples):
			self.dic = {}
			self.dic['xy'] = torch.FloatTensor(self.num_nodes, 2).uniform_(0, 1)
			self.dic['demand'] = (torch.FloatTensor(self.num_nodes).uniform_(0, 9).int() + 1).float() / self.CAPACITIES[self.num_nodes]
			self.dic['depot'] = torch.FloatTensor(2).uniform_(0, 1)
			self.data.append(self.dic)
		return self.data[idx]


if __name__ == '__main__':
	idx = 10
	vrp_dataset = VRP_dataset()
	print(vrp_dataset.__len__())
	print(vrp_dataset.__getitem__(idx)['xy'])
	vrp_dataloader = DataLoader(dataset = vrp_dataset, batch_size = 10)
	for data in vrp_dataloader:
		print(**data)



