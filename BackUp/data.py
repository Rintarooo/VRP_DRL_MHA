import numpy as np
import pickle
import os

def generate_vrp_data(dataset_size, vrp_size):
	CAPACITIES = {10: 20., 20: 30., 
				50: 40., 100: 50.}
	return list(zip(
			depot.tolist(),
			loc.tolist(),
			prize.tolist(),
			np.full(dataset_size, MAX_LENGTHS[op_size]).tolist()  # Capacity, same for whole dataset
			))

def dump_pkl(data, pkl_path):
	with open(pkl_path, 'wb') as f:
		pickle.dump(data, f)
		print('--- save pickle file in %s ---\n'%cfg.pkl_path)

def load_pkl(pkl_path, verbose = True):
	if not os.path.isfile(pkl_path):
		raise FileNotFoundError('pkl_path')
	with open(pkl_path, 'rb') as f:
		data = pickle.load(f)
	return data

if __name__ == '__main__':
	dataset_size = 1# how many car
	vrp_size = 20# how many cutomers 
	data = generate_vrp_data(dataset_size, vrp_size)
	pkl_path = 'vrp_data.pkl'
	dump_pkl(data, pkl_path)
	data = load_pkl(pkl_path)
	print(data[0][3])