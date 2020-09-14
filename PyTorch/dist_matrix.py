import torch
import math
import numpy as np

from data import generate_data

def get_dist(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError

def get_dist_matrix(points, digit = 2):
	n = len(points)
	dist = [[0 for i in range(n)] for i in range(n)]
	for i in range(n):
		for j in range(i, n):
			two = get_dist(points[i], points[j])
			dist[i][j] = dist[j][i] = round(float(two), digit)
	return dist
	
if __name__ == '__main__':
	""" x[0] -- depot_xy: (batch, 2)
		x[1] -- customer_xy: (batch, n_nodes-1, 2)
		xy: (batch, n_nodes, 2)
	"""
	batch = 0
	x = generate_data()
	xy = torch.cat([x[0][:,None,:], x[1]], dim = 1)
	print(xy.size())
	dist = get_dist_matrix(xy[batch])
	print(dist*20)
	