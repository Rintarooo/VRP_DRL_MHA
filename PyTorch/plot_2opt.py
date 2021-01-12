from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from model import AttentionModel
from data import generate_data, data_from_txt
from baseline import load_model
from config import test_parser

def get_clean_path(arr):
	"""Returns extra zeros from path.
	   Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
	"""
	p1, p2 = 0, 1
	output = []
	while p2 < len(arr):
		if arr[p1] != arr[p2]:
			output.append(arr[p1])
			if p2 == len(arr) - 1:
				output.append(arr[p2])
		p1 += 1
		p2 += 1

	if output[0] != 0:
		output.insert(0, 0)# insert 0 in 0th of the array
	if output[-1] != 0:
		output.append(0)# insert 0 at the end of the array
	return output

def plot_route(data, pi, costs, title, idx_in_batch = 0, is_tensor = True):
	"""Plots journey of agent
	Args:
		data: dataset of graphs
		pi: (batch, decode_step) # tour
		idx_in_batch: index of graph in data to be plotted
	"""
	
	if is_tensor:
		# Remove extra zeros
		pi_ = get_clean_path(pi[idx_in_batch].cpu().numpy())
		cost = costs[idx_in_batch].cpu().numpy()
	else:
		pi_ = pi
		cost = costs

	depot_xy = data[0][idx_in_batch].cpu().numpy()
	customer_xy = data[1][idx_in_batch].cpu().numpy()
	demands = data[2][idx_in_batch].cpu().numpy()
	# customer_labels = ['(' + str(i) + ', ' + str(demand) + ')' for i, demand in enumerate(demands.round(2), 1)]
	customer_labels = ['(' + str(demand) + ')' for demand in demands.round(2)]
	
	xy = np.concatenate([depot_xy.reshape(1, 2), customer_xy], axis = 0)

	# Get list with agent loops in path
	list_of_paths, cur_path = [], []
	for idx, node in enumerate(pi_):
		cur_path.append(node)

		if idx != 0 and node == 0:
			if cur_path[0] != 0:
				cur_path.insert(0, 0)
			list_of_paths.append(cur_path)
			cur_path = []

	path_traces = []
	for i, path in enumerate(list_of_paths, 1):
		coords = xy[[int(x) for x in path]]

		# Calculate length of each agent loop
		lengths = np.sqrt(np.sum(np.diff(coords, axis = 0) ** 2, axis = 1))
		total_length = np.sum(lengths)

		path_traces.append(go.Scatter(x = coords[:, 0],
									y = coords[:, 1],
									mode = 'markers+lines',
									name = f'tour{i} Length = {total_length:.3f}',
									opacity = 1.0))

	trace_points = go.Scatter(x = customer_xy[:, 0],
							  y = customer_xy[:, 1],
							  mode = 'markers+text', 
							  name = 'Customer (demand)',
							  text = customer_labels,
							  textposition = 'top center',
							  marker = dict(size = 7),
							  opacity = 1.0
							  )

	trace_depo = go.Scatter(x = [depot_xy[0]],
							y = [depot_xy[1]],
							mode = 'markers+text',
							name = 'Depot (capacity = 1.0)',
							text = ['1.0'],
							textposition = 'bottom center',
							marker = dict(size = 23),
							marker_symbol = 'triangle-up'
							)
	
	layout = go.Layout(title = dict(text = f'<b>VRP{customer_xy.shape[0]} {title}, Total Length = {cost:.3f}</b>', x = 0.5, y = 1, yanchor = 'bottom', yref = 'paper', pad = dict(b = 10)),#https://community.plotly.com/t/specify-title-position/13439/3
					   # xaxis = dict(title = 'X', ticks='outside'),
					   # yaxis = dict(title = 'Y', ticks='outside'),#https://kamino.hatenablog.com/entry/plotly_for_report
					   xaxis = dict(title = 'X', range = [0, 1], showgrid=False, ticks='outside', linewidth=1, mirror=True),
					   yaxis = dict(title = 'Y', range = [0, 1], showgrid=False, ticks='outside', linewidth=1, mirror=True),
					   showlegend = True,
					   width = 750,
					   height = 700,
					   autosize = True,
					   template = "plotly_white",
					   legend = dict(x = 1, xanchor = 'right', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
					   # legend = dict(x = 0, xanchor = 'left', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
					   )

	data = [trace_points, trace_depo] + path_traces
	fig = go.Figure(data = data, layout = layout)
	fig.show()

def opt2(route, dist): 
	size = len(route)
	improved = True
	while improved:
		improved = False
		for i in range(size - 2):
			i1 = i + 1
			a = route[i]
			b = route[i1]
			for j in range(i + 2, size):
				j1 = j + 1
				if j == size - 1:
					j1 = 0

				c = route[j]
				d = route[j1]
				if i == 0 and j1 == 0: continue# if i == j1
				if(dist[a][c] + dist[b][d] < dist[a][b] + dist[c][d]):
					""" i i+1 j j+1
						swap(i+1, j)
					"""
					tmp = route[i1:j1]
					route[i1:j1] = tmp[::-1]# tmp in inverse order
					improved = True 
	return route

def get_sum_dist(route, dist):
	return sum([dist[route[i]][route[i+1]] for i in range(len(route)-1)])

def improve_opt2(pi, data, idx_in_batch):
	from dist_matrix import get_dist_matrix
	depot_xy = data[0][idx_in_batch].cpu().numpy()
	customer_xy = data[1][idx_in_batch].cpu().numpy()
	demands = data[2][idx_in_batch].cpu().numpy()
	xy = np.concatenate([depot_xy.reshape(1, 2), customer_xy], axis = 0)
	
	dist = get_dist_matrix(xy, digit = 4)
	pi_ = list(pi[idx_in_batch].cpu().numpy())
	pi_.insert(0, 0)
	pi_.append(0)
	# print(pi_)
	
	routes = []
	last = 0
	for i in range(1, len(pi_)):
		if pi_[i] == 0:
			tmp = pi_[last:i+1]
			last = i+1
			if tmp[0] != 0: tmp.insert(0, 0)
			routes.append(tmp)
	
	# print(routes)
	new_route = []
	for route in routes:# apply 2-opt to each route
		tmp = opt2(route, dist)
		new_route.extend(tmp)
	# print(new_route)
	new_route = get_clean_path(new_route)
	# print(new_route)
	new_cost = get_sum_dist(new_route, dist)
	return new_route, new_cost

if __name__ == '__main__':
	args = test_parser()
	t1 = time()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	pretrained = load_model(args.path, embed_dim = 128, n_customer = args.n_customer, n_encode_layers = 3)
	print(f'model loading time:{time()-t1}s')
	if args.txt is not None:
		datatxt = data_from_txt(args.txt)
		data = []
		for i in range(3):
			elem = [datatxt[i].squeeze(0) for j in range(args.batch)]
			data.append(torch.stack(elem, 0))
		data = list(map(lambda x: x.to(device), data))
	else:
		data = []
		for i in range(3):
			elem = [generate_data(device, 1, args.n_customer, args.seed)[i].squeeze(0) for j in range(args.batch)]
			data.append(torch.stack(elem, 0))
	print(f'data generate time:{time()-t1}s')
	pretrained = pretrained.to(device)
	pretrained.eval()
	with torch.no_grad():
		costs, _, pi = pretrained(data, return_pi = True, decode_type = args.decode_type)
	print('costs:', costs)
	idx_in_batch = torch.argmin(costs, dim = 0)
	print(f'decode type:{args.decode_type}\nminimum cost: {costs[idx_in_batch]:.3f} and idx: {idx_in_batch} out of {args.batch} solutions')
	print(f'{pi[idx_in_batch]}\ninference time: {time()-t1}s')
	plot_route(data, pi, costs, 'Pretrained', idx_in_batch)
	
	t1 = time()
	pi, costs = improve_opt2(pi, data, idx_in_batch)
	print(f'{pi}\nimproved time: {time()-t1}s')
	plot_route(data, pi, costs, '2OPT', idx_in_batch, is_tensor = False)

	
