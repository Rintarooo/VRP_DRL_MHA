import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from model import AttentionModel
from data import generate_data, data_from_txt
from baseline import load_model
from config import file_parser

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

def plot_route(data, pi, title, cost, idx_in_batch = 0):
	"""Plots journey of agent
	Args:
		data: dataset of graphs
		pi: (batch, decode_step) # tour
		idx_in_batch: index of graph in data to be plotted
	"""

	# Remove extra zeros
	pi_ = get_clean_path(pi[idx_in_batch].numpy())

	depot_xy = data[0][idx_in_batch].numpy()
	customer_xy = data[1][idx_in_batch].numpy()
	demands = data[2][idx_in_batch].numpy()
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
	
	layout = go.Layout(title = dict(text = f'<b>VRP{customer_xy.shape[0]} {title}, Total Length = {cost.numpy():.3f}</b>', x = 0.5, y = 1, yanchor = 'bottom', yref = 'paper', pad = dict(b = 10)),#https://community.plotly.com/t/specify-title-position/13439/3
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
	print('path: ', pi_)
	fig = go.Figure(data = data, layout = layout)
	fig.show()

if __name__ == '__main__':
	model = AttentionModel()
	pretrained = load_model(file_parser().path)

	# dataset = generate_data(n_samples = 1, n_customer = 50, seed = 3) 
	# dataset = data_from_txt('./OpenData/A-n45-k7.txt')
	# for i, data in enumerate(dataset.repeat().batch(100)):
	
	dataset = generate_data(n_samples = 128, n_customer = 50, seed = 153) 
	for i, data in enumerate(dataset.batch(128)):
		cost, _, pi = pretrained(data, return_pi = True)
		idx_min = tf.argmin(cost, axis = 0)
		plot_route(data, pi, 'Pretrained', cost[idx_min], idx_min)
		cost, _, pi = model(data, return_pi = True)
		plot_route(data, pi, 'Untrained', cost[idx_min], idx_min)
		if i == 0:
			break