from multiprocessing import Pool
import os
import sys
import time
from typing import List, Tuple
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def get_row_hash(row, precision=5):
	# Ensure the row is a list of numeric values (e.g., float)
	return '|'.join(map(lambda x: f"{float(x):.{precision}f}", row.tolist()))

def kurtosis(data: np.ndarray) -> np.ndarray:
	"""
	Biased estimator for the kurtosis of a dataset.
 	"""
	n = data.shape[0]
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	
	is_constant = np.std(data, axis=0) == 0
	kurt = np.zeros(data.shape[1])

	non_constant_indices = ~is_constant
	centered = data[:, non_constant_indices] - mean[non_constant_indices]

	kurt[non_constant_indices] = (1/n)*np.sum(centered**4, axis=0) / (std[non_constant_indices])**4

	return kurt

def get_kurtosis_feature_split(data, r):
	"""
	Get attribute split according to Kurtosis Split
	"""
	
	kurtosis_values = kurtosis(data)
	kurtosis_values = kurtosis_values + 1
	kurtosis_values_log = np.log(kurtosis_values)
	kurtosis_values_sum_log = kurtosis_values_log.sum()
	np.random.seed(r)
	while True:			
		random_feature_split = np.random.uniform(0, kurtosis_values_sum_log)
		feature_index = np.digitize(random_feature_split, np.cumsum(kurtosis_values_log), right=True)
		min_ = np.min(data[:, feature_index])
		max_ = np.max(data[:, feature_index])
		feature_value = np.random.uniform(min_, max_)
		if min_ < feature_value < max_ or min_ == max_:
			break

	return feature_index, feature_value


def get_random_feature_split(data):
	"""
	Get attribute split according to Random Split
	"""
	choices = list(range(data.shape[1]))
	np.random.shuffle(choices)
	while len(choices) > 0:
		attribute = choices.pop()
		min_attribute = np.min(data[attribute])
		max_attribute = np.max(data[attribute])

		if min_attribute != max_attribute:
			while True:
				split_value = np.random.uniform(min_attribute, max_attribute)
				if min_attribute < split_value < max_attribute:
					break
			break

	return attribute, split_value


class Node(object):
	"""
	Node object
	"""
	def __init__(self):
		super(Node, self).__init__()

		self.left = None
		self.right = None

		self.split_value = None
		self.split_feature = None
		self.attribute = None

		self.data = None
		self.depth = None
		self.size = None
		self.index = None
		self.type = 0
		self.parent = None
		self.uniques_ = 0
		self.uniques_dict_ = {}
		self.has_duplicates = False
	
	def check_hash(self, data):
		"""
		Checks if there are duplicates in the dataset
		"""		
		if np.unique(data, axis=0).shape[0] < data.shape[0]:
			self.has_duplicates = True
			self.get_hash(data)
			self.uniques_ = len(self.uniques_dict_)
		else:
			self.uniques_ = data.shape[0]
	
	def get_hash(self, data):
		rows_as_strings = [get_row_hash(row) for row in data]
		for row in rows_as_strings:
			self.uniques_dict_[row] = self.uniques_dict_.get(row,0) + 1
	
	def get_hash_old(self, data):
		"""
		Builds hash of data for duplicates identification
		"""
		data_df = pd.DataFrame(data)
		self.data_hash = data_df.apply(lambda row: hash('-'.join([str(x) for x in row])), axis=1)	

class Root(Node):
	"""
	Node (Root) object
	"""
	def __init__(self):
		super().__init__()
		self.depth = 0
		self.index = 0


class RandomHistogramTree(object):
	"""
	Random Histogram Tree object
	"""

	def __init__(self, z, window_size, index, data=None, max_height=None, split_criterion='kurtosis', check_duplicates=True):
		super(RandomHistogramTree, self).__init__()
		self.N = 0
		self.leaves = []
		self.max_height = max_height
		self.split_criterion = split_criterion
		self.index = index
		self.z = z
		self.window_size = window_size
		self.check_duplicates = check_duplicates
		
		if data is not None:
			indices = np.arange(data.shape[0], dtype=int)
			self.build_tree(data, indices)
		else:
			sys.exit('Error data')

	def generate_node(self, depth=None, parent=None):
		"""
		Generates a new new
		"""
		self.N += 1
		node = Node()
		node.depth = depth
		node.index = self.N
		node.parent = parent

		return node

	def reset_leaves(self):		
		"""
		Resets the leaves of the tree
		"""
		def traverse(node):
			if node is None:
				return
			if node.type == 1:
				self.leaves.append(node)
			else:
				traverse(node.left)
				traverse(node.right)
		self.leaves = []
		traverse(self.tree_)
  
	def predict_score(self, node, instance, number_instances=None):
		"""
		Predicts the score of an instance
        """
		if node.type ==	1:
			if node.has_duplicates:					
				p = node.uniques_/number_instances
				return np.log(1/(p))
			else:				
				p = node.size/number_instances
				return np.log(1/(p))
		elif instance[0, node.attribute] < node.value:
			return self.predict_score(node.left, instance, number_instances)
		else:
			return self.predict_score(node.right, instance, number_instances)

	def set_leaf(self, node, data, indices: np.ndarray):
		"""
		Transforms generic node into leaf
		"""
		node.type = 1
		node.size = data.shape[0]
		node.data_indices = indices
		node.get_hash(data)
		node.data = data.copy()
		self.leaves.append(node)

	def build(self, node: Node, data: np.ndarray, indices: np.ndarray):
		"""
		Function which recursively builds the tree
		"""
		if data.shape[0] == 0:
			self.error_node = node
		if data.shape[0] <= 1:
			self.set_leaf(node, data, indices)
			return
		if np.unique(data, axis=0).shape[0] == 1:
			self.set_leaf(node, data, indices)
			return
		if node.depth >= self.max_height:
			self.set_leaf(node, data, indices)
			return

		if self.split_criterion == 'kurtosis':
			attribute, value = get_kurtosis_feature_split(
				data, self.z[self.index, node.index % (self.z.shape[1]-1)])
		else:
			sys.exit('Error: Unknown split criterion')

		mask = data[:, attribute] < value
		if data[mask].shape[0] == 0 or data[~mask].shape[0] == 0:
			self.set_leaf(node, data, None)
		else:
			node.left = self.generate_node(depth=node.depth+1, parent=node)
			node.right = self.generate_node(depth=node.depth+1, parent=node)
			node.attribute = attribute
			node.value = value

			if indices is not None:
				self.build(node.left, data[mask], indices[mask])
				self.build(node.right, data[~mask], indices[~mask])
			else:
				self.build(node.left, data[mask], None)
				self.build(node.right, data[~mask], None)

	def build_tree(self, data, indices):
		"""
		Build tree function: generates the root node and successively builds the tree recursively
		"""
		self.tree_ = Root()
		self.build(self.tree_, data, indices)

	def insert(self, node, new_data):
		if data.shape[0] == 0:
			warnings.warn('Data is empty')
			self.error_node = node
		if node.type == 0:
			attribute, value = get_kurtosis_feature_split(
				new_data, self.z[self.index, node.index % (self.z.shape[1]-1)])
			if attribute != node.attribute:
				self.build(node, new_data, indices = None)
				return
			elif new_data[-1][node.attribute] <= node.value:
				mask = new_data[:, node.attribute] < node.value
				self.insert(node.left, new_data[mask])
			else:
				mask = new_data[:, node.attribute] < node.value	
				self.insert(node.right, new_data[~mask])
		elif node.depth == self.max_height:
			self.set_leaf(node, new_data, None)
			return
		else:	
			self.build(node, new_data, indices = None)
			return

class RHF(object):
	"""
	Random Histogram Forest. Builds and ensemble of Random Histogram Trees
	"""
	def __init__(self, z, window_size, num_trees=100, max_height=5, split_criterion='kurtosis', check_duplicates=True):
		super(RHF, self).__init__()
		self.num_trees = num_trees
		self.max_height = max_height
		self.has_duplicates = False
		self.check_duplicates = check_duplicates
		self.split_criterion = split_criterion
		self.z = z
		self.window_size = window_size
		self.scores = np.zeros(window_size)
		self.number_instances = 0
		self.data = None

	def compute_instance_score(self, new_instance):
		"""
  		Computes the anomaly score of a new instance
        """
		instance_score = 0
		new_instance_hash = get_row_hash(new_instance[0])
		for tree in self.forest:
			for leaf in tree.leaves:
				if leaf.has_duplicates:
					if new_instance_hash in leaf.uniques_dict_:
						p = leaf.uniques_/self.number_instances
						instance_score += np.log(1/(p))				
				elif new_instance_hash in leaf.uniques_dict_:
					p = leaf.size/self.number_instances
					instance_score += np.log(1/(p))
		return instance_score

	def fit(self, data, compute_scores=True):
		"""
		Fit function: builds the ensemble and returns the scores
		"""
		self.number_instances = data.shape[0]
		self.data = data.copy()
		self.forest = []
		scores = np.zeros(data.shape[0])

		for tree_id in range(self.num_trees):
			randomHistogramTree = RandomHistogramTree(
				z=self.z,
				window_size=self.window_size,
				index=len(self.forest),
				data=data,
				max_height=self.max_height,
				split_criterion=self.split_criterion
			)

			self.forest.append(randomHistogramTree)

			if compute_scores:
				for leaf in randomHistogramTree.leaves:
					samples_indexes = leaf.data_indices
					if leaf.has_duplicates:
						p = leaf.uniques_/(self.window_size*2)
						scores[samples_indexes] += np.log(1/(p))
					else:
						p = leaf.size/(self.window_size*2)
						scores[samples_indexes] += np.log(1/(p))
		return scores


	def get_number_of_nodes_with_one_instance(self):
		"""
		Returns the number of nodes with only one instance.
		"""
		return sum([1 for tree in self.forest for leaf in tree.leaves if leaf.size == 1])


def main(data: str):

	t = 100
	h = 5
	z = np.zeros((t, 2**h - 1), dtype=int)
	for i in range(t):
		for j in range(2**h - 1):
			z[i, j] = np.random.randint(0, 2**32 - 1)
   
	w_size_percentage = 1  # percentage
	window_size = int(data.shape[0]*w_size_percentage/100)
	print(f"window_size: {window_size}")
	max_score_possible = t*np.log(2*window_size)
 
	labels = data[:, -1]
	data = data[:, :-1]
  
 	# create reference_window and current_window
	reference_window = np.array([]).reshape(0, data.shape[1])
	current_window = np.array([]).reshape(0, data.shape[1])

	scores = []
 
	# Go through each instance to simulate a stream
	for i in tqdm(range(0, data.shape[0]), desc="Processing data"):
	  
		current_instance = data[i:i+1]
		current_window = np.vstack([current_window, current_instance]) if current_window.size else current_instance
  
		if current_window.shape[0] == window_size:
			reference_window = current_window.copy()
			current_window = np.array([]).reshape(0, data.shape[1])

		if (i+1) == window_size:
			my_rhf = RHF(num_trees=t, max_height=h,
						 split_criterion='kurtosis', z=z, window_size=window_size)
   
			initial_rhf_scores = my_rhf.fit(reference_window)
			# normalize the score
			scores.extend(initial_rhf_scores/max_score_possible)

		if (i+1) > window_size:			
			new_instance = data[i:i+1]
			my_rhf.number_instances += 1
			new_data = np.vstack([my_rhf.data, new_instance])
			for tree in my_rhf.forest:				
				tree.insert(tree.tree_, new_data)
				tree.reset_leaves()
	
			predicted_score = 0
			for tree in my_rhf.forest:	
				predicted_score += tree.predict_score(tree.tree_, new_instance, my_rhf.window_size*2)
			# normalize the score			
			scores.append(predicted_score/max_score_possible)
   
			if i % window_size == 0:
				my_rhf = RHF(num_trees=t, max_height=h,
						 split_criterion='kurtosis', z=z, window_size=window_size)
   
				my_rhf.fit(reference_window, compute_scores=False)		
	# compute final metric
	av_precision = average_precision_score(labels, scores)
	return av_precision

def process_dataset_iteration(args):
	dataset_name, data, iteration = args
 
	tic = time.time()
	av_precision = main(data)
	toc = time.time()
	with open(f"./resultsStreamRHF_6/{dataset_name}.csv", "a") as f:
		f.write(f"{iteration},{av_precision},{toc - tic}\n")
		print(f"Completed {dataset_name} iteration {iteration} with score {av_precision}")
	return dataset_name, iteration, av_precision, toc - tic

if __name__ == "__main__":
	datasets = sys.argv[1:]  # Get dataset names from command line arguments
	parallel = True
	shuffle = True
	if parallel:
		for dataset_name in datasets:
			data = np.loadtxt(f"../data/public/{dataset_name}.gz", delimiter=",", skiprows=1)
			tasks = []
			for i in range(15):
				if shuffle:
					np.random.shuffle(data)
				tasks.append((dataset_name, data.copy(), i))
			# Get number of CPU cores
			num_cores = os.cpu_count()		
			# Create process pool and run tasks
			with Pool(processes=num_cores) as pool:
				results = pool.map(process_dataset_iteration, tasks)			
			# Process results
			for dataset_name, iteration, score, time_elapsed in results:
				if score is not None:
					print(f"Completed {dataset_name} iteration {iteration} with score {score}")
				else:
					print(f"Failed to process {dataset_name} iteration {iteration}")
	else:
		for i, dataset_name in enumerate(datasets):
			data = np.loadtxt(f"../data/public/{dataset_name}.gz", delimiter=",", skiprows=1)
			if shuffle:
				np.random.shuffle(data)
			dataset_name, iteration, av_precision, time_elapsed = process_dataset_iteration((dataset_name, data, i))
			print(f"Completed {dataset_name} iteration {iteration} with score {av_precision} in {time_elapsed} seconds")




