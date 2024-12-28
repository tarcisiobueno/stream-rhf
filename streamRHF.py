"""Main module."""

from concurrent.futures import ThreadPoolExecutor
import sys
import time
from typing import List, Tuple
from scipy.stats import kurtosis
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm


def get_kurtosis_feature_split(data, r):
	"""
	Get attribute split according to Kurtosis Split

	:param data: the dataset of the node
	:returns: 
									- feature_index: the attribute index to split
									- feature_split: the attribute value to split
	"""
	data = data.copy()
	constant_columns = data.columns[data.max() == data.min()]
	data.loc[:, constant_columns] = np.nan
	# if a column has same values, add Nan
	kurtosis_values = kurtosis(data, fisher=False)

	# MODIFIED
	# Some values are nan, for now, we set them to 0.0
	kurtosis_values = np.nan_to_num(kurtosis_values, nan=0.0)

	kurtosis_values_log = np.log(kurtosis_values+1)

	kurtosis_values_sum_log = kurtosis_values_log.sum()

	while True:
		# random_value_feature = np.random.uniform(0, kurtosis_values_sum_log)
		# MODIFIED
		r_adjusted = r * kurtosis_values_sum_log
		feature_index = np.digitize(
			r_adjusted, np.cumsum(kurtosis_values_log), right=True)
		min_ = np.min(data.iloc[:, feature_index])
		max_ = np.max(data.iloc[:, feature_index])
		feature_split = np.random.uniform(min_, max_)
		if min_ < feature_split < max_:
			break

	return feature_index, feature_split


def get_random_feature_split(data):
	"""
	Get attribute split according to Random Split

	:param data: the dataset of the node
	:returns: 
									- feature_index: the attribute index to split
									- feature_split: the attribute value to split
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

	:param int max_height: max height of the tree
	:param bool split_criterion: split criterion to use: 'kurtosis' or 'random'
	"""

	def __init__(self, z, window_size, index, data=None, max_height=None, split_criterion='kurtosis'):
		super(RandomHistogramTree, self).__init__()
		self.N = 0
		self.leaves = []
		self.max_height = max_height
		self.split_criterion = split_criterion
		self.index = index
		self.z = z
		self.window_size = window_size

		if data is not None:
			self.build_tree(data)
		else:
			sys.exit('Error data')

	def generate_node(self, depth=None, parent=None):
		"""
		Generates a new new

		:param int depth: depth of the node
		:param Node parent: parent node
		"""
		self.N += 1

		node = Node()
		node.depth = depth
		node.index = self.N
		node.parent = parent

		return node

	def reset_leaves(self):
		# traverse tree and get all leaves
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

	def set_leaf(self, node, data):
		"""
		Transforms generic node into leaf

		:param node: generic node to transform into leaf
		:param data: node data used to define node size and data indexes corresponding to node 
		"""
		node.type = 1
		node.size = data.shape[0]
		node.data_index = data.index
		node.data = data.copy()
		self.leaves.append(node)

	def build(self, node, data):
		"""
		Function which recursively builds the tree

		:param node: current node
		:param data: data corresponding to current node
		"""
		node.data_index = data.index
		node.data = data

		node_tree_i = self.index

		if data.shape[0] == 0:
			self.error_node = node
		if data.shape[0] <= 1:
			self.set_leaf(node, data)
			return
		if data.duplicated().sum() == data.shape[0] - 1:
			self.set_leaf(node, data)
			return
		if node.depth >= self.max_height:
			self.set_leaf(node, data)
			return

		if self.split_criterion == 'kurtosis':
			attribute, value = get_kurtosis_feature_split(
				data, self.z[node_tree_i, node.index % (self.z.shape[1]-1)])
		elif self.split_criterion == 'random':
			attribute, value = get_random_feature_split(data)
		else:
			sys.exit('Error: Unknown split criterion')

		node.left = self.generate_node(depth=node.depth+1, parent=node)
		node.right = self.generate_node(depth=node.depth+1, parent=node)

		node.attribute = attribute
		node.value = value

		self.build(node.left, data[data.iloc[:, attribute] < value])
		self.build(node.right, data[data.iloc[:, attribute] >= value])

	def build_tree(self, data):
		"""
		Build tree function: generates the root node and successively builds the tree recursively

		:param data: the dataset
		"""
		self.tree_ = Root()
		self.build(self.tree_, data)

	def get_data_indexes(self):
		"""
		Returns the data indexes of the leaves.

		:returns: A set of data indexes of the leaves.
		"""
		return set([index for leaf in self.leaves for index in leaf.data_index.tolist()])

	# def get_leaves(self, node, leaves):

	# 	if node.type == 1:
	# 		leaves.append(node)
	# 		return

	# 	self.get_leaves(node.left, leaves)
	# 	self.get_leaves(node.right, leaves)


class RHF(object):
	"""
	Random Histogram Forest. Builds and ensemble of Random Histogram Trees

	:param int num_trees: number of trees
	:param int max_height: maximum height of each tree
	:param str split_criterion: split criterion to use - 'kurtosis' or 'random'
	:param bool check_duplicates: check duplicates in each leaf
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

	def insert(self, tree, node, tree_index, xi):
		new_data = pd.concat([node.data, xi])
		# loop trough all trees of the forest
		if node.type == 0:
			attribute, value = get_kurtosis_feature_split(
				new_data, self.z[tree_index, node.index % (self.z.shape[1]-1)])
			if attribute != node.attribute:
				tree.build(node, new_data)
				return tree
			elif xi.iloc[0, node.attribute] <= node.value:
				self.insert(tree, node.left, tree_index, xi)
			else:
				self.insert(tree, node.right, tree_index, xi)
		else:
			tree.build(node, new_data)
		return tree

	def compute_scores(self, instance_index):
     
		normalized_index = self.window_size - 1 + instance_index % self.window_size
		self.scores = np.append(self.scores, 0)

		for tree in self.forest:
			if self.has_duplicates:
				for leaf in tree.leaves:
					samples_indexes = leaf.data_index
					if instance_index in samples_indexes:
						p = self.data_hash[samples_indexes].nunique(
						)/self.uniques_
						self.scores[normalized_index] += np.log(1/(p))
						break
			else:
				for leaf in tree.leaves:
					samples_indexes = leaf.data_index
					if instance_index in samples_indexes:
						p = leaf.size/self.uniques_
						self.scores[normalized_index] += np.log(1/(p))
						break
		return self.scores[normalized_index]

	def fit(self, data, compute_scores=True):
		"""
		Fit function: builds the ensemble and returns the scores

		:param data: the dataset to fit
		:return scores: anomaly scores
		"""

		data = pd.DataFrame(data)

		self.check_hash(data)

		self.forest = []

		for tree_id in range(self.num_trees):

			randomHistogramTree = RandomHistogramTree(
				z=self.z,
				window_size=window_size,
				index=len(self.forest),
				data=data,
				max_height=self.max_height,
				split_criterion=self.split_criterion
			)

			# MODIFIED

			self.forest.append(randomHistogramTree)

			if compute_scores:
				if self.has_duplicates:
					for leaf in randomHistogramTree.leaves:
						samples_indexes = leaf.data_index
						p = self.data_hash[samples_indexes].nunique()/self.uniques_
						self.scores[samples_indexes %
									self.window_size] += np.log(1/(p))

				else:
					for leaf in randomHistogramTree.leaves:
						samples_indexes = leaf.data_index
						p = leaf.size/self.uniques_
						self.scores[samples_indexes %
									self.window_size] += np.log(1/(p))
				return self.scores

	def check_hash(self, data):
		"""
		Checks if there are duplicates in the dataset

		:param data: dataset
		"""
		if self.check_duplicates:
			if data.duplicated().sum() > 0:
				self.has_duplicates = True
				self.get_hash(data)
				self.uniques_ = self.data_hash.nunique()
			else:
				self.uniques_ = data.shape[0]
		else:
			self.uniques_ = data.shape[0]

	def get_hash(self, data):
		"""
		Builds hash of data for duplicates identification

		:param data: dataset
		"""
		self.data_hash = data.apply(lambda row: hash(
			'-'.join([str(x) for x in row])), axis=1)

	def get_data_indexes(self):
		"""
		Returns the data indexes of the leaves.

		:returns: A set of data indexes of the leaves from all trees in the forest.
		"""
		return set([index for tree in self.forest for index in tree.get_data_indexes()])

	def get_number_of_nodes_with_one_instance(self):
		"""
		Returns the number of nodes with only one instance.

		:returns: The number of nodes with only one instance.
		"""
		return sum([1 for tree in self.forest for leaf in tree.leaves if leaf.size == 1])


def average_precision(labels, output_scores):
	"""
	Compute Average Precision (AP) score.

	:param labels: True binary labels (0 or 1) as a pandas Series or numpy array.
	:param output_scores: Predicted scores as a pandas Series or numpy array.
	:returns: The Average Precision (AP) score.
	"""
	# Ensure labels and output_scores are numpy arrays
	labels = np.array(labels)
	output_scores = np.array(output_scores)

	# Sort by output_scores in descending order
	sorted_indices = np.argsort(output_scores)[::-1]
	sorted_labels = labels[sorted_indices]
	sorted_scores = output_scores[sorted_indices]

	# Compute Precision and Recall at each threshold
	tp = 0  # True Positives
	fp = 0  # False Positives
	n_positives = np.sum(labels)

	precisions = []
	recalls = []

	for i in range(len(sorted_labels)):
		if sorted_labels[i] == 1:
			tp += 1
		else:
			fp += 1

		precision = tp / (tp + fp)
		recall = tp / n_positives

		precisions.append(precision)
		recalls.append(recall)

	# Compute AP as the weighted mean of precisions at each recall step
	ap = 0.0
	previous_recall = 0.0

	for p, r in zip(precisions, recalls):
		ap += p * (r - previous_recall)
		previous_recall = r

	return ap


if __name__ == "__main__":

	dataset_name = "abalone"
	df = pd.read_csv(f"../data/public/{dataset_name}.gz")
	#df = df.iloc[:400]
	labels = df['label']
	data = df.drop('label', axis=1)

	# maxtrix z E R ^ ( t x 2**h-1 ) (Number of nodes)
	t = 100
	h = 5
	z = np.random.rand(t, 2**h - 1)
	n = 5  # percentage
	window_size = int(data.shape[0]*n/100)
 
 	# create reference_window and current_window

	reference_window = pd.DataFrame()
	current_window = pd.DataFrame()

	AP_scores = []
	all_output_scores = []
 
	# Go through each instance to simulate a stream

	for i in tqdm(range(0, data.shape[0]), desc="Processing data"):
     
		current_window = pd.concat([current_window, data.iloc[i].to_frame().T])
  
		if current_window.shape[0] == window_size:
			reference_window = current_window.copy()
			current_window = pd.DataFrame()

		if (i+1) == window_size:

			my_rhf = RHF(num_trees=t, max_height=h,
						 split_criterion='kurtosis', z=z, window_size=window_size)
   
			output_scores_l = my_rhf.fit(reference_window)
			
			# saves output score given by the initial RHF
			all_output_scores.extend(output_scores_l)

		if (i+1) > window_size:
			
			new_instance, new_instance_index = data.iloc[i].to_frame().T, data.iloc[i].to_frame().T.index[0]
			# with list comprehension
			# [my_rhf.insert(tree, tree.tree_, tree.index, new_instance) for tree_i, tree in enumerate(my_rhf.forest)]
			for tree_i, tree in enumerate(my_rhf.forest):
				my_rhf.insert(tree, tree.tree_, tree.index, new_instance)
				# delete and add old and new leaves to the tree.leaves list
				tree.reset_leaves()
			# check for duplicates - this may affect the score computation
			my_rhf.check_hash(pd.concat([reference_window, current_window]))
			# compute score of the current instance
			all_output_scores.append(my_rhf.compute_scores(new_instance_index))
			if i % window_size == 0:
				my_rhf = RHF(num_trees=t, max_height=h,
						 split_criterion='kurtosis', z=z, window_size=window_size)
   
				my_rhf.fit(reference_window, compute_scores=False)				

	# compute final metric
	print(len(all_output_scores))
	all_output_scores = pd.Series(all_output_scores)
	average_precision_score = average_precision(labels, all_output_scores)
	print(average_precision_score)
