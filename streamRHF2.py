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

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing


def get_kurtosis_feature_split(data, r):
    """
    Optimized version of kurtosis-based feature split
    """
    # Use numpy arrays instead of pandas for faster computation
    data_values = data.values
    # Calculate constant columns more efficiently
    constant_mask = (data_values.max(axis=0) == data_values.min(axis=0))
    data_values[:, constant_mask] = np.nan
    
    kurtosis_values = kurtosis(data_values, fisher=False)
    kurtosis_values = np.nan_to_num(kurtosis_values, nan=0.0)
    kurtosis_values_log = np.log(kurtosis_values + 1)
    kurtosis_values_sum_log = kurtosis_values_log.sum()
    
    r_adjusted = r * kurtosis_values_sum_log
    feature_index = np.digitize(r_adjusted, np.cumsum(kurtosis_values_log), right=True)
    
    feature_data = data_values[:, feature_index]
    min_ = np.min(feature_data)
    max_ = np.max(feature_data)
    
    while True:
        feature_split = np.random.uniform(min_, max_)
        if min_ < feature_split < max_:
            break
            
    return feature_index, feature_split


def get_random_feature_split(data):
    """
    Optimized version of random feature split
    """
    data_values = data.values
    n_features = data_values.shape[1]
    choices = np.random.permutation(n_features)
    
    for attribute in choices:
        min_attribute = np.min(data_values[:, attribute])
        max_attribute = np.max(data_values[:, attribute])
        
        if min_attribute != max_attribute:
            while True:
                split_value = np.random.uniform(min_attribute, max_attribute)
                if min_attribute < split_value < max_attribute:
                    return attribute, split_value
    
    return choices[0], np.random.uniform(min_attribute, max_attribute)


class Node:
    """Optimized Node class with slots for better memory usage"""
    __slots__ = ('left', 'right', 'split_value', 'split_feature', 'attribute',
                 'data', 'depth', 'size', 'index', 'type', 'parent', 'value',
                 'data_index')
    
    def __init__(self):
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
    """Optimized Root class"""
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.index = 0


class RandomHistogramTree:
    """Optimized Random Histogram Tree implementation"""
    
    def __init__(self, z, window_size, index, data=None, max_height=None, split_criterion='kurtosis'):
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
        self.N += 1
        node = Node()
        node.depth = depth
        node.index = self.N
        node.parent = parent
        return node
    
    def reset_leaves(self):
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
        node.type = 1
        node.size = len(data)
        node.data_index = data.index
        node.data = data
        self.leaves.append(node)
    
    def build(self, node, data):
        node.data_index = data.index
        node.data = data
        
        if len(data) == 0:
            self.error_node = node
            return
        if len(data) <= 1:
            self.set_leaf(node, data)
            return
        if data.duplicated().sum() == len(data) - 1:
            self.set_leaf(node, data)
            return
        if node.depth >= self.max_height:
            self.set_leaf(node, data)
            return
            
        node_tree_i = self.index
        
        if self.split_criterion == 'kurtosis':
            attribute, value = get_kurtosis_feature_split(
                data, self.z[node_tree_i, node.index % (self.z.shape[1]-1)])
        else:
            attribute, value = get_random_feature_split(data)
            
        node.left = self.generate_node(depth=node.depth+1, parent=node)
        node.right = self.generate_node(depth=node.depth+1, parent=node)
        
        node.attribute = attribute
        node.value = value
        
        # Use boolean indexing instead of iloc for better performance
        mask = data.iloc[:, attribute] < value
        self.build(node.left, data[mask])
        self.build(node.right, data[~mask])
    
    def build_tree(self, data):
        self.tree_ = Root()
        self.build(self.tree_, data)
    
    def get_data_indexes(self):
        return {idx for leaf in self.leaves for idx in leaf.data_index}


class RHF:
    """Optimized Random Histogram Forest implementation"""
    
    def __init__(self, z, window_size, num_trees=100, max_height=5, split_criterion='kurtosis', check_duplicates=True):
        self.num_trees = num_trees
        self.max_height = max_height
        self.has_duplicates = False
        self.check_duplicates = check_duplicates
        self.split_criterion = split_criterion
        self.z = z
        self.window_size = window_size
        self.scores = np.zeros(window_size)
    
    def insert(self, tree, node, tree_index, xi):
        # Use efficient concatenation
        new_data = pd.concat([node.data, xi], copy=False)
        
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
                    if instance_index in leaf.data_index:
                        p = self.data_hash[leaf.data_index].nunique() / self.uniques_
                        self.scores[normalized_index] += np.log(1/p)
                        break
            else:
                for leaf in tree.leaves:
                    if instance_index in leaf.data_index:
                        p = leaf.size / self.uniques_
                        self.scores[normalized_index] += np.log(1/p)
                        break
        return self.scores[normalized_index]
    
    def fit(self, data, compute_scores=True):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        self.check_hash(data)
        self.forest = []
        
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
                if self.has_duplicates:
                    for leaf in randomHistogramTree.leaves:
                        p = self.data_hash[leaf.data_index].nunique() / self.uniques_
                        self.scores[leaf.data_index % self.window_size] += np.log(1/p)
                else:
                    for leaf in randomHistogramTree.leaves:
                        p = leaf.size / self.uniques_
                        self.scores[leaf.data_index % self.window_size] += np.log(1/p)
                return self.scores
    
    def check_hash(self, data):
        if self.check_duplicates:
            duplicated_sum = data.duplicated().sum()
            if duplicated_sum > 0:
                self.has_duplicates = True
                self.get_hash(data)
                self.uniques_ = self.data_hash.nunique()
            else:
                self.uniques_ = len(data)
        else:
            self.uniques_ = len(data)
    
    def get_hash(self, data):
        # More efficient hashing
        self.data_hash = pd.util.hash_pandas_object(data)
    
    def get_data_indexes(self):
        return {idx for tree in self.forest for idx in tree.get_data_indexes()}
    
    def get_number_of_nodes_with_one_instance(self):
        return sum(1 for tree in self.forest for leaf in tree.leaves if leaf.size == 1)

def build_tree_parallel(args):
    """
    Parallel version of tree building
    """
    z, window_size, index, data, max_height, split_criterion = args
    return RandomHistogramTree(
        z=z,
        window_size=window_size,
        index=index,
        data=data,
        max_height=max_height,
        split_criterion=split_criterion
    )

class RHF_Parallel(RHF):
    """
    Parallel version of Random Histogram Forest
    """
    def __init__(self, z, window_size, num_trees=100, max_height=5, split_criterion='kurtosis', 
                 check_duplicates=True, n_jobs=None):
        super().__init__(z, window_size, num_trees, max_height, split_criterion, check_duplicates)
        self.n_jobs = n_jobs if n_jobs is not None else multiprocessing.cpu_count()

    def fit_parallel(self, data, compute_scores=True):
        """
        Parallel version of fit method
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        self.check_hash(data)
        self.forest = []
        
        # Prepare arguments for parallel processing
        args_list = [(self.z, self.window_size, i, data, self.max_height, self.split_criterion) 
                    for i in range(self.num_trees)]
        
        # Build trees in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            self.forest = list(executor.map(build_tree_parallel, args_list))
        
        if compute_scores:
            self.compute_scores_parallel()
            return self.scores
            
    def compute_scores_parallel(self):
        """
        Parallel version of score computation
        """
        def process_tree(tree):
            scores = np.zeros(self.window_size)
            if self.has_duplicates:
                for leaf in tree.leaves:
                    p = self.data_hash[leaf.data_index].nunique() / self.uniques_
                    scores[leaf.data_index % self.window_size] += np.log(1/p)
            else:
                for leaf in tree.leaves:
                    p = leaf.size / self.uniques_
                    scores[leaf.data_index % self.window_size] += np.log(1/p)
            return scores

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            all_scores = list(executor.map(process_tree, self.forest))
            
        self.scores = np.sum(all_scores, axis=0)

    def insert_parallel(self, new_instance, new_instance_index):
        """
        Parallel version of insert operation
        """
        def process_insert(args):
            tree_i, tree = args
            self.insert(tree, tree.tree_, tree.index, new_instance)
            tree.reset_leaves()
            return tree

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            self.forest = list(executor.map(process_insert, enumerate(self.forest)))


def main_parallel():
    """
    Parallel version of the main routine
    """
    dataset_name = "abalone"
    df = pd.read_csv(f"../data/public/{dataset_name}.gz")
    labels = df['label'].values
    data = df.drop('label', axis=1)
    
    t = 100
    h = 5
    z = np.random.rand(t, 2**h - 1)
    n = 1
    window_size = int(len(data) * n/100)
    
    reference_window = pd.DataFrame()
    current_window = pd.DataFrame()
    
    AP_scores = []
    all_output_scores = []
    
    # Use number of CPU cores for parallelization
    #n_jobs = multiprocessing.cpu_count()
    n_jobs = 4
    for i in tqdm(range(len(data)), desc="Processing data"):
        current_window = pd.concat([current_window, data.iloc[[i]]], copy=False)
        
        if len(current_window) == window_size:
            reference_window = current_window.copy()
            current_window = pd.DataFrame()
        
        if (i+1) == window_size:
            my_rhf = RHF_Parallel(num_trees=t, max_height=h,
                                split_criterion='kurtosis', z=z, 
                                window_size=window_size, n_jobs=n_jobs)
            output_scores_l = my_rhf.fit_parallel(reference_window)
            all_output_scores.extend(output_scores_l)
        
        if (i+1) > window_size:
            new_instance = data.iloc[[i]]
            new_instance_index = new_instance.index[0]
            
            # Parallel insert and update
            my_rhf.insert_parallel(new_instance, new_instance_index)
            
            # Check for duplicates
            my_rhf.check_hash(pd.concat([reference_window, current_window], copy=False))
            all_output_scores.append(my_rhf.compute_scores(new_instance_index))
            
            if i % window_size == 0:
                my_rhf = RHF_Parallel(num_trees=t, max_height=h,
                                    split_criterion='kurtosis', z=z, 
                                    window_size=window_size, n_jobs=n_jobs)
                my_rhf.fit_parallel(reference_window, compute_scores=False)
    
    all_output_scores = pd.Series(all_output_scores)
    print(len(all_output_scores))
    average_precision_score = average_precision(labels, all_output_scores)
    print(average_precision_score)

def main():
    dataset_name = "abalone"
    df = pd.read_csv(f"../data/public/{dataset_name}.gz")
    labels = df['label'].values  # Convert to numpy array immediately
    data = df.drop('label', axis=1)
    
    t = 100
    h = 5
    z = np.random.rand(t, 2**h - 1)
    n = 1
    window_size = int(len(data) * n/100)
    
    reference_window = pd.DataFrame()
    current_window = pd.DataFrame()
    
    AP_scores = []
    all_output_scores = []
    
    for i in tqdm(range(len(data)), desc="Processing data"):
        # Use more efficient concatenation
        current_window = pd.concat([current_window, data.iloc[[i]]], copy=False)
        
        if len(current_window) == window_size:
            reference_window = current_window.copy()
            current_window = pd.DataFrame()
        
        if (i+1) == window_size:
            my_rhf = RHF(num_trees=t, max_height=h,
                        split_criterion='kurtosis', z=z, window_size=window_size)
            output_scores_l = my_rhf.fit(reference_window)
            all_output_scores.extend(output_scores_l)
        
        if (i+1) > window_size:
            new_instance = data.iloc[[i]]
            new_instance_index = new_instance.index[0]
            
            # Update trees
            for tree_i, tree in enumerate(my_rhf.forest):
                my_rhf.insert(tree, tree.tree_, tree.index, new_instance)
                tree.reset_leaves()
            
            # Check for duplicates
            my_rhf.check_hash(pd.concat([reference_window, current_window], copy=False))
            all_output_scores.append(my_rhf.compute_scores(new_instance_index))
            
            if i % window_size == 0:
                my_rhf = RHF(num_trees=t, max_height=h,
                            split_criterion='kurtosis', z=z, window_size=window_size)
                my_rhf.fit(reference_window, compute_scores=False)
    
    # Final computations
    all_output_scores = pd.Series(all_output_scores)
    print(len(all_output_scores))
    average_precision_score = average_precision(labels, all_output_scores)
    print(average_precision_score)


def average_precision(labels, output_scores):
    """Optimized average precision calculation"""
    labels = np.asarray(labels)
    output_scores = np.asarray(output_scores)
    
    sorted_indices = np.argsort(output_scores)[::-1]
    labels = labels[sorted_indices]
    
    n_positives = np.sum(labels)
    if n_positives == 0:
        return 0.0
    
    tp_cumsum = np.cumsum(labels)
    precision = tp_cumsum / np.arange(1, len(labels) + 1)
    recall = tp_cumsum / n_positives
    
    # Compute AP using the trapezoidal rule
    return np.sum((recall[1:] - recall[:-1]) * precision[1:])


if __name__ == "__main__":
    main_parallel()