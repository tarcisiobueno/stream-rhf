"""Main module."""

from concurrent.futures import ThreadPoolExecutor
import sys
import time
from typing import List, Tuple
from scipy.stats import kurtosis
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

def get_kurtosis_feature_split(data: np.ndarray, r: float) -> Tuple[int, float]:
    """
    Get attribute split according to Kurtosis Split using NumPy arrays.
    
    Args:
        data: The dataset of the node (numpy array)
        r: Random value for split selection
    Returns:
        Tuple of (feature_index, feature_split)
    """
    # Handle constant columns
    var = np.var(data, axis=0)
    data = data.copy()
    data[:, var == 0] = np.nan
    
    # Calculate kurtosis values
    kurtosis_values = kurtosis(data, fisher=False, axis=0)
    kurtosis_values = np.nan_to_num(kurtosis_values, nan=0.0)
    kurtosis_values_log = np.log(kurtosis_values + 1)
    kurtosis_values_sum_log = kurtosis_values_log.sum()
    
    while True:
        r_adjusted = r * kurtosis_values_sum_log
        feature_index = np.digitize(r_adjusted, np.cumsum(kurtosis_values_log), right=True)
        min_ = np.min(data[:, feature_index])
        max_ = np.max(data[:, feature_index])
        feature_split = np.random.uniform(min_, max_)
        if min_ < feature_split < max_:
            break
            
    return feature_index, feature_split

def get_random_feature_split(data: np.ndarray) -> Tuple[int, float]:
    """
    Get attribute split according to Random Split using NumPy arrays.
    """
    choices = np.arange(data.shape[1])
    np.random.shuffle(choices)
    
    while len(choices) > 0:
        attribute = choices[-1]
        choices = choices[:-1]
        min_attribute = np.min(data[:, attribute])
        max_attribute = np.max(data[:, attribute])
        
        if min_attribute != max_attribute:
            while True:
                split_value = np.random.uniform(min_attribute, max_attribute)
                if min_attribute < split_value < max_attribute:
                    break
            return attribute, split_value
            
    return None, None

class Node:
    """Node object using NumPy arrays."""
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
        self.data_indices = None

class Root(Node):
    """Root node object."""
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.index = 0

class RandomHistogramTree:
    """Random Histogram Tree using NumPy arrays."""
    
    def __init__(self, z: np.ndarray, window_size: int, index: int, data: np.ndarray = None, 
                 max_height: int = None, split_criterion: str = 'kurtosis'):
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

    def generate_node(self, depth: int = None, parent: Node = None) -> Node:
        """Generates a new node."""
        self.N += 1
        node = Node()
        node.depth = depth
        node.index = self.N
        node.parent = parent
        return node

    def reset_leaves(self):
        """Reset leaves list by traversing tree."""
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

    def set_leaf(self, node: Node, data: np.ndarray, indices: np.ndarray):
        """Transform generic node into leaf."""
        node.type = 1
        node.size = data.shape[0]
        node.data_indices = indices
        node.data = data.copy()
        self.leaves.append(node)

    def build(self, node: Node, data: np.ndarray, indices: np.ndarray):
        """Recursively builds the tree."""
        node.data_indices = indices
        node.data = data

        if data.shape[0] == 0:
            self.error_node = node
            return
        if data.shape[0] <= 1:
            self.set_leaf(node, data, indices)
            return
        if np.all(np.all(data == data[0, :], axis=0)):
            self.set_leaf(node, data, indices)
            return
        if node.depth >= self.max_height:
            self.set_leaf(node, data, indices)
            return

        if self.split_criterion == 'kurtosis':
            attribute, value = get_kurtosis_feature_split(
                data, self.z[self.index, node.index % (self.z.shape[1]-1)])
        else:
            attribute, value = get_random_feature_split(data)

        node.left = self.generate_node(depth=node.depth+1, parent=node)
        node.right = self.generate_node(depth=node.depth+1, parent=node)
        
        node.attribute = attribute
        node.value = value
        
        mask = data[:, attribute] < value
        self.build(node.left, data[mask], indices[mask])
        self.build(node.right, data[~mask], indices[~mask])

    def build_tree(self, data: np.ndarray):
        """Build tree starting from root."""
        self.tree_ = Root()
        indices = np.arange(data.shape[0])
        self.build(self.tree_, data, indices)

class RHF:
    """Random Histogram Forest using NumPy arrays."""
    
    def __init__(self, z: np.ndarray, window_size: int, num_trees: int = 100, 
                 max_height: int = 5, split_criterion: str = 'kurtosis'):
        self.num_trees = num_trees
        self.max_height = max_height
        self.split_criterion = split_criterion
        self.z = z
        self.window_size = window_size
        self.scores = np.zeros(window_size)
        self.unique_instances = None

    def insert(self, tree: RandomHistogramTree, node: Node, tree_index: int, 
              xi: np.ndarray, xi_index: int):
        """Insert new instance into tree."""
        new_data = np.vstack([node.data, xi])
        new_indices = np.append(node.data_indices, xi_index)
        
        if node.type == 0:
            attribute, value = get_kurtosis_feature_split(
                new_data, self.z[tree_index, node.index % (self.z.shape[1]-1)])
            if attribute != node.attribute:
                tree.build(node, new_data, new_indices)
                return tree
            elif xi[0, node.attribute] <= node.value:
                self.insert(tree, node.left, tree_index, xi, xi_index)
            else:
                self.insert(tree, node.right, tree_index, xi, xi_index)
        else:
            tree.build(node, new_data, new_indices)
        return tree

    def compute_scores(self, instance_index: int) -> float:
        """Compute anomaly score for instance."""
        normalized_index = self.window_size - 1 + instance_index % self.window_size
        self.scores = np.append(self.scores, 0)
        
        for tree in self.forest:
            for leaf in tree.leaves:
                if instance_index in leaf.data_indices:
                    p = leaf.size / self.unique_instances
                    self.scores[normalized_index] += np.log(1/p)
                    break
                    
        return self.scores[normalized_index]

    def fit(self, data: np.ndarray, compute_scores: bool = True) -> np.ndarray:
        """Fit forest to data and compute initial scores."""
        self.unique_instances = data.shape[0]
        self.forest = []
        
        for tree_id in range(self.num_trees):
            tree = RandomHistogramTree(
                z=self.z,
                window_size=self.window_size,
                index=len(self.forest),
                data=data,
                max_height=self.max_height,
                split_criterion=self.split_criterion
            )
            
            self.forest.append(tree)
            
            if compute_scores:
                for leaf in tree.leaves:
                    p = leaf.size / self.unique_instances
                    self.scores[leaf.data_indices % self.window_size] += np.log(1/p)
                    
        return self.scores if compute_scores else None

    def process_tree(self, tree_data: Tuple) -> None:
        """
        Process a single tree in parallel.
        
        Args:
            tree_data: Tuple containing (tree, tree index, new instance, instance index)
        """
        tree, tree_i, new_instance, instance_idx = tree_data
        self.insert(tree, tree.tree_, tree.index, new_instance, instance_idx)
        tree.reset_leaves()    
    

    def parallel_process_trees(self, new_instance: np.ndarray, instance_idx: int, max_workers: int = None) -> None:
        """
        Process trees in parallel using ThreadPoolExecutor.
        
        Args:
            new_instance: New data instance to be inserted
            instance_idx: Index of the instance
            max_workers: Maximum number of parallel workers (default: None, uses CPU count)
        """
        # Create list of arguments for each tree
        tree_data = [(tree, i, new_instance, instance_idx) 
                     for i, tree in enumerate(self.forest)]
        
        # Process trees in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self.process_tree, tree_data))

def average_precision(labels: np.ndarray, output_scores: np.ndarray) -> float:
    """
    Compute Average Precision (AP) score using NumPy arrays.
    
    Args:
        labels: True binary labels (0 or 1)
        output_scores: Predicted scores
    Returns:
        Average Precision score
    """
    # Convert labels to boolean array
    labels = labels.astype(bool)
    
    # Sort by output_scores in descending order
    sorted_indices = np.argsort(output_scores)[::-1]
    sorted_labels = labels[sorted_indices]
    
    # Compute Precision and Recall at each threshold
    tp = np.cumsum(sorted_labels)  # True Positives
    fp = np.cumsum(~sorted_labels)  # False Positives
    n_positives = np.sum(labels)
    
    # Calculate precision and recall
    precision = tp / (tp + fp)
    recall = tp / n_positives
    
    # Compute AP using the area under the precision-recall curve
    # Add a zero recall point at the start
    recall = np.concatenate(([0.], recall))
    precision = np.concatenate(([1.], precision))
    
    # Compute the area under the PR curve using the trapezoidal rule
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    
    return ap

def main_parallel():
    # Load and prepare data
    dataset_name = "abalone"
    data = np.loadtxt(f"../data/public/{dataset_name}.gz", delimiter=",", skiprows=1)
    #data = data[:100]
    labels = data[:, -1]
    features = data[:, :-1]
    
    # Initialize parameters
    t = 100  # number of trees
    h = 5    # max height
    z = np.random.rand(t, 2**h - 1)
    n = 5    # percentage
    window_size = int(features.shape[0] * n/100)
    
    # Initialize windows
    reference_window = np.array([]).reshape(0, features.shape[1])
    current_window = np.array([]).reshape(0, features.shape[1])
    
    all_output_scores = []
    
    # Process stream
    for i in tqdm(range(features.shape[0]), desc="Processing data"):
        current_instance = features[i:i+1]
        current_window = np.vstack([current_window, current_instance]) if current_window.size else current_instance
        
        if current_window.shape[0] == window_size:
            reference_window = current_window.copy()
            current_window = np.array([]).reshape(0, features.shape[1])
            
        if (i+1) == window_size:
            my_rhf = RHF(num_trees=t, max_height=h, split_criterion='kurtosis', 
                        z=z, window_size=window_size)
            output_scores_l = my_rhf.fit(reference_window)
            all_output_scores.extend(output_scores_l)
            
        if (i+1) > window_size:
            new_instance = features[i:i+1]
            
            # Replace the sequential processing with parallel processing
            my_rhf.parallel_process_trees(new_instance, i, max_workers=10)
            
            my_rhf.unique_instances = reference_window.shape[0] + current_window.shape[0]
            all_output_scores.append(my_rhf.compute_scores(i))
            
            if i % window_size == 0:
                my_rhf = RHF(num_trees=t, max_height=h, split_criterion='kurtosis',
                        z=z, window_size=window_size)
                my_rhf.fit(reference_window, compute_scores=False)

def main():
    # Load and prepare data
    dataset_name = "abalone"
    data = np.loadtxt(f"../data/public/{dataset_name}.gz", delimiter=",", skiprows=1)
    #data = data[:100]
    labels = data[:, -1]
    features = data[:, :-1]
    
    # Initialize parameters
    t = 100  # number of trees
    h = 5    # max height
    z = np.random.rand(t, 2**h - 1)
    n = 5    # percentage
    window_size = int(features.shape[0] * n/100)
    
    # Initialize windows
    reference_window = np.array([]).reshape(0, features.shape[1])
    current_window = np.array([]).reshape(0, features.shape[1])
    
    all_output_scores = []
    
    # Process stream
    for i in tqdm(range(features.shape[0]), desc="Processing data"):
        current_instance = features[i:i+1]
        current_window = np.vstack([current_window, current_instance]) if current_window.size else current_instance
        
        if current_window.shape[0] == window_size:
            reference_window = current_window.copy()
            current_window = np.array([]).reshape(0, features.shape[1])
            
        if (i+1) == window_size:
            my_rhf = RHF(num_trees=t, max_height=h, split_criterion='kurtosis', 
                        z=z, window_size=window_size)
            output_scores_l = my_rhf.fit(reference_window)
            all_output_scores.extend(output_scores_l)
            
        if (i+1) > window_size:
            new_instance = features[i:i+1]
            
            for tree_i, tree in enumerate(my_rhf.forest):
                my_rhf.insert(tree, tree.tree_, tree.index, new_instance, i)
                tree.reset_leaves()
                
            my_rhf.unique_instances = reference_window.shape[0] + current_window.shape[0]
            all_output_scores.append(my_rhf.compute_scores(i))
            
            if i % window_size == 0:
                my_rhf = RHF(num_trees=t, max_height=h, split_criterion='kurtosis',
                           z=z, window_size=window_size)
                my_rhf.fit(reference_window, compute_scores=False)
    
    # Compute final metric
    print(f"Number of scores: {len(all_output_scores)}")
    average_precision_score = average_precision(labels, np.array(all_output_scores))
    print(f"Average Precision Score: {average_precision_score}")

if __name__ == "__main__":
    main_parallel()