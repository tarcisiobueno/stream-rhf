
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from typing import List
from scipy.stats import kurtosis
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# Libraries for profiling
import cProfile
import pstats


def get_kurtosis_feature_split(data: np.ndarray, r, kurtosis_cache=None):
    """
Get attribute split according to Kurtosis Split.

:param data: 2D NumPy array of the dataset of the node.
:param r: Random factor for selecting the split point.
:param kurtosis_cache: Optional precomputed kurtosis values (for caching).
:returns:
    - feature_index: The attribute index to split.
    - feature_split: The attribute value to split.
"""
    data = data.astype('float64')

    # Mask constant columns
    constant_columns = np.where(
        np.nanmax(data, axis=0) == np.nanmin(data, axis=0))[0]

    # if a column has same values, add Nan
    if constant_columns.size > 0:
        data[:, constant_columns] = np.nan

    # COmpute kurtosis values if not cached
    if kurtosis_cache is None:
        kurtosis_values = kurtosis(
            data, axis=0, fisher=False, nan_policy='omit')
        kurtosis_values = np.nan_to_num(kurtosis_values, nan=0.0)
    else:
        kurtosis_values = kurtosis_cache

    # Compute logarithmic values
    kurtosis_values_log = np.log(kurtosis_values+1)
    # Start with all features as valid
    valid_features = np.arange(data.shape[1])

    # Select feature and compute the split value
    while True:
        if valid_features.size == 0:
            raise ValueError('No valid feature available for splitting')

        kurtosis_values_sum_log = np.sum(kurtosis_values_log[valid_features])
        r_adjusted = r * kurtosis_values_sum_log
        feature_index = valid_features[
            np.digitize(r_adjusted, np.cumsum(
                kurtosis_values_log[valid_features]), right=True)
        ]
        min_ = np.nanmin(data[:, feature_index])
        max_ = np.nanmax(data[:, feature_index])

        if np.isnan(min_) or np.isnan(max_):
            print(
                f"Feature {feature_index} has NaN bounds. Removing from valid features")
            valid_features = valid_features[valid_features != feature_index]
            continue

        # print("min, max", (min_, max_))
        feature_split = np.random.uniform(min_, max_)
        if min_ < feature_split < max_:
            break

    return feature_index, feature_split


class TreeNode:
    """Flat array representation of a tree node."""

    def __init__(self):
        self.left = -1  # Index of the left child
        self.right = -1  # Index of the right child
        self.split_feature = None
        self.split_value = None
        self.depth = 0
        self.is_leaf = False
        self.size = 0
        self.data_indices = None  # Indices of data points in this node


class RandomHistogramTreeArray:
    """Array-based Random Histogram Tree."""

    def __init__(self, data, z, tree_index, max_height, split_criterion='kurtosis'):
        self.z = z
        self.tree_index = tree_index
        self.max_height = max_height
        self.split_criterion = split_criterion
        self.tree: List[TreeNode] = []  # Array of TreeNodes
        self.data = None
        self.data_len = 0

        if data is not None:
            self.build_node(data, node_index=0)
        else:
            sys.exit('Error data')

    def build_node(self, data, node_index=0):

        if node_index == 0:
            self.data = data
            self.data_len = data.shape[0]
            root = TreeNode()
            root.data_indices = np.arange(data.shape[0])
            self.tree.append(root)

        stack = [node_index]

        while stack:
            node_index = stack.pop()
            node = self.tree[node_index]

            if len(node.data_indices) <= 1 or node.depth >= self.max_height or np.all(data[1:] == data[:-1]):
                node.is_leaf = True
                node.size = len(node.data_indices)
                continue

            # Determine split criterion
            if self.split_criterion == 'kurtosis':
                attribute, value = get_kurtosis_feature_split(
                    self.data[node.data_indices],
                    self.z[self.tree_index, node_index % (self.z.shape[1] - 1)]
                )
            else:
                pass
                # attribute, value = get_random_feature_split(self.data[node.data_indices])

            node.split_feature = attribute
            node.split_value = value

            left_indices = node.data_indices[self.data[node.data_indices,
                                                       attribute] < value]
            right_indices = node.data_indices[self.data[node.data_indices,
                                                        attribute] >= value]

            left_node = TreeNode()
            left_node.data_indices = left_indices
            left_node.depth = node.depth + 1
            self.tree.append(left_node)

            right_node = TreeNode()
            right_node.data_indices = right_indices
            right_node.depth = node.depth + 1
            self.tree.append(right_node)

            node.left = len(self.tree) - 2
            node.right = len(self.tree) - 1

            stack.append(node.left)
            stack.append(node.right)

    def predict(self, x):
        node_index = 0
        while not self.tree[node_index].is_leaf:
            node = self.tree[node_index]
            if x[node.split_feature] < node.split_value:
                node_index = node.left
            else:
                node_index = node.right

        node = self.tree[node_index]
        p = len(np.unique(node.data_indices)) / self.data_len
        return np.log(1 / p)


class RHF:
    """
    Random Histogram Forest. Builds an ensemble of Random Histogram Trees using an array-based structure.

    :param int num_trees: Number of trees in the forest.
    :param int max_height: Maximum height of each tree.
    :param str split_criterion: Split criterion to use - 'kurtosis' or 'random'.
    :param np.ndarray z: Precomputed random matrix for split points.
    :param int window_size: Size of the sliding window for the forest.
    """

    def __init__(self, z, window_size, num_trees=100, max_height=5, split_criterion='kurtosis'):
        self.num_trees = num_trees
        self.max_height = max_height
        self.split_criterion = split_criterion
        self.z = z
        self.window_size = window_size
        self.forest = []  # List to store the array-based trees

    def fit(self, data, compute_scores=True):
        """
        Fit function: builds the ensemble and returns the anomaly scores.

        :param data: The dataset to fit.
        :param bool compute_scores: Whether to compute scores during the fit.
        :return: Anomaly scores for the input data.
        """
        self.forest = []

        scores = np.zeros(data.shape[0])

        # Parallelize tree construction
        with ThreadPoolExecutor() as executor:
            futures = []
            for tree_id in range(self.num_trees):
                futures.append(executor.submit(
                    self._build_tree, tree_id, data))

            for future in futures:
                tree = future.result()
                self.forest.append(tree)

        if compute_scores:
            for i in range(data.shape[0]):
                scores[i] = sum(tree.predict(data[i])
                                for tree in self.forest) / self.num_trees

        return scores

    def _build_tree(self, tree_id, data):
        """
        Helper function to build an individual tree.
        :param tree_id: ID of the tree being built.
        :param data: The dataset for the tree.
        :return: The constructed RandomHistogramTreeArray.
        """
        tree = RandomHistogramTreeArray(
            data=data,
            max_height=self.max_height,
            split_criterion=self.split_criterion,
            z=self.z,
            tree_index=tree_id
        )
        return tree

    def insert(self, instance: np.ndarray, tree: RandomHistogramTreeArray, node_index=0):
        """
        Inserts a new data point into each tree in the forest.

        :param instance: New data point to be inserted (1D Numpy array).
        """
        # Update the count of data seen by the tree
        tree.data_len += 1

        # Concatenate new data with node data
        node = tree.tree[node_index]
        new_data = np.vstack((tree.data[node.data_indices], instance))

        if not node.is_leaf:
            attribute, _ = get_kurtosis_feature_split(
                new_data, self.z[tree.tree_index, node_index % (self.z.shape[1]-1)])
            if attribute != node.split_feature:
                tree.build_node(new_data, node_index)
                return tree
            elif instance[:, node.split_feature] <= node.split_value:
                self.insert(instance, tree, node_index=node.left)
            else:
                self.insert(instance, tree, node_index=node.right)
        else:
            tree.build_node(new_data, node_index)

    def predict(self, instance):
        """
        Predict the anomaly score for a single instance.

        :param instance: The data point to predict.
        :return: The anomaly score for the instance.
        """
        score = sum(tree.predict(instance)
                    for tree in self.forest) / self.num_trees
        return score


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


def insert_tree(tree, new_instance, rhf):
    """Function to insert new_instance into a specific tree."""
    rhf.insert(new_instance, tree, node_index=0)


if __name__ == "__main__":

    dataset_name = "abalone"
    df = pd.read_csv(f"../data/public/{dataset_name}.gz")
    labels = np.array(df['label'])
    data = np.array(df.drop('label', axis=1))

    # maxtrix z E R ^ ( t x 2**h-1 ) (Number of nodes)
    t = 100
    h = 5
    z = np.random.rand(t, 2**h - 1)
    n = 1  # percentage
    window_size = int(data.shape[0]*n/100)

    # create reference_window and current_window

    reference_window = np.empty((0, data.shape[1]))
    current_window = np.empty((0, data.shape[1]))

    AP_scores = []
    all_output_scores = []

    # Go through each instance to simulate a stream

    # ------ START PROFILING ------ #

    profiler = cProfile.Profile()
    profiler.enable()

    def insert_tree(args):
        tree, instance, rhf = args
        rhf.insert(instance, tree, node_index=0)

    for i in tqdm(range(0, data.shape[0]), desc="Processing data"):
        current_window = np.vstack((current_window, data[i].reshape(1, -1)))

        if current_window.shape[0] == window_size:
            reference_window = current_window.copy()
            current_window = np.empty((0, data.shape[1]))

        if (i+1) == window_size:
            my_rhf = RHF(
                num_trees=t,
                max_height=h,
                split_criterion='kurtosis',
                z=z,
                window_size=window_size
            )

            output_scores_l = my_rhf.fit(reference_window)
            # saves output score given by the initial RHF
            all_output_scores.extend(output_scores_l)

        if (i+1) > window_size:

            new_instance = data[i].reshape(1, -1)

            with ThreadPoolExecutor() as executor:
                executor.map(insert_tree, my_rhf.forest, [
                             new_instance] * len(my_rhf.forest), [my_rhf] * len(my_rhf.forest))

            # compute score of the current instance
            all_output_scores.append(my_rhf.predict(data[i]))

            if (i+1) % window_size == 0:
                my_rhf = RHF(
                    num_trees=t,
                    max_height=h,
                    split_criterion='kurtosis',
                    z=z,
                    window_size=window_size
                )

                my_rhf.fit(reference_window, compute_scores=False)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    # compute final metric
    print(len(all_output_scores))
    all_output_scores = pd.Series(all_output_scores)
    average_precision_score = average_precision(labels, all_output_scores)
    print(f"Average Precision Score: {average_precision_score}")