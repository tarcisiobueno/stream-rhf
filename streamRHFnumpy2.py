import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from tqdm import tqdm
import sys

def get_kurtosis_feature_split(data, r):
    """
    Get attribute split according to Kurtosis Split

    :param data: the dataset of the node
    :returns: 
                    - feature_index: the attribute index to split
                    - feature_split: the attribute value to split
    """
    data = np.copy(data)
    constant_columns = np.all(data == data[0, :], axis=0)  # Identify constant columns
    data[:, constant_columns] = np.nan  # Set constant columns to NaN

    kurtosis_values = kurtosis(data, fisher=False, axis=0)

    # Some values are nan, for now, we set them to 0.0
    kurtosis_values = np.nan_to_num(kurtosis_values, nan=0.0)

    kurtosis_values_log = np.log(kurtosis_values + 1)

    kurtosis_values_sum_log = np.sum(kurtosis_values_log)

    while True:
        r_adjusted = r * kurtosis_values_sum_log
        feature_index = np.digitize(r_adjusted, np.cumsum(kurtosis_values_log), right=True)
        feature_min = np.min(data[:, feature_index])
        feature_max = np.max(data[:, feature_index])
        feature_split = np.random.uniform(feature_min, feature_max)

        if feature_min < feature_split < feature_max:
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
    feature_index = np.random.choice(data.shape[1])
    feature_min = np.min(data[:, feature_index])
    feature_max = np.max(data[:, feature_index])

    if feature_min != feature_max:
        feature_split = np.random.uniform(feature_min, feature_max)
    else:
        feature_split = feature_min  # If the feature is constant, no split.

    return feature_index, feature_split


class Node:
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
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.index = 0


class RandomHistogramTree:
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
        node.size = data.shape[0]
        node.data_index = np.arange(data.shape[0])  # Use numpy indexing
        node.data = data
        self.leaves.append(node)

    def build(self, node, data):
        node.data_index = np.arange(data.shape[0])  # Use numpy indexing
        node.data = data

        if data.shape[0] <= 1 or node.depth >= self.max_height:
            self.set_leaf(node, data)
            return

        if self.split_criterion == 'kurtosis':
            attribute, value = get_kurtosis_feature_split(data, self.z[self.index, node.index % (self.z.shape[1] - 1)])
        elif self.split_criterion == 'random':
            attribute, value = get_random_feature_split(data)
        else:
            sys.exit('Error: Unknown split criterion')

        node.left = self.generate_node(depth=node.depth + 1, parent=node)
        node.right = self.generate_node(depth=node.depth + 1, parent=node)

        node.attribute = attribute
        node.value = value

        left_data = data[data[:, attribute] < value]
        right_data = data[data[:, attribute] >= value]
        self.build(node.left, left_data)
        self.build(node.right, right_data)

    def build_tree(self, data):
        self.tree_ = Root()
        self.build(self.tree_, data)


class RHF:
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
        new_data = np.vstack([node.data, xi])
        if node.type == 0:
            attribute, value = get_kurtosis_feature_split(new_data, self.z[tree_index, node.index % (self.z.shape[1] - 1)])
            if attribute != node.attribute:
                tree.build(node, new_data)
                return tree
            elif xi[0, node.attribute] <= node.value:
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
            for leaf in tree.leaves:
                samples_indexes = leaf.data_index
                if instance_index in samples_indexes:
                    p = leaf.size / self.uniques_
                    self.scores[normalized_index] += np.log(1 / p)
                    break
        return self.scores[normalized_index]

    def fit(self, data, compute_scores=True):
        data = np.array(data)

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
                for leaf in randomHistogramTree.leaves:
                    samples_indexes = leaf.data_index
                    p = leaf.size / self.uniques_
                    self.scores[samples_indexes % self.window_size] += np.log(1 / p)

        return self.scores

    def check_hash(self, data):
        if self.check_duplicates:
            if np.any(np.diff(np.sort(data, axis=0), axis=0) == 0):
                self.has_duplicates = True
                self.get_hash(data)
                self.uniques_ = len(np.unique(data, axis=0))
            else:
                self.uniques_ = data.shape[0]
        else:
            self.uniques_ = data.shape[0]

    def get_hash(self, data):
        self.data_hash = np.array([hash('-'.join(map(str, row))) for row in data])

# Function for average precision (AP) calculation
def average_precision(labels, output_scores):
    labels = np.array(labels)
    output_scores = np.array(output_scores)
    sorted_indices = np.argsort(output_scores)[::-1]
    sorted_labels = labels[sorted_indices]
    sorted_scores = output_scores[sorted_indices]

    tp = 0
    fp = 0
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

    ap = 0.0
    previous_recall = 0.0

    for p, r in zip(precisions, recalls):
        ap += p * (r - previous_recall)
        previous_recall = r

    return ap

if __name__ == "__main__":
    dataset_name = "abalone"
    df = pd.read_csv(f"../data/public/{dataset_name}.gz")
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data
    labels = df['label']
    data = df.drop('label', axis=1).to_numpy()  # Convert data to numpy

    t = 100
    h = 5
    z = np.random.rand(t, 2**h - 1)
    n = 1
    window_size = int(data.shape[0] * n / 100)

    reference_window = np.empty((0, data.shape[1]))
    current_window = np.empty((0, data.shape[1]))

    AP_scores = []
    all_output_scores = []

    for i in tqdm(range(data.shape[0])):
        current_window = np.vstack([current_window, data[i]])

        if current_window.shape[0] == window_size:
            reference_window = current_window.copy()
            current_window = np.empty((0, data.shape[1]))

        if (i + 1) == window_size:
            my_rhf = RHF(num_trees=t, max_height=h, split_criterion='kurtosis', z=z, window_size=window_size)
            output_scores_l = my_rhf.fit(reference_window)
            all_output_scores.extend(output_scores_l)

        if (i + 1) > window_size:
            new_instance = data[i].reshape(1, -1)
            for tree_i, tree in enumerate(my_rhf.forest):
                my_rhf.insert(tree, tree.tree_, tree.index, new_instance)
                tree.reset_leaves()

            my_rhf.check_hash(np.vstack([reference_window, current_window]))
            all_output_scores.append(my_rhf.compute_scores(i))

    all_output_scores = np.array(all_output_scores)
    average_precision_score = average_precision(labels, all_output_scores)
    print(average_precision_score)
