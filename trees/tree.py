from math import log
import random


class Tree:
    """Tree node

    Args:
        leaf (bool): if the tree node is a leaf node
        prediction (dict): {label1: prob(label1), label2: prob(label2)}
        feature (int): data feature index
        threshold (float): threshhold value of the feature
        left (list): datapoints of class Point, def in data.py, left of threshold
        right (list): datapoints of class Point, def in data.py, right of threshold
        gain (float): info gain

    """

    leaf = True
    prediction = None
    feature = None
    threshold = None
    left = None
    right = None
    gain = None


def most_likely_class(prediction):
    """Return label with highest probability

    Args:
        prediction (dict): {label1: prob(label1), label2: prob(label2)}

    Returns:
        str: label with the highest probability

    """
    labels = list(prediction.keys())
    probs = list(prediction.values())
    return labels[probs.index(max(probs))]


def accuracy(data, predictions):
    """Calculate accuracy of the predictions

    Args:
        data (list): a list of data points. Each dp is def class in data.py
                     with attributes: label(str), values(list)
        predictions (list): ls of dict {label: probability}

    Returns:
        float: accuracy of predictions (0 to 1)

    """
    total = 0
    correct = 0
    for i in range(len(data)):
        point = data[i]
        pred = predictions[i]
        total += 1
        guess = most_likely_class(pred)
        if guess == point.label:
            correct += 1
    return correct / total


def split_data(data, feature, threshold):
    """Split data on the feature and the threshhold

    Args:
        data (list): a list of data points. Each dp is def class in data.py
                     with attributes: label(str), values(list)
        feature (int): data feature index
        threshold (float): threshhold value of the feature

    Returns:
        tuple: 2 lists of data: left and right of the threshold

    """
    left = []
    right = []
    for point in data:
        feature_val = point.values[feature]
        if feature_val < threshold:
            left.append(point)
        else:
            right.append(point)
    return left, right


def count_labels(data):
    """Count labels in the data.

    Args:
        data (list): a list of data points. Each dp is def class in data.py
                     with attributes: label(str), values(list)

    Returns:
        dict: {label: occurance}

    """
    counts = {}
    for point in data:
        label = point.label
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts


def counts_to_entropy(counts):
    """Calculate entropy from label counts in the data.

    Args:
        counts (dict): {label: occurance}

    Returns:
        float: entropy

    """
    entropy = 0.0
    total = sum(counts.values())
    for val in counts.values():
        if val != 0:
            prob = val / total
            entropy -= prob * log(prob, 2)
    return entropy


def get_entropy(data):
    """Calculate entropy.

    Args:
        data (list): a list of data points. Each dp is def class in data.py
                     with attributes: label(str), values(list)

    Returns:
        float: entropy

    """
    counts = count_labels(data)
    entropy = counts_to_entropy(counts)
    return entropy


def find_best_threshold(data, feature):
    """Find threshold of the feature with a best gain

    Args:
        data (list): a list of data points. Each dp is def class in data.py
                     with attributes: label(str), values(list)
        feature (type): Description of parameter `feature`.

    Returns:
        tuple: (float) best_gain, (float) best_threshold

    """
    sorted_data = sorted(
        data, key=lambda k: k.values[feature])  # sort data by feature
    total_points = float(len(data))

    right_count = count_labels(sorted_data)
    left_count = {k: 0 for k in right_count}

    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    for i in range(len(sorted_data)):
        # preventing repeated vals by cheking the previous vals
        if (i > 0 and sorted_data[i - 1].values[feature] == sorted_data[i].values[feature]):
            pass
        else:
            left_tot = i
            right_tot = total_points - i
            curr = (counts_to_entropy(left_count) * left_tot
                    + counts_to_entropy(right_count) * right_tot) / total_points
            gain = entropy - curr
            if gain > best_gain:
                best_gain = gain
                best_threshold = sorted_data[i].values[feature]
        # counting labels left and right
        left_count[sorted_data[i].label] += 1
        right_count[sorted_data[i].label] -= 1
    return best_gain, best_threshold


def find_best_split(data):
    """
    Finds the best split for a tree according to information entropy.

    Args:
        data (list): a list of data points. Each dp is def class with
                     attributes: label(str), values(list)

    Returns (tuple): best_feature, best_threshold for a split

    """
    if len(data) < 2:
        return None, None
    best_feature = None
    best_threshold = None
    best_gain = 0
    total_features = len(data[0].values)
    for feature in range(total_features):
        gain, threshold = find_best_threshold(data, feature)
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_feature = feature
    return best_feature, best_threshold


def find_rand_split(data):
    """
    Does a random split for a tree on the given data

    Args:
        data (list): a list of data points. Each dp is def class
                     with attributes: label(str), values(list)

    Returns
        tuple: rand_feature, rand_threshold for a split

    """
    if len(data) < 2:
        return None, None
    rand_feature = random.randint(0, len(data[0].values) - 1)
    gain, rand_threshold = find_best_threshold(data, rand_feature)
    return rand_feature, rand_threshold


def make_leaf(data):
    """Make a leaf tree node

    Args:
        data (list): a list of data points. Each dp is def class with
                     attributes: label(str), values(list)

    Returns:
        class Tree: Description of returned object.

    """
    tree = Tree()
    counts = count_labels(data)
    tree.prediction = {label: counts[label] / len(data) for label in counts}
    return tree


def predict(tree, point):
    """Predict the label for the point given a tree
    Args:
        tree (Tree): learned tree
        point (Point class in data.py): data point with attributes:
                                        label(str), values(list)

    Returns
        dict: {label: probablilty}

    """
    if tree.leaf:
        return tree.prediction
    if point.values[tree.feature] < tree.threshold:
        return predict(tree.left, point)
    return predict(tree.right, point)
