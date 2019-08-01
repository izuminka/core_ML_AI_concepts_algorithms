from math import log
import random


class Tree:
    leaf = True
    prediction = None
    feature = None
    threshold = None
    left = None
    right = None
    gain = None


def most_likely_class(prediction):
    labels = list(prediction.keys())
    probs = list(prediction.values())
    return labels[probs.index(max(probs))]


def accuracy(data, predictions):
    total = 0
    correct = 0
    for i in range(len(data)):
        point = data[i]
        pred = predictions[i]
        total += 1
        guess = most_likely_class(pred)
        if guess == point.label:
            correct += 1
    return float(correct) / total


def split_data(data, feature, threshold):
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
    counts = {}
    for point in data:
        label = point.label
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    return counts


def counts_to_entropy(counts):
    entropy = 0.0
    total = float(sum(counts.values()))
    for val in counts.values():
        if val == 0:
            pass
        else:
            prob = val / total
            entropy -= prob * log(prob, 2)
        # print entropy
    return entropy


def get_entropy(data):
    counts = count_labels(data)
    entropy = counts_to_entropy(counts)
    return entropy

def entopy_track(count_dict, entropy_dict):
    count_tup = tuple(count_dict.items())
    if count_tup not in entropy_dict:
        entropy = counts_to_entropy(count_dict)
        entropy_dict[count_tup] = counts_to_entropy(count_dict)
    else:
        entropy = entropy_dict[count_tup]
    return entropy, entropy_dict


def find_best_threshold(data, feature):
    sorted_data = sorted(data, key=lambda k: k.values[feature])  # sort data by feature
    total_points = float(len(data))
    entropy_dict = {}

    right_count = count_labels(sorted_data)
    left_count = {k: 0 for k in right_count}

    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    for i in range(len(sorted_data)):
        # preventing repeated vals by cheking the previous vals
        if i > 0 and sorted_data[i - 1].values[feature] == sorted_data[i].values[feature]:
            pass
        else:
            left = sorted_data[:i]
            right = sorted_data[i:]

            left_tot = i
            right_tot = total_points - i
            curr = (counts_to_entropy(left_count) * left_tot + counts_to_entropy(
                right_count) * right_tot) / total_points
            gain = entropy - curr
            # print gain
            if gain > best_gain:
                best_gain = gain
                best_threshold = sorted_data[i].values[feature]
        # counting labels left and right
        left_count[sorted_data[i].label] += 1
        right_count[sorted_data[i].label] -= 1
    return best_gain, best_threshold


def find_best_split(data):
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
    if len(data) < 2:
        return None, None
    best_feature = random.randint(0, len(data[0].values) - 1)
    gain, best_threshold = find_best_threshold(data, best_feature)
    return best_feature, best_threshold


def make_leaf(data):
    tree = Tree()
    counts = count_labels(data)
    prediction = {}
    for label in counts:
        prediction[label] = float(counts[label]) / len(data)
    tree.prediction = prediction
    return tree


def predict(tree, point):
    if tree.leaf:
        return tree.prediction
    i = tree.feature
    if point.values[i] < tree.threshold:
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)
