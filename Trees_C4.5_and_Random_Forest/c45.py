from tree import *


def c45(data, max_levels):
    """Predict the labeling for a point.

    Args:
        data (list): a list of data points. Each dp is def class with /
                     attributes: label(str), values(list)
        max_levels (int): depth of a tree

    Returns:
        class Tree: class def in tree.py learned tree.

    """
    if max_levels <= 0:  # the maximum level depth is reached
        return make_leaf(data)
    feature, threshold = find_best_split(data)
    if threshold is None:  # there is no split that gains information
        return make_leaf(data)
    new_tree = Tree()
    new_tree.leaf = False
    new_tree.feature, new_tree.threshold = feature, threshold
    data_left, data_right = split_data(data, new_tree.feature, new_tree.threshold)
    new_tree.left = c45(data_left, max_levels - 1)
    new_tree.right = c45(data_right, max_levels - 1)
    return new_tree


if __name__ == "__main__":
    # load dummy data
    from data import get_train_data, get_test_data

    train = get_train_data()
    test = get_test_data()
    TREE_DEPTH = 9
    tree = c45(train, TREE_DEPTH)
    predictions = [predict(tree, point) for point in test]
    acc = accuracy(test, predictions)
    print(acc)
