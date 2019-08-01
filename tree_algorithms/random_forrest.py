from tree import *

def rand_decision_tree(data, max_levels):
    """
    Create a descision tree with a random split. The algorithm is
    simalar to C4.5, except the best split function is substituted for
    random split function.

    Args:
        data:
        max_levels:

    Returns:

    """
    if max_levels <= 0:  # the maximum level depth is reached
        return make_leaf(data)

    feature, threshold = find_rand_split(data)
    if threshold == None:  # there is no split that gains information
        return make_leaf(data)
    else:
        tree = Tree()
        tree.leaf = False
        tree.feature, tree.threshold = feature, threshold
        data_left, data_right = split_data(data, tree.feature, tree.threshold)
        tree.left = rand_decision_tree(data_left, max_levels - 1)
        tree.right = rand_decision_tree(data_right, max_levels - 1)
        return tree