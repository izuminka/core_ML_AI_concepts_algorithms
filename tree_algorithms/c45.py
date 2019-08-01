from tree import *

def c45(data, max_levels):
    if max_levels <= 0:  # the maximum level depth is reached
        return make_leaf(data)
    feature, threshold = find_best_split(data)
    if threshold == None:  # there is no split that gains information
        return make_leaf(data)
    else:
        tree = Tree()
        tree.leaf = False
        tree.feature, tree.threshold = feature, threshold
        data_left, data_right = split_data(data, tree.feature, tree.threshold)
        tree.left = c45(data_left, max_levels - 1)
        tree.right = c45(data_right, max_levels - 1)
        return tree