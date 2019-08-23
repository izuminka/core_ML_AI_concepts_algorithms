from tree import *

def rand_decision_tree(data, max_levels):
    """Create a descision tree with a random split. The algorithm is
    simalar to C4.5, except the best split function is substituted for
    random split function.

    Args:
        data (list): a list of data points. Each dp is def class with
                     attributes: label(str), values(list)
        max_levels (int): depth of each a tree

    Returns
        class Tree: learned tree

    """
    if max_levels <= 0:  # the maximum level depth is reached
        return make_leaf(data)

    feature, threshold = find_rand_split(data)
    if threshold is None:  # there is no split that gains information
        return make_leaf(data)
    tree = Tree()
    tree.leaf = False
    tree.feature, tree.threshold = feature, threshold
    data_left, data_right = split_data(data, tree.feature, tree.threshold)
    tree.left = rand_decision_tree(data_left, max_levels - 1)
    tree.right = rand_decision_tree(data_right, max_levels - 1)
    return tree


def dict_average_resuts_rand(point, forest):
    """Predict the labeling for a point with a random forest

    Args:
        point (defined class): data point with attributes: label(str), values(list)
        forest(list): a list of random forest desision trees

    Returns
        dict: {label1: prob(label1), label2: prob(label2)}
    """
    trees_pred = {}
    for i in range(len(forest)):
        pred_dict = predict(forest[i], point)
        print(pred_dict)
        for key in pred_dict.keys():
            if key not in trees_pred:
                trees_pred[key] = []
            else:
                trees_pred[key].append(pred_dict[key])
    return {k: sum(trees_pred[k]) / len(forest) for k in trees_pred}



if __name__ == '__main__':
    #load dummy data
    from data import get_train_data, get_test_data

    train = get_train_data()
    test = get_test_data()

    num_trees = 30
    depth = 30
    forest = [rand_decision_tree(train, depth) for i in range(num_trees)]
    predictions = [dict_average_resuts_rand(point, forest) for point in test]
    acc = accuracy(test, predictions)
    print(acc)
