from math import log

class Tree:
    leaf = True

    prediction = None
    feature = None
    threshold = None
    left = None
    right = None
    #gain = None

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
    return (left, right)

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
        if val==0:
            pass
        else:
            prob = val/total
            entropy -= prob * log(prob, 2)
        #print entropy
    return entropy

def get_entropy(data):
    counts = count_labels(data)
    entropy = counts_to_entropy(counts)
    return entropy

# This is a correct but inefficient way to find the best threshold to maximize
# information gain.
def find_best_threshold(data, feature):
    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    for point in data:
        left, right = split_data(data, feature, point.values[feature])
        curr = (get_entropy(left)*len(left) + get_entropy(right)*len(right))/len(data)
        gain = entropy - curr
        if gain > best_gain:
            best_gain = gain
            best_threshold = point.values[feature]
    return (best_gain, best_threshold)

def entopy_track(count_dict, entropy_dict):
    count_tup = tuple(count_dict.items())
    if count_tup not in entropy_dict:
        entropy = counts_to_entropy(count_dict)
        entropy_dict[count_tup] = counts_to_entropy(count_dict)
    else:
        entropy = entropy_dict[count_tup]
    return entropy, entropy_dict

def find_best_threshold_fast(data, feature):
    sorted_data = sorted(data, key = lambda k: k.values[feature]) #sort data by feature
    total_points = float(len(data))
    entropy_dict = {}

    right_count = count_labels(sorted_data)
    left_count = {k:0 for k in right_count}

    entropy = get_entropy(data)
    best_gain = 0
    best_threshold = None
    for i in range(len(sorted_data)):
        # preventing repeated vals by cheking the previous vals
        if i>0 and sorted_data[i-1].values[feature] == sorted_data[i].values[feature]:
            pass
        else:
            left = sorted_data[:i]
            right = sorted_data[i:]

            left_tot = i
            right_tot = total_points - i 
            curr = (counts_to_entropy(left_count)*left_tot + counts_to_entropy(right_count)*right_tot)/total_points
            gain = entropy - curr
            # print gain
            if gain > best_gain:
                best_gain = gain
                best_threshold = sorted_data[i].values[feature]
        #counting labels left and right
        left_count[sorted_data[i].label]+=1
        right_count[sorted_data[i].label]-=1
    return (best_gain, best_threshold)

def find_best_split(data):
    if len(data) < 2:
        return None, None
    best_feature = None
    best_threshold = None
    best_gain = 0
    total_features = len(data[0].values)
    for feature in range(total_features):
        gain, threshold = find_best_threshold_fast(data, feature)
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_feature = feature
    return (best_feature, best_threshold)

def make_leaf(data):
    tree = Tree()
    counts = count_labels(data)
    prediction = {}
    for label in counts:
        prediction[label] = float(counts[label])/len(data)
    tree.prediction = prediction
    return tree

def predict(tree, point):
    if tree.leaf:
        return tree.prediction
    i = tree.feature
    if (point.values[i] < tree.threshold):
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)

def c45(data, max_levels):
    if max_levels <= 0:         #the maximum level depth is reached
        return make_leaf(data)
    
    feature, threshold = find_best_split(data)
    if threshold == None:       #there is no split that gains information
        return make_leaf(data)
    else:
        tree = Tree()
        tree.leaf = False
        tree.feature, tree.threshold = feature, threshold
        data_left, data_right = split_data(data, tree.feature, tree.threshold)
        tree.left  = c45(data_left, max_levels-1)
        tree.right = c45(data_right, max_levels-1)
        return tree

######################## random forrest ##############################
import random

def find_best_split_rand(data):
    if len(data) < 2:
        return None, None
    best_feature = random.randint(0,len(data[0].values)-1)
    gain, best_threshold = find_best_threshold_fast(data, best_feature)
    return (best_feature, best_threshold)

def c45_rand(data, max_levels):
    if max_levels <= 0:         #the maximum level depth is reached
        return make_leaf(data)
    
    feature, threshold = find_best_split_rand(data)
    if threshold == None: #there is no split that gains information
        return make_leaf(data)
    else:
        tree = Tree()
        tree.leaf = False
        tree.feature, tree.threshold = feature, threshold
        data_left, data_right = split_data(data, tree.feature, tree.threshold)
        tree.left  = c45_rand(data_left, max_levels-1)
        tree.right = c45_rand(data_right,max_levels-1)
        return tree

def dict_average_resuts_rand(point, forest):
    num_trees = len(forest)
    trees_pred_ham = []
    trees_pred_spam = []
    for i in range(num_trees):
        pred_dict = predict(forest[i], point)
        if 'ham' in pred_dict:
            trees_pred_ham.append(pred_dict['ham'])
        else:
            trees_pred_ham.append(0)
        if 'spam' in pred_dict:
            trees_pred_spam.append(pred_dict['spam'])
        else:
            trees_pred_spam.append(0)
    d = {'ham':sum(trees_pred_ham)/num_trees, 'spam':sum(trees_pred_spam)/num_trees}
    return d

def submission_random_forest(train, test, num_trees,depth):
    forest = []
    for i in range(num_trees):
        forest.append(c45_rand(train, depth))

    predictions = []
    for point in test:
        predictions.append(dict_average_resuts_rand(point, forest))
    return predictions

# In order to improve accuracy I ran a forloop of values optimizing max_levels
# Changing the depth of the tree improves or worsens your accuracy
# max_levels   acc
#     4        0.688
#     5        0.714
#     6        0.7456
#     9        0.7636
#    10        0.7456
#    15        0.7466
#    16        0.7442       
#    20        0.7418

# I've modified c45 fun to create random_forest algorithm
# In random forest I randomized feature selection.
# For prediction I used the average of the trees outputs.
# Although the output takes a bit to cumpute, the accuracy is beyond 80%
# trees   features    acc 
# 10         9        0.7346
# 20         9        0.7522
# 20         20       0.8034
# 20         30       0.8112
# 30         30       0.8222
# 40         30       0.8216
# 60         30       0.8298
# 150        30       0.8482

def submission(train, test):
#     tree = c45(train, 9)
#     predictions = []
#     for point in test:
#         predictions.append(predict(tree, point))
#     return predictions
      return submission_random_forest(train, test, 30, 30)


# This might be useful for debugging.
def print_tree(tree, level=0):
    if tree.leaf:
        print "Leaf", tree.prediction
    else:
        print "branch: ",level,'---', tree.feature, tree.threshold
        print 'left'
        print_tree(tree.left, level+1)
        print 'right'
        print_tree(tree.right, level+1)


### CODE MODIFICATION FOR PROBLEM 2 ###
def find_best_split2(data):
    if len(data) < 2:
        return None, None, None
    best_feature = None
    best_threshold = None
    best_gain = None
    total_features = len(data[0].values)
    for feature in range(total_features):
        gain, threshold = find_best_threshold_fast(data, feature)
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_feature = feature
    return (best_feature, best_threshold, best_gain)

def c45_gain_record(data, max_levels):
    if max_levels <= 0:         #the maximum level depth is reached
        return make_leaf(data)
    
    feature, threshold, gain = find_best_split2(data)
    if threshold == None: #there is no split that gains information
        return make_leaf(data)
    else:
        tree = Tree()
        tree.gain = gain
        tree.leaf = False
        tree.feature, tree.threshold = feature, threshold
        data_left, data_right = split_data(data, tree.feature, tree.threshold)
        tree.left  = c45_gain_record(data_left, max_levels-1)
        tree.right = c45_gain_record(data_right, max_levels-1)
        return tree

def print_tree_gain_record(tree, level=0):
    if tree.leaf:
        print "Leaf level", tree.prediction
    else:
        print "branch: ",level,':::', tree.feature, tree.threshold, tree.gain
        print 'left'
        print_tree_gain_record(tree.left, level+1)
        print 'right'
        print_tree_gain_record(tree.right, level+1)



# ##### TESTING #####
# from data import get_spam_train_data, get_spam_valid_data, get_college_data
# train = get_spam_train_data()
# print 'train loaded'
# valid = get_spam_valid_data()
# print 'valid loaded'
# predictions = submission_random_forest(train, valid, 20, 30)        
# acc = accuracy(valid, predictions)
# print acc


