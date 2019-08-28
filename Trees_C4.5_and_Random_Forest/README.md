## Trees: C4.5 and Random Forest
C45.py and random_forrest.py include the algorithms and are dependent on tree.py and
data.py. data.py contains data samples to test the algorithms. Random fore

### C4.5 Architecture and Procedure
1. (list) training_data containing points of class Point (def in data.py) is imported
2. (list) training_data, (int) TREE_DEPTH are passed to c45 function to create a decision tree
3. c45 calculates the information gain in order to select the best feature to split on
4. Threshold of the selected feature is calculated and data is split into left and right lists
5. Steps 3, 4 are executed recursively until either the cutoff depth is reached or there is no split that gains information
6. c45 returns class Tree (def in tree.py): learned_tree
7. Function predict(tree, point) predicts the label of the point

To test the algorithm run:

    python c45.py

### Random Forest Architecture and Procedure

1. (list) training_data containing points of class Point (def in data.py) is imported
2. (list) training_data, (int) TREE_DEPTH are passed to rand_decision_tree function to create a decision tree
3. rand_decision_tree chooses a random feature to split on
4. Similar to c45 threshold of the selected feature is calculated and data is split into left and right lists
5. Steps 3, 4 are executed recursively until either the cutoff depth is reached or there is no split that gains information
6. rand_decision_tree returns class Tree (def in tree.py): learned_tree
7. A list of rand_decision_tree is created, constituting a random forest
8. Function dict_average_resuts_rand(point, forest) predicts the label of the point

To test the algorithm run:

    python random_forest.py
