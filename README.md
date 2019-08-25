# Organization
Here is the collection of projects I wrote for my upper division CS165A - Artificial Intelligence and CS165B Machine Learning classes I took at UCSB. Projects are implemented in Python from scratch and demonstrate my understanding of core concepts and algorithms in AI and ML.

### Projects list (all are implemented in Python 3)
- **Naive Bayes Classifier**
- **Trees: C4.5 and Random Forest**

### Uploading soon
- **HEX Bot Player (minimax with alpha-beta pruning)**
- **Logistic Regression with SGD**
- **Perception**
- **Multiplayer perception**
- **K Means**


&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Naive Bayes Classifier

I’ve implemented bag of words multinomial Naïve Bayes Classifier. The code is tested on Authors classification data taken from UCI database.

### Architecture and Procedure
1. Code parses the input file and imports data
2. Each sample for each author gets converted to BOW (bag of words).
3. Probability tables are computed and stored based on training data
a. P_c - number of author samples/total num samples
b. P_tc - conditional probability P(word | author)
4. Testing file gets imported in BOW format
5. Using probability tables computes the given unseen BOW samples
6. Performance metric is computed, times, etc. gets printed

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

<<<<<<< HEAD
## Trees: C4.5 and Random Forest
C45.py and random_forrest.py include the algorithms and are dependent on tree.py and data.py.
data.py contains data samples to test the algorithms. Random fore

### C4.5 Architecture and Procedure
1. (list) training_data containing points of class Point (def in data.py) is imported
2. (list) training_data, (int) TREE_DEPTH are passed to c45 function to create a decision tree
3. c45 calculates the information gain in order to select the best feature to split on
4. Threshold of the selected feature is calculated and data is split into left and right lists
5. Steps 3, 4 are executed recursively until either the cutoff depth is reached or there is no split that gains information
6. c45 returns class Tree (def in tree.py): learned_tree
7. Function predict(tree, point) predicts the label of the point

### Random Forest Architecture and Procedure

1. (list) training_data containing points of class Point (def in data.py) is imported
2. (list) training_data, (int) TREE_DEPTH are passed to rand_decision_tree function to create a decision tree
3. rand_decision_tree chooses a random feature to split on
4. Similar to c45 threshold of the selected feature is calculated and data is split into left and right lists
5. Steps 3, 4 are executed recursively until either the cutoff depth is reached or there is no split that gains information
6. rand_decision_tree returns class Tree (def in tree.py): learned_tree
7. A list of rand_decision_tree is created, constituting a random forest
8. Function dict_average_resuts_rand(point, forest) predicts the label of the point
=======
## Trees: C4.5 and Random Forrest
C45.py and random_forrest.py include the algorithms and are dependent on tree.py and data.py.
data.py contains data samples to test the algorithms. 

### C4.5 Architecture
Core function is c45. Fun inputs the list of datapoints and int cutoff dupth of the tree. The function recursively builds the tree untill either the cutoff depth is reached or there is no split that gains information. The best split of the data is found according to get the maximum gain.

### Random Forrest Architecture
The core function is rand_decision_tree. Fun inputs list of datapoints and int cutoff dupth of the tree. It is similar to c45 function, but the feature to split on is chosen randomly, not according to the best gain. 
The forrest is a list of such trees.
>>>>>>> 1eb5a5edb1733b6caeac89f6b7ffd09c45ce015b

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

## HEX Bot Player
Bot for [HEX game](https://en.wikipedia.org/wiki/Hex_(board_game)) using minimax with alpha-beta pruning.

### Architecture
I’ve implemented alpha beta algorithm with a simple evaluation function. I’ve implemented class Hex which incorporates everything
needed to play the game. Key functions: alphabeta(), evalFun(), bfs().

Algorithm uses Standard Alpha Beta search. I cap the time to insure the search will not go beyond 30
sec. I additionally tried to cap the depth, but unsuccessfully. I’ve implemented
A quick evaluation function that awards making straight connections, or connections that are
only slightly off course. I additionally penalize if the opponent is cutting of my path.
