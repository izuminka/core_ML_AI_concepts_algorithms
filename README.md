# Organization
Here is the collection of projects I wrote for my upper division CS165A - Artificial Intelligence and CS165B Machine Learning classes I took at UCSB. Projects are implemented in Python from scratch and demostrate my undestanding of core concepts and algorithms in AI and ML.

### Projects list (all are implemented in Python 3)
- **Naive Bayes Classifier**
- **Trees: C4.5 and Random Forrest**

### Uploading soon
- **HEX Bot Player (minimax with alpha-beta pruning)**
- **Logistic Regression with SGD**
- **Perceptron**
- **Multilayer perceptron**
- **K Means**

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Naive Bayes Classifier

I’ve implemented bag of words multinomial Naïve Bayes Classifier. The code is tested on Authors classification data taken from UCI database.

### Architecture
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

## Trees: C4.5 and Random Forrest
C45.py and random_forrest.py include the algorithms and are dependent on tree.py and data.py.
data.py contains data samples to test the algorithms. 

### C4.5 Architecture
Core function is c45. Fun inputs the list of datapoints and int cutoff dupth of the tree. The function recursively builds the tree untill either the cutoff depth is reached or there is no split that gains information. The best split of the data is found according to get the maximum gain.

### Random Forrest Architecture
The core function is rand_decision_tree. Fun inputs list of datapoints and int cutoff dupth of the tree. It is similar to c45 function, but the feature to split on is chosen randomly, not according to the best gain. 
The forrest is a list of such trees.

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



