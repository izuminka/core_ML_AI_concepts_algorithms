# Organization
Here is the collection of code from upper division CS165A - Artificial Intelligence and CS165B Machine Learning classes I took at UCSB.

### Projects list (all are implemented in Python 3)
- **Naive Bayes Classifier**
- **C4.5 algorithm**
- **Logistic Regression with SGD**
- **Perceptron**
- **K Means**
- **Classifier**

***

# Naive Bayes Classifier

### General
I’ve implemented bag of words multinomial Naïve Bayes Classifier. The code is tested on Authors classification data taken from UCI database.
### Procedure
1. Code parses the input file and imports data
2. Each sample for each author gets converted to BOW (bag of words).
3. Probability tables are computed and stored based on training data
a. P_c - number of author samples/total num samples
b. P_tc - conditional probability P(word | author)
4. Testing file gets imported in BOW format
5. Using probability tables computes the given unseen BOW samples
6. Performance metric is computed, times, etc. gets printed

***

## HEX Bot Player.
Bot for [HEX game](https://en.wikipedia.org/wiki/Hex_(board_game)) using standard Alpha Beta search.

### Architecture
I’ve used Python to complete the assignment. I’ve implemented alpha beta algorithm
with a simple evaluation function. I’ve implemented class Hex which incorporates everything
needed to play the game. Key functions: alphabeta(), evalFun(), bfs().

### Search
Standard Alpha Beta search. I cap the time to insure the search will not go beyond 30
sec. I additionally tried to cap the depth, but unsuccessfully. I’ve implemented
A quick evaluation function that awards making straight connections, or connections that are
only slightly off course. I additionally penalize if the opponent is cutting of my path.


