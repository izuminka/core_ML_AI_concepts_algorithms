# Organization
Here is the collection of code from upper division CS165A - Artificial Intelligence and CS165B Machine Learning classes I took at UCSB.

### Projects list (all are implemented in Python 3)
- **Naive Bayes Classifier**
- **C4.5 algorithm**
- **Logistic Regression with SGD**
- **Perceptron**
- **K Means**
- **Classifier**


## Naive Bayes Classifier

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
