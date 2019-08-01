## Naive Bayes Classifier

### General
I’ve implemented bag of words multinomial Naïve Bayes in python3
### Procedure
1. Code parses the input file and imports data
2. Each sample for each author gets converted to BOW (bag of words).
3. Probability tables are computed and stored based on training data
a. P_c - number of author samples/total num samples
b. P_tc - conditional probability P(word | author)
4. Testing file gets imported in BOW format
5. Using probability tables computes the given unseen BOW samples
6. Performance metric is computed, times, etc. gets printed
