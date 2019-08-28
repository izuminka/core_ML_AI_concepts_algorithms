from math import log
import time


def get_raw_labeled_data(file_name, out_type):
    """Read data and put in dict or list format.

    Args:
        file_name (str): Name if the inputting file
        out_type (str): Choice of import: dict or string

    Returns:
        dict or list: organized dataset
        if dict: {author: [corpus]}
        if list: [[labels_author_1..labels_author_n],
                 [corpus_author_1..corpus_author_n] in order of the inputted text

    """
    f = open(file_name, errors='ignore')
    samples_authors = f.readlines()
    f.close()

    if out_type == 'dict':
        dataset = {}
        for s_a in samples_authors:
            sample, author = s_a.strip().split(',')
            if author in dataset:
                dataset[author].append(sample)
            else:
                dataset[author] = [sample]
    elif out_type == 'list':
        dataset_text = []
        dataset_authors = []
        for s_a in samples_authors:
            sample, author = s_a.strip().split(',')
            dataset_text.append(sample)
            dataset_authors.append(author)
        dataset = [dataset_text, dataset_authors]
    return dataset


def bag_of_words(text_string):
    """Crete bow dictionary counting the number of times the word occurs in
    text_string

    Args:
        text_string (string): the imported sample of text of author_i

    Returns:
        dict: {word:num_occurances}

    """

    words_ls = text_string.split()
    bow = {}
    for w in words_ls:
        if w in bow:
            bow[w] += 1
        else:
            bow[w] = 1
    return bow


def combine_bows(bow_ls):
    """Make disjoint union of all provided bows for a given label/author

    Args:
        bow_ls (list): [bow_sample_1..bow_sample_n], bow_sample_i = {word:occurance_in_sample_i}

    Returns:
        dict: {word:total_occurance_in_samples}

    """

    combined_bow = {}
    for bow in bow_ls:
        for w in bow:
            if w in combined_bow:
                combined_bow[w] += bow[w]
            else:
                combined_bow[w] = bow[w]
    return combined_bow


def parse_raw_data_bow(dataset):
    """Create bow representation for each author and sample

    Args:
        dataset: dataset (dict): {author: [corpus]} where corpus is
                 [sample_1..sample_n], sample_i (str)

    Returns:
        dict: {author: {word:occurance_in_all_samples}}

    """
    vocabulary = []
    bow_data_set = {}
    training_samples_stat = {}
    for author in dataset:
        list_samples_bow = []
        for sample in dataset[author]:
            bow = bag_of_words(sample)
            vocabulary += list(bow.keys())
            list_samples_bow.append(bow)
        bow_data_set[author] = combine_bows(list_samples_bow)
        training_samples_stat[author] = len(list_samples_bow)
    return bow_data_set, list(set(vocabulary)), training_samples_stat


def get_P_tc(authors_total_words, vocab, bow_data_set):
    """Calculate conditional probability

    Args:
        authors_total_words (dict): {author: total_words_in_corpus}
        vocab (list): list of all unique words across all samples all authors
        bow_data_set (dict): {author: {word:multiplicity_across_all_corpus}}

    Returns:
        dict: {author: {word:prob}} Conditional probability of word given
              class, for each author

    """
    authors_ls = list(authors_total_words.keys())
    total_unique_words = len(vocab)
    P_tc = {a: {} for a in authors_ls}
    for author in authors_ls:
        for word in vocab:
            if word not in bow_data_set[author]:
                word_mult = 0
            else:
                word_mult = bow_data_set[author][word]
            P_tc[author][word] = (
                word_mult + 1) / (authors_total_words[author] + total_unique_words)
    return P_tc


def get_probabilities(training_samples_stat, vocab, bow_data_set):
    """Calculate probabilities for all labels given training data
    Args:
        training_samples_stat (dict): {author: number_samples}
        vocab (list): list of all unique words across all samples all authors
        bow_data_set (dict): {author: {word:multiplicity_across_all_corpus}}

    Returns:
        tuple: P_c (dict): {author: num_samples_corpus/total_samples}
               P_tc (dict): {author: {word:prob}} where prob is conditional
                            probability of word given class, for each author

    """
    total_text_samples = sum(training_samples_stat.values())
    P_c = {a: training_samples_stat[a] /
           total_text_samples for a in training_samples_stat}

    authors_total_words = {
        a: sum(bow_data_set[a].values()) for a in bow_data_set}
    P_tc = get_P_tc(authors_total_words, vocab, bow_data_set)

    return P_c, P_tc


def train(training_file_name):
    """Training function
    Args:
        training_file_name (str): name of the training file

    Returns:
        tuple: P_c (dict): {author: num_samples_corpus/total_samples}
               P_tc (dict): {author: {word:prob}} where prob is conditional
                            probability of word given class, for each author
               vocab_len (int): num of all unique words across all samples, authors

    """
    training_dataset_raw = get_raw_labeled_data(training_file_name, 'dict')
    bow_data_set, vocab, training_samples_stat = parse_raw_data_bow(
        training_dataset_raw)
    P_c, P_tc = get_probabilities(training_samples_stat, vocab, bow_data_set)
    return P_c, P_tc, len(vocab)


def predict_label(bow_test, P_tc, P_c, vocab_len):
    """Predict the best match for a given BOW

    Args:
        bow_test (dict): BOW of the testing text
        P_c (dict): {author: num_samples_corpus/total_samples}
        P_tc (dict): {author: {word:prob}} where prob is conditional
                    probability of word given class, for each author
        vocab_len (int): number of all unique words across all samples all authors

    Returns:
        str: Predicted label

    """
    authors_ls = P_c.keys()
    results = {}
    for author in authors_ls:
        P_cd = 0
        for word in bow_test:
            word_mult = bow_test[word]
            if word not in P_tc[author]:  # handles unsceen words
                p_tc = 1 / vocab_len
            else:
                p_tc = P_tc[author][word]
            P_cd += word_mult * log(p_tc)
        P_cd += P_c[author]
        results[author] = P_cd
    best_match = max(results, key=results.get)
    return best_match


def predict(testing_file_name, P_tc, P_c, vocab_len):
    """Predict labels for all inputted samples

    Args:
        testing_file_name: name of the inputting test file
        P_tc (dict): {author: {word:prob}} Conditional probability of word given class, for each author
        P_tc (dict): {author: {word:prob}} Conditional probability of word given class, for each author
        vocab_len (int): number of all unique words across all samples all authors

    Returns:
        tuple: predictions (list): labels in order of inputted samples
               accuracy (float): correct / (correct + incorrect)

    """
    testing_texts, testing_authors = get_raw_labeled_data(
        testing_file_name, 'list')
    testing_bows = [bag_of_words(text) for text in testing_texts]
    correct = 0
    incorrect = 0
    predictions = []
    for i in range(len(testing_bows)):
        test_bow = testing_bows[i]
        correct_author = testing_authors[i]
        predicted_author = predict_label(test_bow, P_tc, P_c, vocab_len)
        predictions.append(predicted_author)
        if predicted_author == correct_author:
            correct += 1
        else:
            incorrect += 1
    return predictions, correct / (correct + incorrect)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str)
    parser.add_argument('test_file', type=str)

    args = parser.parse_args()
    training_file_name = args.train_file
    testing_file_name = args.test_file

    # train, time
    train_start = time.time()
    P_c, P_tC, vocab_len = train(training_file_name)
    train_time = time.time() - train_start

    # predict on train set, no time
    predicted_authors_train, accuracy_train = predict(
        training_file_name, P_tC, P_c, vocab_len)

    # predict on test set, time
    test_start = time.time()
    predicted_authors_test, accuracy_test = predict(
        testing_file_name, P_tC, P_c, vocab_len)
    test_time = time.time() - test_start

    # print results
    for a in predicted_authors_test:
        print(a)
    print(round(train_time, 5), 'seconds (training)')
    print(round(test_time, 5), 'seconds (testing)')
    print(round(accuracy_train, 5), '(training)')
    print(round(accuracy_test, 5), '(testing)')
