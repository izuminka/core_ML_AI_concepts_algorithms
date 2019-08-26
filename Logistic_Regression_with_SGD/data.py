import csv
# from random import shuffle


def accuracy(data_points, predictions):
    """Calulcate accuracy of predictions.

    Args:
        data_points (list): each p is {'features': list_floats, 'label': int_zero_or_one}
        predictions (list): floats (0 to 1)

    Returns:
        float: Description of returned object.

    """
    correct = 0
    for i in range(len(data_points)):
        isPredLabel = predictions[i] >= 0.5
        if data_points[i]['label'] == isPredLabel:
            correct += 1
    return correct / len(data_points)


def load_csv(filename):
    """Load data

    Args:
        filename (str): name of the input file

    Returns:
        list: points of class OrderedDict([(str_feature, str_val)..])

    """

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        lines = [line for line in reader]
    return lines


def from_raw_to_clean_dp(raw_dp):
    """Clean the imported string features and convert to float features.
    Assign the lable. Get rid of unnesessary inputs as well.

    Args:
        raw_dp (list): points of class OrderedDict([(str_feature, str_val)..])

    Returns:
        list: each dp is {'features': list_floats, 'label': int_zero_or_one}

    """
    clean_dp = {}

    # create label for the datapoint, if inome is greater than 50k
    clean_dp["label"] = (raw_dp['income'] == '>50K')

    # create features from a datapoint converting from string vals
    features = []
    # float values only
    features.append(float(raw_dp['age']) / 100)
    features.append(float(raw_dp['education_num']) / 20)
    features.append(float(raw_dp['hr_per_week']) / 60)
    # from string to binary
    features.append(raw_dp['marital'] == 'Married-civ-spouse')
    features.append(raw_dp['relationship'] not in ('Husband', 'Wife'))
    features.append(raw_dp['education'] in ('1st-4th', '5th-6th', '7th-8th',
                                            '9th', '10th', '11th', '12th'))
    features.append(raw_dp['education'] == 'HS-grad')
    features.append(raw_dp['education'] == 'Some-college')
    features.append(raw_dp['education'] in ('Assoc-acdm', 'Assoc-voc'))
    features.append(raw_dp['education'] == 'Bachelors')
    features.append(raw_dp['education'] == 'Masters')
    features.append(raw_dp['education'] in ('Prof-school', 'Doctorate'))
    features.append(raw_dp['occupation'] == 'Prof-specialty')
    features.append(raw_dp['occupation'] == 'Exec-managerial')

    clean_dp['features'] = features
    return clean_dp


def get_clean_data(filename):
    """Upload data in csv, clean features and convert for workable format

    Args:
        filename (str): name of the input file with data

    Returns:
        list: each dp is {'features': list_floats, 'label': int_zero_or_one}

    """
    raw_data = load_csv(filename)
    return [from_raw_to_clean_dp(p) for p in raw_data]


def get_train_test_data():
    """Temorary funcrtion for quick testing of the algorithm. Splits the data
    into train and test

    Returns:
        tuple: (list) training_dp, (list) testing_dp

    """
    data = get_clean_data("adult.data")

    train_split = 0.7
    split_ind = int(train_split * len(data))
    # shuffle(data)
    train_data = data[:split_ind]
    test_data = data[split_ind:]
    return train_data, test_data
