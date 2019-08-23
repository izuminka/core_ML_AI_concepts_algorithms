class Point:
    """Input data point

    Args:
        label (str): Label of the data point (binary for this algorithm).
        values (list): List of floats constituting a datapoint.

    """
    def __init__(self, label, values):
        self.label = label
        self.values = values

def get_train_data():
    """Tiny training data sample just for the demo

    Returns:
        list: data points of class Point

    """
    data = \
        [
            Point('College', [22, 38000]),
            Point('No College', [23, 25000]),
            Point('College', [24, 40000]),
            Point('College', [25, 77000]),
            Point('College', [32, 48000]),
            Point('No College', [43, 44000]),
            Point('College', [52, 110000]),
            Point('No College', [53, 52000]),
        ]
    return data


def get_test_data():
    """Tiny testing data sample just for the demo

    Returns:
        list: List of data points of class Point

    """
    data = \
        [
            Point('College', [42, 75000]),
            Point('No College', [52, 27000]),
        ]
    return data
