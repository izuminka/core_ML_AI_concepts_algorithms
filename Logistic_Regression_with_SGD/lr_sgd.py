from math import exp
import random


def logistic(float_x):
    """Logistic function for a float value

    """
    return 1 / (1 + exp(-float_x))


def dot(ls_x, ls_y):
    """Dot product btw 2 lists.

    Args:
        ls_x (list): ls of floats
        ls_y (list): ls of floats

    Returns:
        float: Dot product

    """

    if len(ls_x) != len(ls_y):
        raise ValueError("Lists have different lengths")

    dot_product = 0
    for i in range(len(ls_x)):
        dot_product += ls_x[i] * ls_y[i]
    return dot_product


def l2_norm(ls_x, ls_y):
    """L2 product btw 2 lists.

    Args:
        ls_x (list): ls of floats
        ls_y (list): ls of floats

    Returns:
        float: Dot product

    """
    return dot(ls_x, ls_y)**(1 / 2)


def predict(model, data_point):
    """Classify a datapoint

    Args:
        model (list): The learned model
        data_point (dict): {'features': list_floats, 'label': int_zero_or_one}

    Returns:
        float: prediction (btw 0 or 1)

    """
    w_x = dot(model, data_point['features'])
    return logistic(w_x)


def update(model, point, rate, lam):
    """Update the model via Stochastic Gradient Descent.
     Adjust the model weights according the gradient of the error.
     loss = squared_error_loss + regularization_loss
     loss = (Prediction - Truth)^2 + Lambda * || W ||

    Args:
        model (list): model weights
        point (dict): {'features': list_floats, 'label': int_zero_or_one}
        rate (float): learning rate
        lam (float): regularization parameter

    Returns:
        list: updated model weights

    """
    truth = point['label']
    prediction = predict(model, point)

    for i in range(len(model)):
        # loss = squared_error_loss + regularization_loss
        # loss = (prediction - truth)^2 + lam * l2_norm(model, model)
        # d_loss_dt = d_squared_error_loss_dt + d_regularization_loss_dt
        d_prediction_dt = point['features'][i] * prediction * (1 - prediction)
        d_squared_error_loss_dt = 2 * (prediction - truth) * d_prediction_dt
        d_regularization_loss_dt = lam * model[i] / l2_norm(model, model)
        d_loss_dt = d_squared_error_loss_dt + d_regularization_loss_dt
        model[i] -= rate * d_loss_dt
    return model


def initialize_model(model_len):
    """Initialize the model (the list of weights)

    Args:
        model_len (int): length of the list.

    Returns:
        list: model weights with random vals in range 0 to 1.

    """
    return [random.gauss(0, 1) for _ in range(model_len)]


def train(data_points, epochs, rate, lam):
    """Train the model weights. Use SGD for weight update.

    Args:
        data_points (list): each p is {'features': list_floats, 'label': int_zero_or_one}
        epochs (int): number of epochs to perform
        rate (float): learning rate
        lam (float): regularization parameter

    Returns:
        list: model weights

    """
    model = initialize_model(len(data_points[0]['features']))
    random_index_ls = list(range(len(data_points)))
    for _ in range(epochs):
        random.shuffle(random_index_ls)  # randomize sequence of features
        for i in random_index_ls:
            model = update(model, data_points[i], rate, lam)
    return model


if __name__ == '__main__':
    from data import accuracy, get_train_test_data

    train_dp, test_dp = get_train_test_data()
    EPOCHS = 5
    RATE = 0.01 #learning rate
    LAM = 0.001 #regularization parameter
    trained_model = train(train_dp, EPOCHS, RATE, LAM)
    predictions = [predict(trained_model, dp) for dp in test_dp]
    acc = accuracy(test_dp, predictions)
    print(acc)
