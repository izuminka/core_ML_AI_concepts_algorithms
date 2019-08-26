## Logistic Regression with SGD


### Architecture
The Logistic Regression performed with Stochastic Gradient Descent for model update.

    Loss = (Prediction - Truth)^2 + Lambda * || W ||

Derivative of the Loss function is calculated with respect to the time step and
the weight is updated. To control the learning process learning rate is multiplied
by the derivative before the weight of each feature is updated.

### Procedure
0. (list) training_data containing points of class Point (def in data.py) is imported

1. The core function is train(data_points, epochs, rate, lam), where
data_points (list): each p is {'features': list_floats, 'label': int_zero_or_one}
epochs (int): number of epochs to perform
rate (float): learning rate
lam (float): regularization parameter

2. Train the model weights. Update the model via Stochastic Gradient Descent.
Adjust the model weights according the gradient of the error where
loss = (Prediction - Truth)^2 + Lambda * || W ||


To test the algorithm run:

    python lr_sgd.py
