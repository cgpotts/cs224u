from nltk.tree import Tree
from np_shallow_neural_classifier import ShallowNeuralClassifier
from np_rnn_classifier import RNNClassifier
from np_autoencoder import Autoencoder
from np_tree_nn import TreeNN
import numpy as np
import pytest
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


class GradientCheckError(Exception):
    """Raised if a gradient check fails."""


@pytest.mark.parametrize("hidden_activation, d_hidden_activation", [
    [np.tanh, utils.d_tanh],
    [utils.relu, utils.d_relu]
])
def test_np_shallow_neural_classifier_gradients(hidden_activation, d_hidden_activation):
    model = ShallowNeuralClassifier(
        max_iter=10,
        hidden_activation=hidden_activation,
        d_hidden_activation=d_hidden_activation)
    # A tiny dataset so that we can run `fit` and set all the model
    # parameters:
    X = utils.randmatrix(5, 2)
    y = np.random.choice((0,1), 5)
    model.fit(X, y)
    # Use the first example for the check:
    ex = X[0]
    label = model._onehot_encode([y[0]])[0]
    # Forward and backward to get the gradients:
    hidden, pred = model.forward_propagation(ex)
    d_W_hy, d_b_hy, d_W_xh, d_b_xh = model.backward_propagation(
        hidden, pred, ex, label)
    # Model parameters to check:
    param_pairs = (
        ('W_hy', d_W_hy),
        ('b_hy', d_b_hy),
        ('W_xh', d_W_xh),
        ('b_xh', d_b_xh)
    )
    gradient_check(param_pairs, model, ex, label)


@pytest.mark.parametrize("hidden_activation, d_hidden_activation", [
    [np.tanh, utils.d_tanh],
    [utils.relu, utils.d_relu]
])
def test_np_rnn_classifier(hidden_activation, d_hidden_activation):
    # A tiny dataset so that we can run `fit` and set all the model
    # parameters:
    vocab = ['a', 'b', '$UNK']
    data = [
        [list('ab'), 'good'],
        [list('aab'), 'good'],
        [list('abb'), 'good']]
    model = RNNClassifier(
        vocab,
        max_iter=10,
        hidden_dim=2,
        hidden_activation=hidden_activation,
        d_hidden_activation=d_hidden_activation)
    X, y = zip(*data)
    model.fit(X, y)
    # Use the first example for the check:
    ex = X[0]
    label = model._onehot_encode([y[0]])[0]
    # Forward and backward to get the gradients:
    hidden, pred = model.forward_propagation(ex)
    d_W_hy, d_b, d_W_hh, d_W_xh = model.backward_propagation(
        hidden, pred, ex, label)
    # Model parameters to check:
    param_pairs = (
        ('W_xh', d_W_xh),
        ('W_hh', d_W_hh),
        ('W_hy', d_W_hy),
        ('b', d_b)
    )
    gradient_check(param_pairs, model, ex, label)


@pytest.mark.parametrize("hidden_activation, d_hidden_activation", [
    [np.tanh, utils.d_tanh],
    [utils.relu, utils.d_relu]
])
def test_np_autoencoder(hidden_activation, d_hidden_activation):
    model = Autoencoder(
        max_iter=10,
        hidden_dim=2,
        hidden_activation=hidden_activation,
        d_hidden_activation=d_hidden_activation)
    # A tiny dataset so that we can run `fit` and set all the model
    # parameters:
    X = utils.randmatrix(5, 5)
    model.fit(X)
    # Use the first example for the check:
    ex = X[0]
    label = X[0]
    # Forward and backward to get the gradients:
    hidden, pred = model.forward_propagation(ex)
    d_W_hy, d_b_hy, d_W_xh, d_b_xh = model.backward_propagation(
        hidden, pred, ex, label)
    # Model parameters to check:
    param_pairs = (
        ('W_hy', d_W_hy),
        ('b_hy', d_b_hy),
        ('W_xh', d_W_xh),
        ('b_xh', d_b_xh)
    )
    gradient_check(param_pairs, model, ex, label)


@pytest.mark.parametrize("hidden_activation, d_hidden_activation", [
    [np.tanh, utils.d_tanh],
    [utils.relu, utils.d_relu]
])
def test_np_tree_nn(hidden_activation, d_hidden_activation):
    # A tiny dataset so that we can run `fit` and set all the model
    # parameters:
    vocab = ["1", "+", "2"]
    X = [
        "(even (odd 1) (neutral (neutral +) (odd 1)))",
        "(odd (odd 1) (neutral (neutral +) (even 2)))"]
    X = [Tree.fromstring(ex) for ex in X]
    model = TreeNN(
        vocab,
        max_iter=10,
        hidden_dim=5,
        hidden_activation=hidden_activation,
        d_hidden_activation=d_hidden_activation)
    model.fit(X)
    # Use the first example for the check:
    ex = X[0]
    label = model._onehot_encode([ex.label()])[0]
    # Forward and backward to get the gradients:
    hidden, pred = model.forward_propagation(ex)
    d_W_hy, d_b_y, d_W, d_b = model.backward_propagation(
        hidden, pred, ex, label)
    # Model parameters to check:
    param_pairs = (
        ('W_hy', d_W_hy),
        ('b_y', d_b_y),
        ('W', d_W),
        ('b', d_b)
    )
    gradient_check(param_pairs, model, ex, label)


def gradient_check(param_pairs, model, ex, label, epsilon=0.0001, threshold=0.001):
    """Numerical gradient check following the method described here:

    http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization

    Parameters
    ----------
    param_pairs : list of str, np.aray pairs
        In each pair, the first is the name of the parameter to check,
        and the second is its purported derivatives. We use the name
        as the first pair so that we can raise an informative error
        message in the case of a failure.
    model : trained model instance
        This should have attributes for all of the parameters named in
        `param_pairs`, and it must have methods `forward_propagation`,
        and `get_error`.
    ex : an example that `model` can process
    label : a label vector that `model` can learn from directly
    epsilon : float
        The small constant by which the parameter values are changed.
    threshold : float
        Tolerance for raising an error.

    Raises
    ------
    GradientCheckError

    """
    for param_name, d_params in param_pairs:
        params = getattr(model, param_name)
        # This iterator will allow is to cycle over all the values for
        # arrays of any dimension:
        iterator = np.nditer(params, flags=['multi_index'], op_flags=['readwrite'])
        while not iterator.finished:
            idx = iterator.multi_index
            actual = params[idx]
            params[idx] = actual + epsilon
            _, pred = model.forward_propagation(ex)
            grad_pos = model.get_error(pred, label)
            params[idx] = actual - epsilon
            _, pred = model.forward_propagation(ex)
            grad_neg = model.get_error(pred, label)
            grad_est = (grad_pos - grad_neg) / (epsilon * 2.0)
            params[idx] = actual
            grad_bp = d_params[idx]
            # Relative error to control for differences in proportion
            # across parameter values:
            err = np.abs(grad_bp - grad_est) / (np.abs(grad_bp) + np.abs(grad_est))
            if err >= threshold:
                raise GradientCheckError(
                    "Gradient check error for {} at {}: error is {}".format(
                        param_name, idx, err))
            iterator.iternext()
