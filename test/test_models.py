import numpy as np
import pytest
import tensorflow as tf
import utils

import sgd_classifier
from sgd_classifier import BasicSGDClassifier
import rnn_classifier
from rnn_classifier import RNNClassifier
import shallow_neural_network
from shallow_neural_network import ShallowNeuralNetwork
import tf_autoencoder
from tf_autoencoder import TfAutoencoder
import tf_shallow_neural_classifier
from tf_shallow_neural_classifier import TfShallowNeuralClassifier
import tf_rnn_classifier
from tf_rnn_classifier import TfRNNClassifier
import tree_nn
from tree_nn import TreeNN
import tf_snorkel_lite


@pytest.fixture
def X_xor():
    return [
        ([1.,1.], 1),
        ([1.,0.], 0),
        ([0.,1.], 0),
        ([0.,0.], 1)]


@pytest.fixture
def X_sequence():
    vocab = ['a', 'b', '$UNK']
    train = [
        [list('ab'), 'good'],
        [list('aab'), 'good'],
        [list('abb'), 'good'],
        [list('aabb'), 'good'],
        [list('ba'), 'bad'],
        [list('baa'), 'bad'],
        [list('bba'), 'bad'],
        [list('bbaa'), 'bad']]
    test = [
        [list('aaab'), 'good'],
        [list('baaa'), 'bad']]
    return train, test, vocab


def test_shallow_neural_network(X_xor):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X, y = zip(*X_xor)
    X = np.array(X)
    y = np.array([[x] for x in y])
    model = ShallowNeuralNetwork(
        hidden_dim=4,
        afunc=np.tanh,
        d_afunc=utils.d_tanh,
        eta=0.05,
        tol=1.5e-8,
        display_progress=True,
        max_iter=100)
    model.fit(X, y)
    model.predict(X)


def test_shallow_neural_network_simple_example():
    shallow_neural_network.simple_example()


def test_tf_shallow_neural_classifier(X_xor):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X, y = zip(*X_xor)
    model = TfShallowNeuralClassifier(
        hidden_dim=4,
        hidden_activation=tf.nn.tanh,
        max_iter=100,
        eta=0.01,
        tol=1e-4,
        display_progress=1)
    model.fit(X, y)
    model.predict(X)
    model.predict_proba(X)


def test_tf_shallow_neural_classifier_simple_example():
    tf_shallow_neural_classifier.simple_example()


def test_rnn_classifier(X_sequence):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    train, test, vocab = X_sequence
    embedding = np.array([utils.randvec(10) for _ in vocab])
    mod = RNNClassifier(
        vocab=vocab,
        embedding=embedding,
        hidden_dim=20,
        max_iter=100)
    X, y = zip(*train)
    X_test, _ = zip(*test)
    mod.fit(X, y)
    mod.predict(X_test)
    mod.predict_proba(X_test)
    mod.predict_one(X_test[0])
    mod.predict_one_proba(X_test[0])


def test_rnn_classifier_simple_example():
    rnn_classifier.simple_example()


def test_tf_rnn_classifier(X_sequence):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    train, test, vocab = X_sequence
    mod = TfRNNClassifier(vocab=vocab, max_iter=100, max_length=4)
    X, y = zip(*train)
    X_test, _ = zip(*test)
    mod.fit(X, y)
    mod.predict(X_test)
    mod.predict_proba(X_test)


def test_tf_rnn_classifier_simple_example():
    tf_rnn_classifier.simple_example()


def test_tf_autoencoder():
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X = utils.randmatrix(20, 50)
    ae = TfAutoencoder(hidden_dim=5, max_iter=100)
    ae.fit(X)
    ae.predict(X)


def test_tf_autoencoder_simple_example():
    tf_autoencoder.simple_example()


def test_sgd_classifier():
    sgd_classifier.simple_example()


@pytest.mark.parametrize("model, params", [
    [
        BasicSGDClassifier(max_iter=10, eta=0.1),
        {'max_iter': 100, 'eta': 1.0}
    ],
    [
        RNNClassifier(vocab=[], max_iter=10, hidden_dim=5, eta=0.1),
        {'hidden_dim': 10, 'eta': 1.0, 'max_iter': 100}
    ],
    [
        TfRNNClassifier(vocab=[], max_iter=10, hidden_dim=5, eta=0.1, max_length=5),
        {'hidden_dim': 10, 'eta': 1.0, 'max_iter': 100, 'max_length': 10}
    ],
    [
        TreeNN(vocab=[], max_iter=10, hidden_dim=5, eta=0.1),
        {'embed_dim': 5, 'hidden_dim': 10, 'eta': 1.0, 'max_iter': 100}
    ],
    [
        TfShallowNeuralClassifier(hidden_dim=5, hidden_activation=tf.nn.tanh, max_iter=1, eta=1.0),
        {'hidden_dim': 10, 'hidden_activation': tf.nn.relu, 'max_iter': 10, 'eta': 0.1}
    ],
    [
        TfAutoencoder(hidden_dim=5, hidden_activation=tf.nn.tanh, max_iter=1, eta=1.0),
        {'hidden_dim': 10, 'hidden_activation': tf.nn.relu, 'max_iter': 10, 'eta': 0.1}
    ]
])
def test_parameter_setting(model, params):
    model.set_params(**params)
    for p, val in params.items():
        assert getattr(model, p) == val


def test_snorkel_generative():
    result = tf_snorkel_lite.simple_example_generative()
    expected = ['disease', 'disease', 'cheese', 'cheese', 'cheese', 'cheese']
    assert result == expected


def test_snorkel_logistic_regression():
    tf_snorkel_lite.simple_example_logistic_regression()
