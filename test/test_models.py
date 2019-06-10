from nltk.tree import Tree
import numpy as np
import os
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
import tensorflow as tf
import torch.nn as nn
import utils

import np_sgd_classifier
import np_shallow_neural_classifier
import np_rnn_classifier
import np_autoencoder
import np_tree_nn

import tf_shallow_neural_classifier
import tf_rnn_classifier
import tf_autoencoder

import torch_shallow_neural_classifier
import torch_rnn_classifier
import torch_autoencoder
import torch_tree_nn
import torch_subtree_nn

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


@pytest.fixture
def XOR():
    dataset = [
        ([1.,1.], True),
        ([1.,0.], False),
        ([0.,1.], False),
        ([0.,0.], True)]
    X, y = zip(*dataset)
    X = np.array(X)
    y = list(y)
    return X, y


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


@pytest.fixture
def X_tree():
    vocab = ["1", "+", "2", "$UNK"]
    train = [
        "(odd 1)",
        "(even 2)",
        "(odd (pdd 1))",
        "(even (even 2))",
        "(even (odd 1) (neutral (neutral +) (odd 1)))",
        "(odd (odd 1) (neutral (neutral +) (even 2)))",
        "(odd (even 2) (neutral (neutral +) (odd 1)))",
        "(even (even 2) (neutral (neutral +) (even 2)))",
        "(even (odd 1) (neutralB (neutral +) (odd (odd 1) (neutral (neutral +) (even 2)))))"]
    X_train = [Tree.fromstring(x) for x in train]
    return X_train, vocab


@pytest.fixture
def cheese_disease_dataset():
    X = []
    y = []
    src_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "cheeseDisease.train.txt")
    with open(src_filename, encoding='utf8') as f:
        for line in f:
            label, ex = line.split("\t", 1)
            label = "cheese" if label.strip() == "1" else "disease"
            ex = list(ex.lower().strip())
            X.append(ex)
            y.append(label)
    vocab = list(string.ascii_lowercase) + ["$UNK"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    return {'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'vocab': vocab}


def test_np_shallow_neural_classifier(XOR):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X, y = XOR
    model = np_shallow_neural_classifier.ShallowNeuralClassifier(
        hidden_dim=4,
        hidden_activation=np.tanh,
        d_hidden_activation=utils.d_tanh,
        eta=0.05,
        tol=1.5e-8,
        display_progress=True,
        max_iter=100)
    model.fit(X, y)
    model.predict(X)


def test_np_shallow_neural_classifier_simple_example():
    acc = np_shallow_neural_classifier.simple_example()
    assert acc >= 0.90


def test_torch_shallow_neural_classifier(XOR):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X, y = XOR
    model = torch_shallow_neural_classifier.TorchShallowNeuralClassifier(
        hidden_dim=4,
        hidden_activation=nn.ReLU(),
        max_iter=100,
        eta=0.01)
    model.fit(X, y)
    model.predict(X)
    model.predict_proba(X)


def test_torch_shallow_neural_classifier_simple_example():
    acc = torch_shallow_neural_classifier.simple_example()
    assert acc >= 0.90


def test_torch_shallow_neural_classifier_incremental(XOR):
    X, y = XOR
    model = torch_shallow_neural_classifier.TorchShallowNeuralClassifier(
        hidden_dim=4,
        hidden_activation=nn.ReLU(),
        max_iter=100,
        eta=0.01)
    model.fit(X, y, X_dev=X, dev_iter=1)
    epochs = list(model.dev_predictions.keys())
    assert epochs == list(range(1, 101))
    assert all(len(v)==len(X) for v in model.dev_predictions.values())


def test_tf_shallow_neural_classifier(XOR):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X, y = XOR
    model = tf_shallow_neural_classifier.TfShallowNeuralClassifier(
        hidden_dim=4,
        hidden_activation=tf.nn.tanh,
        max_iter=100,
        eta=0.01)
    model.fit(X, y)
    model.predict(X)
    model.predict_proba(X)


def test_tf_shallow_neural_classifier_simple_example():
    acc = tf_shallow_neural_classifier.simple_example()
    assert acc >= 0.90


def test_np_rnn_classifier(X_sequence):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    train, test, vocab = X_sequence
    embedding = np.array([utils.randvec(10) for _ in vocab])
    mod = np_rnn_classifier.RNNClassifier(
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


@pytest.mark.parametrize("initial_embedding, use_embedding",[
    [True, False],
    [True, True],
    [False, False],
    [False, True]
])
def test_np_rnn_classifier_simple_example(initial_embedding, use_embedding):
    np_rnn_classifier.simple_example()


def test_torch_rnn_classifier(X_sequence):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    train, test, vocab = X_sequence
    mod = torch_rnn_classifier.TorchRNNClassifier(
        vocab=vocab, max_iter=100)
    X, y = zip(*train)
    X_test, _ = zip(*test)
    mod.fit(X, y)
    mod.predict(X_test)
    mod.predict_proba(X_test)


def test_torch_rnn_classifier_incremental(X_sequence):
    train, test, vocab = X_sequence
    model = torch_rnn_classifier.TorchRNNClassifier(
        vocab=vocab, max_iter=100)
    X, y = zip(*train)
    X_test, _ = zip(*test)
    model.fit(X, y, X_dev=X_test, dev_iter=20)
    epochs = list(model.dev_predictions.keys())
    assert epochs == list(range(20, 101, 20))
    assert all(len(v)==len(X_test) for v in model.dev_predictions.values())


def test_torch_rnn_classifier_cheese_disease(cheese_disease_dataset):
    mod = torch_rnn_classifier.TorchRNNClassifier(
        vocab=cheese_disease_dataset['vocab'],
        embed_dim=20,
        hidden_dim=20,
        max_iter=20)
    mod.fit(cheese_disease_dataset['X_train'], cheese_disease_dataset['y_train'])
    pred = mod.predict(cheese_disease_dataset['X_test'])
    assert accuracy_score(cheese_disease_dataset['y_test'], pred) > 0.80


@pytest.mark.parametrize("initial_embedding, use_embedding",[
    [True, False],
    [True, True],
    [False, False],
    [False, True]
])
def test_torch_rnn_classifier_simple_example(initial_embedding, use_embedding):
    torch_rnn_classifier.simple_example(initial_embedding)


def test_tf_rnn_classifier(X_sequence):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    train, test, vocab = X_sequence
    mod = tf_rnn_classifier.TfRNNClassifier(
        vocab=vocab, max_iter=100)
    X, y = zip(*train)
    X_test, _ = zip(*test)
    mod.fit(X, y)
    mod.predict(X_test)
    mod.predict_proba(X_test)


def test_tf_rnn_classifier_cheese_disease(cheese_disease_dataset):
    mod = tf_rnn_classifier.TfRNNClassifier(
        vocab=cheese_disease_dataset['vocab'],
        embed_dim=20,
        hidden_dim=20,
        max_iter=20)
    mod.fit(cheese_disease_dataset['X_train'], cheese_disease_dataset['y_train'])
    pred = mod.predict(cheese_disease_dataset['X_test'])
    assert accuracy_score(cheese_disease_dataset['y_test'], pred) > 0.80


@pytest.mark.parametrize("initial_embedding", [True, False])
def test_tf_rnn_classifier_simple_example(initial_embedding):
    tf_rnn_classifier.simple_example(initial_embedding)


@pytest.mark.parametrize("pandas", [True, False])
def test_np_autoencoder(pandas):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X = utils.randmatrix(20, 50)
    if pandas:
        X = pd.DataFrame(X)
    ae = np_autoencoder.Autoencoder(hidden_dim=5, max_iter=100)
    H = ae.fit(X)
    ae.predict(X)
    H_is_pandas = isinstance(H, pd.DataFrame)
    assert H_is_pandas == pandas


@pytest.mark.parametrize("pandas", [True, False])
def test_torch_autoencoder(pandas):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X = utils.randmatrix(20, 50)
    if pandas:
        X = pd.DataFrame(X)
    ae = torch_autoencoder.TorchAutoencoder(hidden_dim=5, max_iter=100)
    H = ae.fit(X)
    ae.predict(X)
    H_is_pandas = isinstance(H, pd.DataFrame)
    assert H_is_pandas == pandas

@pytest.mark.parametrize("pandas", [True, False])
def test_tf_autoencoder(pandas):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X = utils.randmatrix(20, 50)
    if pandas:
        X = pd.DataFrame(X)
    ae = tf_autoencoder.TfAutoencoder(hidden_dim=5, max_iter=100)
    H = ae.fit(X)
    ae.predict(X)
    H_is_pandas = isinstance(H, pd.DataFrame)
    assert H_is_pandas == pandas


def test_np_autoencoder_simple_example():
    mse = np_autoencoder.simple_example()
    assert mse < 0.003


def test_torch_autoencoder_simple_example():
    mse = torch_autoencoder.simple_example()
    assert mse < 0.0001

def test_tf_autoencoder_simple_example():
    mse = tf_autoencoder.simple_example()
    assert mse < 0.0001


def test_np_tree_nn_simple_example():
    np_tree_nn.simple_example()


@pytest.mark.parametrize("initial_embedding", [True, False])
def test_torch_tree_nn_simple_example(initial_embedding):
    torch_tree_nn.simple_example(initial_embedding)


def test_torch_tree_nn_incremental(X_tree):
    X, vocab = X_tree
    model = torch_tree_nn.TorchTreeNN(
        vocab,
        embed_dim=50,
        hidden_dim=50,
        max_iter=100,
        embedding=None)
    model.fit(X, X_dev=X, dev_iter=20)
    epochs = list(model.dev_predictions.keys())
    assert epochs == list(range(20, 101, 20))
    assert all(len(v)==len(X) for v in model.dev_predictions.values())


def test_torch_subtree_nn_simple_example():
    torch_subtree_nn.simple_example()


@pytest.mark.parametrize("tree", [
    "(special 1)",
    "(special 2)",
    "(special (odd 1))",
    "(special (even 2))",
    "(special (odd 1) (neutral (neutral +) (odd 1)))"
])
def test_subtree_supervision_root_label_alignment(tree):
    tree = Tree.fromstring(tree)
    model = torch_subtree_nn.TorchSubtreeNNModel(
        vocab=["1", "+", "2", "$UNK"],
        embed_dim=5,
        embedding=None,
        output_dim=2,
        hidden_activation=nn.Tanh())
    reps, labels = model.interpret(tree, reps=[], labels=[])
    assert labels[-1] == 'special'


def test_sgd_classifier():
    acc = np_sgd_classifier.simple_example()
    assert acc >= 0.89


@pytest.mark.parametrize("model, params", [
    [
        np_sgd_classifier.BasicSGDClassifier(max_iter=10, eta=0.1),
        {'max_iter': 100, 'eta': 1.0}
    ],
    [
        np_rnn_classifier.RNNClassifier(
            vocab=[], max_iter=10, hidden_dim=5, eta=0.1),
        {'hidden_dim': 10, 'eta': 1.0, 'max_iter': 100}
    ],
    [
        torch_rnn_classifier.TorchRNNClassifier(
            vocab=[], max_iter=10, hidden_dim=5, eta=0.1),
        {
            'hidden_dim': 10,
            'eta': 1.0,
            'max_iter': 100,
            'l2_strength': 0.01,
            'embed_dim': 100,
            'bidirectional': False
        }
    ],
    [
        tf_rnn_classifier.TfRNNClassifier(
            vocab=[], max_iter=10, hidden_dim=5, eta=0.1),
        {'hidden_dim': 10, 'eta': 1.0, 'max_iter': 100}
    ],
    [
        np_tree_nn.TreeNN(
            vocab=[], max_iter=10, hidden_dim=5, eta=0.1),
        {'embed_dim': 5, 'hidden_dim': 10, 'eta': 1.0, 'max_iter': 100}
    ],
    [
        torch_tree_nn.TorchTreeNN(
            vocab=[], max_iter=10, hidden_dim=5, eta=0.1),
        {
            'embed_dim': 5,
            'hidden_dim': 10,
            'hidden_activation': nn.ReLU(),
            'eta': 1.0,
            'max_iter': 100,
            'l2_strength': 0.01
        }
    ],
    [
        np_shallow_neural_classifier.ShallowNeuralClassifier(
            hidden_dim=5, max_iter=1, eta=1.0),
        {
            'hidden_dim': 10,
            # Reset to ReLU:
            'hidden_activation': lambda z: np.maximum(0, z),
            'd_hidden_activation': lambda z: np.where(z > 0, 1, 0),
            'max_iter': 10,
            'eta': 0.1
        }
    ],
    [
        torch_shallow_neural_classifier.TorchShallowNeuralClassifier(
            hidden_dim=5, hidden_activation=nn.ReLU(), max_iter=1, eta=1.0),
        {
            'hidden_dim': 10,
            'hidden_activation': nn.ReLU(),
            'max_iter': 10,
            'eta': 0.1
        }
    ],
    [
        np_autoencoder.Autoencoder(hidden_dim=5, max_iter=1, eta=1.0),
        {'hidden_dim': 10, 'max_iter': 10, 'eta': 0.1}
    ],
    [
        torch_autoencoder.TorchAutoencoder(
            hidden_dim=5, hidden_activation=nn.ReLU(), max_iter=1, eta=1.0),
        {
            'hidden_dim': 10,
            'hidden_activation': nn.ReLU(),
            'max_iter': 10,
            'eta': 0.1
        }
    ],
    [
        tf_shallow_neural_classifier.TfShallowNeuralClassifier(
            hidden_dim=5, hidden_activation=tf.nn.tanh, max_iter=1, eta=1.0),
        {
            'hidden_dim': 10,
            'hidden_activation': tf.nn.relu,
            'max_iter': 10,
            'eta': 0.1
        }
    ],
    [
        tf_autoencoder.TfAutoencoder(
            hidden_dim=5, hidden_activation=tf.nn.tanh, max_iter=1, eta=1.0),
        {
            'hidden_dim': 10,
            'hidden_activation': tf.nn.relu,
            'max_iter': 10,
            'eta': 0.1
        }
    ]
])
def test_parameter_setting(model, params):
    model.set_params(**params)
    for p, val in params.items():
        assert getattr(model, p) == val


@pytest.mark.parametrize("model_class", [
    np_rnn_classifier.RNNClassifier,
    torch_rnn_classifier.TorchRNNClassifier,
    tf_rnn_classifier.TfRNNClassifier
])
def test_rnn_classifier_cross_validation(model_class, X_sequence):
    train, test, vocab = X_sequence
    mod = model_class(vocab, max_iter=2)
    X, y = zip(*train)
    best_mod = utils.fit_classifier_with_crossvalidation(
        X, y, mod, cv=2, param_grid={'hidden_dim': [10, 20]})
