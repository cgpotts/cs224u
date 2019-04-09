import numpy as np
import os
import pytest
import random
import tensorflow as tf
import torch
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"

tf.enable_eager_execution()


@pytest.mark.parametrize("arg, expected", [
    [
        np.array([0.0, 0.25, 0.75]),
        np.array([0.22721977, 0.29175596, 0.48102426])
    ]
])
def test_softmax(arg, expected):
    result = utils.softmax(arg).round(8)
    expected = expected.round(8)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("arg, expected", [
    [-1, 0],
    [np.array([-1.0, 1.0]), np.array([0.0, 0.0])]
])
def test_d_tanh(arg, expected):
    assert np.array_equal(utils.d_tanh(arg), expected)


def test_randvec():
    x = utils.randvec(10)
    assert len(x) == 10


def test_randmatrix():
    X = utils.randmatrix(10, 20)
    assert X.shape == (10, 20)


def test_safe_macro_f1():
    y = [1, 1, 2, 2, 1]
    y_pred = [1, 2, 2, 1, 1]
    utils.safe_macro_f1(y, y_pred)

@pytest.mark.parametrize("arg, expected", [
    [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]])
    ]
])
def test_log_of_array_ignoring_zeros(arg, expected):
    result = utils.log_of_array_ignoring_zeros(arg)
    return np.array_equal(result, expected)


def test_glove2dict():
    src_filename = os.path.join("data", "glove.6B", "glove.6B.50d.txt")
    data = utils.glove2dict(src_filename)
    assert len(data) == 400000

@pytest.mark.parametrize("X, n_words, expected", [
    [
        [["a", "b", "c"], ["b", "c", "d"]],
        None,
        ["$UNK", "a", "b", "c", "d"]
    ],
    [
        [["a", "b", "c"], ["b", "c", "d"]],
        2,
        ["$UNK", "b", "c"]
    ],
    [
        [],
        2,
        ["$UNK"]
    ]
])
def test_get_vocab(X, n_words, expected):
    result = utils.get_vocab(X, n_words=n_words)
    assert result == expected


@pytest.mark.parametrize("set_value", [True, False])
def test_fix_random_seeds_system(set_value):
    utils.fix_random_seeds(seed=42, set_system=set_value)
    x = np.random.random()
    utils.fix_random_seeds(seed=42, set_system=set_value)
    y = np.random.random()
    assert (x == y) == set_value


@pytest.mark.parametrize("set_value", [True, False])
def test_fix_random_seeds_pytorch(set_value):
    utils.fix_random_seeds(seed=42, set_torch=set_value)
    x = torch.rand(1)
    utils.fix_random_seeds(seed=42, set_torch=set_value)
    y = torch.rand(1)
    assert (x == y) == set_value


@pytest.mark.parametrize("set_value", [True, False])
def test_fix_random_seeds_tensorflow(set_value):
    utils.fix_random_seeds(seed=42, set_tensorflow=set_value)
    x = tf.random.uniform([1]).numpy()
    utils.fix_random_seeds(seed=42, set_tensorflow=set_value)
    y = tf.random.uniform([1]).numpy()
    assert (x == y) == set_value
