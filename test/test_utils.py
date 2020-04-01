import numpy as np
import os
import pytest
import random
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


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

@pytest.mark.parametrize("X, n_words, mincount, expected", [
    [
        [["a", "b", "c"], ["b", "c", "d"]],
        None,
        1,
        ["$UNK", "a", "b", "c", "d"]
    ],
    [
        [["a", "b", "c"], ["b", "c", "d"]],
        2,
        1,
        ["$UNK", "b", "c"]
    ],
    [
        [],
        2,
        1,
        ["$UNK"]
    ],
    [
        [["a", "b", "b"], ["b", "c", "a"]],
        None,
        3,
        ["$UNK", "b"]
    ],
    [
        [["b", "b", "b"], ["b", "a", "a", "c"]],
        2,
        3,
        ["$UNK", "b"]
    ],
])
def test_get_vocab(X, n_words, mincount, expected):
    result = utils.get_vocab(X, n_words=n_words, mincount=mincount)
    assert result == expected


@pytest.mark.parametrize("lookup, vocab, required_tokens, expected_shape", [
    [
        {"a": [1,2]}, ["a", "b"], ["$UNK"], (3,2)
    ],
    [
        {"a": [1,2], "b": [3,4]}, ["b"], ["$UNK"], (2,2)
    ]
])
def test_create_pretrained_embedding(lookup, vocab, required_tokens, expected_shape):
    result, new_vocab = utils.create_pretrained_embedding(lookup, vocab, required_tokens)
    assert result.shape == expected_shape
    assert "$UNK" in new_vocab
    new_vocab.remove("$UNK")
    assert vocab == new_vocab


@pytest.mark.parametrize("set_value", [True, False])
def test_fix_random_seeds_system(set_value):
    params = dict(
        seed=42,
        set_system=set_value,
        set_tensorflow=False,
        set_torch=False,
        set_torch_cudnn=False)
    utils.fix_random_seeds(**params)
    x = np.random.random()
    utils.fix_random_seeds(**params)
    y = np.random.random()
    assert (x == y) == set_value


@pytest.mark.parametrize("set_value", [True, False])
def test_fix_random_seeds_pytorch(set_value):
    import torch
    params = dict(
        seed=42,
        set_system=False,
        set_tensorflow=False,
        set_torch=set_value,
        set_torch_cudnn=set_value)
    utils.fix_random_seeds(**params)
    x = torch.rand(1)
    utils.fix_random_seeds(**params)
    y = torch.rand(1)
    assert (x == y) == set_value


@pytest.mark.parametrize("set_value", [True, False])
def test_fix_random_seeds_tensorflow(set_value):
    import tensorflow as tf
    params = dict(
        seed=42,
        set_system=False,
        set_tensorflow=set_value,
        set_torch=True,
        set_torch_cudnn=True)
    utils.fix_random_seeds(**params)
    x = tf.random.uniform([1]).numpy()
    utils.fix_random_seeds(**params)
    y = tf.random.uniform([1]).numpy()
    assert (x == y) == set_value
