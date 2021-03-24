from nltk.tree import Tree
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate
import tempfile
import torch
import torch.nn as nn
import utils

from test_torch_model_base import PARAMS_WITH_TEST_VALUES as BASE_PARAMS
from torch_tree_nn import TorchTreeNN
from torch_tree_nn import simple_example
from np_tree_nn import TreeNN
from np_tree_nn import simple_example as np_simple_example

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


PARAMS_WITH_TEST_VALUES = [
    ["embed_dim", 10],
    ["embedding", utils.randmatrix(4, 10)],
    ["hidden_activation", nn.ReLU()],
    ['freeze_embedding', True]]


PARAMS_WITH_TEST_VALUES += BASE_PARAMS


@pytest.fixture
def dataset():
    vocab = ["1", "+", "2", "$UNK"]

    train = [
        "(odd 1)",
        "(even 2)",
        "(even (odd 1) (neutral (neutral +) (odd 1)))",
        "(odd (odd 1) (neutral (neutral +) (even 2)))",
        "(odd (even 2) (neutral (neutral +) (odd 1)))",
        "(even (even 2) (neutral (neutral +) (even 2)))",
        "(even (odd 1) (neutral (neutral +) (odd (odd 1) (neutral (neutral +) (even 2)))))"]

    test = [
        "(odd (odd 1))",
        "(even (even 2))",
        "(odd (odd 1) (neutral (neutral +) (even (odd 1) (neutral (neutral +) (odd 1)))))",
        "(even (even 2) (neutral (neutral +) (even (even 2) (neutral (neutral +) (even 2)))))",
        "(odd (even 2) (neutral (neutral +) (odd (even 2) (neutral (neutral +) (odd 1)))))",
        "(even (odd 1) (neutral (neutral +) (odd (even 2) (neutral (neutral +) (odd 1)))))",
        "(odd (even 2) (neutral (neutral +) (odd (odd 1) (neutral (neutral +) (even 2)))))"]

    X_train = [Tree.fromstring(x) for x in train]
    y_train = [t.label() for t in X_train]

    X_test = [Tree.fromstring(x) for x in test]
    y_test = [t.label() for t in X_test]

    return X_train, X_test, y_train, y_test, vocab


def test_simple_example():
    acc = simple_example()
    assert acc >= 4/7


def test_np_simple_example():
    acc = np_simple_example()
    assert acc >= 4/7


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_params(param, expected):
    vocab = []
    mod = TorchTreeNN(vocab, **{param: expected})
    result = getattr(mod, param)
    if param == "embedding":
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_simple_example_params(dataset, param, expected):
    X_train, X_test, y_train, y_test, vocab = dataset
    model = TorchTreeNN(vocab, **{param: expected})
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert accuracy_score(y_test, preds)


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_parameter_setting(param, expected):
    vocab = []
    mod = TorchTreeNN(vocab)
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    if param == "embedding":
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize("param, value", [
    ['embed_dim', 5],
    ['eta', 1.0],
    ["embedding", np.ones((10,10))],
    ['max_iter', 100]
])
def test_np_parameter_setting(param, value):
    vocab = []
    mod = TreeNN(vocab)
    mod.set_params(**{param:value})
    result = getattr(mod, param)
    if param == "embedding":
        assert np.array_equal(result, value)
    else:
        assert result == value


def test_np_set_embed_dim():
    value = 26
    vocab = []
    mod = TreeNN(vocab, embed_dim=5)
    mod.embed_dim = value
    assert mod.embedding.shape[1] == value


@pytest.mark.parametrize("with_y, expected", [
    [True, 4],
    [False, 3]
])
def test_build_dataset(dataset, with_y, expected):
    X_train, X_test, y_train, y_test, vocab = dataset
    mod = TorchTreeNN(vocab)
    if with_y:
        dataset = mod.build_dataset(X_train, y_train)
    else:
        dataset = mod.build_dataset(X_train)
    result = next(iter(dataset))
    assert len(result) == expected


@pytest.mark.parametrize("mod_attr, graph_attr, dim", [
    ["embed_dim", "tree_layer", 0],
    ["n_classes_", "classifier_layer", 0]
])
def test_model_graph_dimensions(dataset, mod_attr, dim, graph_attr):
    X_train, X_test, y_train, y_test, vocab = dataset
    mod = TorchTreeNN(vocab, max_iter=1)
    mod.fit(X_train, y_train)
    mod_attr_val = getattr(mod, mod_attr)
    graph_attr_val = getattr(mod.model, graph_attr).weight.shape[dim]
    assert mod_attr_val == graph_attr_val


def test_pretrained_embedding(dataset):
    X_train, X_test, y_train, y_test, vocab = dataset
    embed_dim = 5
    embedding = np.ones((len(vocab), embed_dim))
    mod = TorchTreeNN(
        vocab,
        max_iter=1,
        embedding=embedding,
        freeze_embedding=True)
    mod.fit(X_train, y_train)
    graph_emb = mod.model.embedding.weight.detach().cpu().numpy()
    assert np.array_equal(embedding, graph_emb)


def test_np_pretrained_embedding(dataset):
    X_train, X_test, y_train, y_test, vocab = dataset
    embed_dim = 5
    embedding = np.ones((len(vocab), embed_dim))
    mod = TreeNN(
        vocab,
        max_iter=1,
        embedding=embedding)
    mod.fit(X_train, y_train)
    graph_emb = mod.embedding
    assert np.array_equal(embedding, graph_emb)


@pytest.mark.parametrize("model_class", [
    TorchTreeNN,
    TreeNN
])
def test_predict_proba(dataset, model_class):
    X_train, X_test, y_train, y_test, vocab = dataset
    mod = model_class(vocab, max_iter=1)
    mod.fit(X_train, y_train)
    probs = mod.predict_proba(X_test)
    assert all(np.round(x.sum(), 6) == 1.0 for x in probs)


@pytest.mark.parametrize("model_class", [
    TorchTreeNN,
    TreeNN
])
def test_hyperparameter_selection(dataset, model_class):
    X_train, X_test, y_train, y_test, vocab = dataset
    param_grid = {'embed_dim': [10, 20]}
    mod = model_class(vocab, max_iter=5)
    xval = RandomizedSearchCV(mod, param_grid, cv=2)
    xval.fit(X_train, y_train)


@pytest.mark.parametrize("model_class", [
    TorchTreeNN,
    TreeNN
])
def test_cross_validation_sklearn(dataset, model_class):
    X_train, X_test, y_train, y_test, vocab = dataset
    mod = model_class(vocab, max_iter=5)
    xval = cross_validate(mod, X_train, y_train, cv=2)


@pytest.mark.parametrize("model_class", [
    TorchTreeNN,
    TreeNN
])
def test_cross_validation_nlu(dataset, model_class):
    X_train, X_test, y_train, y_test, vocab = dataset
    param_grid={'embed_dim': [10, 20]}
    mod = model_class(vocab, max_iter=2)
    best_mod = utils.fit_classifier_with_hyperparameter_search(
        X_train, y_train, mod, cv=2, param_grid=param_grid)


def test_torch_tree_nn_save_load(dataset):
    X_train, X_test, y_train, y_test, vocab = dataset
    mod = TorchTreeNN(
        vocab,
        embed_dim=50,
        max_iter=100,
        embedding=None)
    mod.fit(X_train, y_train)
    mod.predict(X_test)
    with tempfile.NamedTemporaryFile(mode='wb') as f:
        name = f.name
        mod.to_pickle(name)
        mod2 = TorchTreeNN.from_pickle(name)
        mod2.predict(X_test)
        mod2.fit(X_test, y_test)


@pytest.mark.parametrize("tree, subtree_indices, emb_indices, n", [
    ["(0 1)", [[0, 0, 0]], [1], 0],
    ["(0 (1 1))", [[0, 0, 0]], [1], 0],
    ["(0 (1 1) (2 2))", [[0, 1, 2], [1, 1, 1], [2, 2, 2]], [False, 1, 2], 2],
    [
        "(0 (1 (2 2) (3 3)) (4 (5 5) (6 6)))",
        [[0, 1, 4], [1, 2, 3], [2, 2, 2], [3, 3, 3], [4, 5, 6], [5, 5, 5], [6, 6, 6]],
        [False, False, 2, 3, False, 5, 6],
        6
    ]
])
def test_build_tree_rep(tree, subtree_indices, emb_indices, n):
    tree = Tree.fromstring(tree)
    vocab = ["0", "1", "2", "3","4", "5", "6", "$UNK"]
    mod = TorchTreeNN(vocab)
    result = mod._build_tree_rep(tree)
    assert result[0] == subtree_indices
    assert result[1] == emb_indices
    assert result[2] == n


@pytest.mark.parametrize("func", ["predict", "predict_proba"])
def test_predict_functions_honor_device(dataset, func):
    X_train, X_test, y_train, y_test, vocab = dataset
    mod = TorchTreeNN(vocab, max_iter=2)
    mod.fit(X_train, y_train)
    prediction_func = getattr(mod, func)
    with pytest.raises(RuntimeError):
        prediction_func(X_test, device="FAKE_DEVICE")


@pytest.mark.parametrize("func", ["predict", "predict_proba"])
def test_predict_restores_device(dataset, func):
    X_train, X_test, y_train, y_test, vocab = dataset
    mod = TorchTreeNN(vocab, max_iter=2)
    mod.fit(X_train, y_train)
    current_device = mod.device
    assert current_device != torch.device("cpu:0")
    prediction_func = getattr(mod, func)
    prediction_func(X_test, device="cpu:0")
    assert mod.device == current_device
