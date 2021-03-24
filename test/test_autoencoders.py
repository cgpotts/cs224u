import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RandomizedSearchCV, cross_validate
import tempfile
import torch
import torch.nn as nn
import utils

from test_torch_model_base import PARAMS_WITH_TEST_VALUES as BASE_PARAMS
from torch_autoencoder import TorchAutoencoder, simple_example
from np_autoencoder import Autoencoder
from np_autoencoder import simple_example as np_simple_example

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


PARAMS_WITH_TEST_VALUES = [
    ["hidden_dim", 10],
    ["hidden_activation", nn.ReLU()]]


PARAMS_WITH_TEST_VALUES += BASE_PARAMS


@pytest.fixture
def random_matrix():
    return utils.randmatrix(20, 50)


@pytest.fixture
def random_low_rank_matrix():

    def randmatrix(m, n, sigma=0.1, mu=0):
        return sigma * np.random.randn(m, n) + mu

    rank = 20
    nrow = 2000
    ncol = 100

    X = randmatrix(nrow, rank).dot(randmatrix(rank, ncol))
    return X


@pytest.mark.parametrize("model_class, pandas", [
    [TorchAutoencoder, True],
    [TorchAutoencoder, False],
    [Autoencoder, True],
    [Autoencoder, False]
])
def test_model(random_matrix, model_class, pandas):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X = random_matrix
    if pandas:
        X = pd.DataFrame(X)
    ae = TorchAutoencoder(hidden_dim=5, max_iter=100)
    H = ae.fit(X)
    ae.predict(X)
    H_is_pandas = isinstance(H, pd.DataFrame)
    assert H_is_pandas == pandas


def test_simple_example():
    r2 = simple_example()
    assert r2 > 0.92


def test_np_simple_example():
    r2 = np_simple_example()
    assert r2 > 0.92


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_params(param, expected):
    mod = TorchAutoencoder(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_simple_example_params(random_low_rank_matrix, param, expected):
    X = random_low_rank_matrix
    ae = TorchAutoencoder()
    H = ae.fit(X)
    X_pred = ae.predict(X)
    r2 = ae.score(X)
    assert r2 > 0.92


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_parameter_setting(param, expected):
    mod = TorchAutoencoder()
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


@pytest.mark.parametrize("param, expected", [
    ['hidden_dim', 10],
    ['max_iter', 10],
    ['eta', 0.1]
])
def test_np_parameter_setting(param, expected):
    mod = Autoencoder()
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


@pytest.mark.parametrize("with_y, expected", [
    [True, 2],
    [False, 1]
])
def test_build_dataset(random_matrix, with_y, expected):
    X = random_matrix
    mod = TorchAutoencoder()
    if with_y:
        dataset = mod.build_dataset(X, X)
    else:
        dataset = mod.build_dataset(X)
    result = next(iter(dataset))
    assert len(result) == expected


@pytest.mark.parametrize("attr, layer_index, weight_dim", [
    ["hidden_dim", 0, 0], # We write xW; PyTorch does Wx^T
    ["hidden_dim", 2, 1],
    ["input_dim", 0, 1],
    ["output_dim", 2, 0]
])
def test_model_graph_dimensions(random_matrix, attr, layer_index, weight_dim):
    X = random_matrix
    mod = TorchAutoencoder(max_iter=1)
    mod.fit(X)
    mod_attr_val = getattr(mod, attr)
    graph_dim = mod.model[layer_index].weight.shape[weight_dim]
    assert mod_attr_val == graph_dim


def test_hidden_activation_in_graph(random_matrix):
    X = random_matrix
    mod = TorchAutoencoder(max_iter=1, hidden_activation=nn.ReLU())
    mod.fit(X)
    mod_hidden_activation = mod.hidden_activation.__class__
    graph_activation_class = mod.model[1].__class__
    assert mod_hidden_activation == graph_activation_class


@pytest.mark.parametrize("early_stopping", [True, False])
def test_build_dataset_input_dim(random_matrix, early_stopping):
    X = random_matrix
    mod = TorchAutoencoder(early_stopping=early_stopping)
    dataset = mod.build_dataset(X)
    assert mod.input_dim == X.shape[1]


@pytest.mark.parametrize("model_class", [
    TorchAutoencoder,
    Autoencoder
])
def test_hyperparameter_selection(random_matrix, model_class):
    X = random_matrix
    param_grid = {'hidden_dim': [10, 20]}
    mod = model_class(max_iter=5)
    xval = RandomizedSearchCV(mod, param_grid, cv=2)
    xval.fit(X)


@pytest.mark.parametrize("model_class", [
    TorchAutoencoder,
    Autoencoder
])
def test_cross_validation_sklearn(random_matrix, model_class):
    X = random_matrix
    mod = model_class(max_iter=5)
    xval = cross_validate(mod, X, cv=2)


@pytest.mark.parametrize("func", ["predict", "score"])
def test_predict_functions_honor_device(random_matrix, func):
    X = random_matrix
    mod = TorchAutoencoder(hidden_dim=5, max_iter=2)
    mod.fit(X)
    prediction_func = getattr(mod, func)
    with pytest.raises(RuntimeError):
        prediction_func(X, device="FAKE_DEVICE")


@pytest.mark.parametrize("func", ["predict", "score"])
def test_predict_functions_restore_device(random_matrix, func):
    X = random_matrix
    mod = TorchAutoencoder(hidden_dim=5, max_iter=2)
    mod.fit(X)
    current_device = mod.device
    assert current_device != torch.device("cpu:0")
    prediction_func = getattr(mod, func)
    prediction_func(X, device="cpu:0")
    assert mod.device == current_device


def test_save_load(random_matrix):
    X = random_matrix
    mod = TorchAutoencoder(hidden_dim=5, max_iter=2)
    mod.fit(X)
    mod.predict(X)
    with tempfile.NamedTemporaryFile(mode='wb') as f:
        name = f.name
        mod.to_pickle(name)
        mod2 = TorchAutoencoder.from_pickle(name)
        mod2.predict(X)
        mod2.fit(X)
