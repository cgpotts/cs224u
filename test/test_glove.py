import numpy as np
import pandas as pd
import pytest
import tempfile
import torch.nn as nn
import utils

from test_torch_model_base import PARAMS_WITH_TEST_VALUES as BASE_PARAMS
from torch_glove import TorchGloVe, simple_example
from np_glove import GloVe

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


PARAMS_WITH_TEST_VALUES = [
    ["embed_dim", 20],
    ["alpha", 0.65],
    ["xmax", 75]]


PARAMS_WITH_TEST_VALUES += BASE_PARAMS


@pytest.fixture
def count_matrix():
    return np.array([
        [  4.,   4.,   2.,   0.],
        [  4.,  61.,   8.,  18.],
        [  2.,   8.,  10.,   0.],
        [  0.,  18.,   0.,   5.]])


@pytest.mark.parametrize("pandas", [True, False])
def test_model(count_matrix, pandas):
    X = count_matrix
    if pandas:
        X = pd.DataFrame(X)
    glove = TorchGloVe()
    G = glove.fit(X)
    G_is_pandas = isinstance(G, pd.DataFrame)
    assert G_is_pandas == pandas


def test_simple_example():
    corr = simple_example()
    assert corr > 0.43


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_params(param, expected):
    mod = TorchGloVe(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_simple_example_params(count_matrix, param, expected):
    X = count_matrix
    mod = TorchGloVe(**{param: expected})
    G = mod.fit(X)
    corr = mod.score(X)
    if not (param == "max_iter" and expected == 0):
        assert corr > 0.40


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_parameter_setting(param, expected):
    mod = TorchGloVe()
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


def test_build_dataset(count_matrix):
    X = count_matrix
    # We needn't do the actual calculation to test here:
    weights = X
    mod = TorchGloVe()
    dataset = mod.build_dataset(X, weights)
    result = next(iter(dataset))
    assert len(result) == 3


@pytest.mark.parametrize("param", ["W", "C"])
def test_model_graph_embed_dim(count_matrix, param):
    X = count_matrix
    mod = TorchGloVe(max_iter=1)
    mod.fit(X)
    mod_attr_val = mod.embed_dim
    graph_param = getattr(mod.model, param)
    graph_attr_val = graph_param.shape[1]
    assert mod_attr_val == graph_attr_val


def test_save_load(count_matrix):
    X = count_matrix
    mod = TorchGloVe(max_iter=2)
    mod.fit(X)
    with tempfile.NamedTemporaryFile(mode='wb') as f:
        name = f.name
        mod.to_pickle(name)
        mod2 = TorchGloVe.from_pickle(name)
        mod2.fit(X)


def test_np_glove(count_matrix):
    """
    Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    mod = GloVe()
    mod.fit(count_matrix)
