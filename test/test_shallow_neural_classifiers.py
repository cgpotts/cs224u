import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.model_selection import train_test_split
import tempfile
import torch
import torch.nn as nn
import utils

from test_torch_model_base import PARAMS_WITH_TEST_VALUES as BASE_PARAMS
from np_shallow_neural_classifier import ShallowNeuralClassifier
from np_shallow_neural_classifier import simple_example as np_simple_example
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_shallow_neural_classifier import simple_example

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


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
def digits():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


PARAMS_WITH_TEST_VALUES = [
    ["hidden_dim", 10],
    ["hidden_activation", nn.ReLU()]]


PARAMS_WITH_TEST_VALUES += BASE_PARAMS


def test_model(XOR):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X, y = XOR
    model = TorchShallowNeuralClassifier(
        hidden_dim=4,
        hidden_activation=nn.ReLU(),
        max_iter=100,
        eta=0.01)
    model.fit(X, y)
    model.predict(X)
    model.predict_proba(X)


def test_np_model(XOR):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X, y = XOR
    model = ShallowNeuralClassifier(
        hidden_dim=4,
        hidden_activation=np.tanh,
        d_hidden_activation=utils.d_tanh,
        eta=0.05,
        tol=1.5e-8,
        display_progress=True,
        max_iter=100)
    model.fit(X, y)
    model.predict(X)


def test_simple_example():
    acc = simple_example()
    assert acc >= 0.95


def test_np_simple_example():
    acc = np_simple_example()
    assert acc >= 0.88


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_params(param, expected):
    mod = TorchShallowNeuralClassifier(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_simple_example_params(digits, param, expected):
    X_train, X_test, y_train, y_test = digits
    mod = TorchShallowNeuralClassifier(**{param: expected})
    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if not (param in ["max_iter", "batch_size"] and expected <= 1):
        assert acc >= 0.86


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_parameter_setting(param, expected):
    mod = TorchShallowNeuralClassifier()
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


@pytest.mark.parametrize("expected", [True, False])
def test_optimizer_keywords(XOR, expected):
    X, y = XOR
    mod = TorchShallowNeuralClassifier(amsgrad=expected)
    mod.fit(X, y)
    assert mod.amsgrad == expected
    assert mod.optimizer.param_groups[0]['amsgrad'] == expected


@pytest.mark.parametrize("param, value", [
    ['hidden_dim', 10],
    ['hidden_activation', lambda z: np.maximum(0, z)],
    ['d_hidden_activation', lambda z: np.where(z > 0, 1, 0)],
    ['max_iter', 10],
    ['eta', 0.1]
])
def test_np_parameter_setting(param, value):
    mod = ShallowNeuralClassifier()
    mod.set_params(**{param:value})
    assert getattr(mod, param) == value


@pytest.mark.parametrize("with_y, expected", [
    [True, 2],
    [False, 1]
])
def test_build_dataset(digits, with_y, expected):
    X_train, X_test, y_train, y_test = digits
    mod = TorchShallowNeuralClassifier()
    if with_y:
        dataset = mod.build_dataset(X_train, y_train)
    else:
        dataset = mod.build_dataset(X_train)
    result = next(iter(dataset))
    assert len(result) == expected


@pytest.mark.parametrize("attr, layer_index, weight_dim", [
    ["hidden_dim", 0, 0], # We write xW; PyTorch does Wx^T
    ["hidden_dim", 2, 1],
    ["input_dim", 0, 1],
    ["n_classes_", 2, 0]
])
def test_model_graph_dimensions(digits, attr, layer_index, weight_dim):
    X_train, X_test, y_train, y_test = digits
    mod = TorchShallowNeuralClassifier(max_iter=1)
    mod.fit(X_train, y_train)
    mod_attr_val = getattr(mod, attr)
    graph_dim = mod.model[layer_index].weight.shape[weight_dim]
    assert mod_attr_val == graph_dim


def test_hidden_activation_in_graph(digits):
    X_train, X_test, y_train, y_test = digits
    mod = TorchShallowNeuralClassifier(
        max_iter=1, hidden_activation=nn.ReLU())
    mod.fit(X_train, y_train)
    mod_hidden_activation = mod.hidden_activation.__class__
    graph_activation_class = mod.model[1].__class__
    assert mod_hidden_activation == graph_activation_class


@pytest.mark.parametrize("early_stopping", [True, False])
def test_build_dataset_input_dim(digits, early_stopping):
    X_train, X_test, y_train, y_test = digits
    mod = TorchShallowNeuralClassifier(early_stopping=early_stopping)
    dataset = mod.build_dataset(X_train, y_train)
    assert mod.input_dim == X_train.shape[1]


@pytest.mark.parametrize("model_class", [
    TorchShallowNeuralClassifier,
    ShallowNeuralClassifier
])
def test_predict_proba(digits, model_class):
    X_train, X_test, y_train, y_test = digits
    mod = model_class(max_iter=1)
    mod.fit(X_train, y_train)
    probs = mod.predict_proba(X_test)
    assert all(np.round(x.sum(), 6) == 1.0 for x in probs)


@pytest.mark.parametrize("model_class", [
    TorchShallowNeuralClassifier,
    ShallowNeuralClassifier
])
def test_hyperparameter_selection(digits, model_class):
    X_train, X_test, y_train, y_test = digits
    param_grid = {'hidden_dim': [10, 20]}
    mod = model_class(max_iter=5)
    xval = RandomizedSearchCV(mod, param_grid, cv=2)
    xval.fit(X_train, y_train)


@pytest.mark.parametrize("model_class", [
    TorchShallowNeuralClassifier,
    ShallowNeuralClassifier
])
def test_cross_validation_sklearn(digits, model_class):
    X_train, X_test, y_train, y_test = digits
    mod = model_class(max_iter=5)
    xval = cross_validate(mod, X_train, y_train, cv=2)


@pytest.mark.parametrize("model_class", [
    TorchShallowNeuralClassifier,
    ShallowNeuralClassifier
])
def test_cross_validation_nlu(digits, model_class):
    X_train, X_test, y_train, y_test = digits
    param_grid = {'hidden_dim': [10, 20]}
    mod = model_class(max_iter=2)
    best_mod = utils.fit_classifier_with_hyperparameter_search(
        X_train, y_train, mod, cv=2, param_grid=param_grid)


def test_save_load(XOR):
    X, y = XOR
    mod = TorchShallowNeuralClassifier(
        hidden_dim=4,
        hidden_activation=nn.ReLU(),
        max_iter=100,
        eta=0.01)
    mod.fit(X, y)
    mod.predict(X)
    with tempfile.NamedTemporaryFile(mode='wb') as f:
        name = f.name
        mod.to_pickle(name)
        mod2 = TorchShallowNeuralClassifier.from_pickle(name)
        mod2.predict(X)
        mod2.fit(X, y)


@pytest.mark.parametrize("func", ["predict", "predict_proba"])
def test_predict_functions_honor_device(digits, func):
    X_train, X_test, y_train, y_test = digits
    mod = TorchShallowNeuralClassifier(max_iter=2)
    mod.fit(X_train, y_train)
    prediction_func = getattr(mod, func)
    with pytest.raises(RuntimeError):
        prediction_func(X_test, device="FAKE_DEVICE")


@pytest.mark.parametrize("func", ["predict", "predict_proba"])
def test_predict_restores_device(digits, func):
    X_train, X_test, y_train, y_test = digits
    mod = TorchShallowNeuralClassifier(max_iter=2)
    mod.fit(X_train, y_train)
    current_device = mod.device
    assert current_device != torch.device("cpu:0")
    prediction_func = getattr(mod, func)
    prediction_func(X_test, device="cpu:0")
    assert mod.device == current_device
