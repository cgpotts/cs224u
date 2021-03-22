import copy
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import utils

from torch_model_base import TorchModelBase

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


PARAMS_WITH_TEST_VALUES = [
    ["batch_size", 150],
    ["batch_size", 1],
    ["max_iter", 0],
    ["max_iter", 250],
    ["eta", 0.02],
    ["optimizer_class", torch.optim.Adagrad],
    ["l2_strength", 0.01],
    ["gradient_accumulation_steps", 1],
    ["gradient_accumulation_steps", 5],
    ["max_grad_norm", 1.0],
    ["warm_start", True],
    ["early_stopping", True],
    ["validation_fraction", 0.12],
    ["n_iter_no_change", 11],
    ["tol", 0.0001]]


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


class SoftmaxClassifier(TorchShallowNeuralClassifier):
    def build_graph(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.n_classes_))


def test_softmax_classifier_subclass(digits):
    X_train, X_test, y_train, y_test = digits
    mod = SoftmaxClassifier()
    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    acc = accuracy_score(y_test, preds)


@pytest.mark.parametrize("expected", [True, False])
def test_optimizer_keywords(XOR, expected):
    X, y = XOR
    mod = SoftmaxClassifier(amsgrad=expected)
    mod.fit(X, y)
    assert mod.amsgrad == expected
    assert mod.optimizer.param_groups[0]['amsgrad'] == expected


@pytest.mark.parametrize("arg_count", [1, 2, 3, 4, 5])
def test_build_validation_split(arg_count):
    n_features = 2
    n_examples = 10
    validation_fraction = 0.2
    expected_dev_size = int(n_examples * validation_fraction)
    expected_train_size = n_examples - expected_dev_size
    args = [np.ones((n_examples, n_features)) for _ in range(arg_count)]
    train, dev = TorchModelBase._build_validation_split(
        *args,
        validation_fraction=validation_fraction)
    assert len(train) == arg_count
    assert len(dev) == arg_count
    assert all(x.shape == (expected_train_size, n_features) for x in train)
    assert all(x.shape == (expected_dev_size, n_features) for x in dev)


@pytest.mark.parametrize("epoch_error, expected", [
    [0.75, 6],
    [0.50, 0],
    [0.25, 0]
])
def test_update_no_improvement_count_errors(epoch_error, expected):
    mod = TorchModelBase(tol=0.5)
    mod.no_improvement_count = 5
    mod.best_error = 1
    mod.errors = []
    mod._update_no_improvement_count_errors(epoch_error)
    assert mod.no_improvement_count == expected


def test_early_stopping(digits):
    X_train, X_test, y_train, y_test = digits
    mod = SoftmaxClassifier(max_iter=100, warm_start=True, early_stopping=True)
    # This fit call should lead to a good model:
    mod.fit(X_train, y_train)
    # Store the best model params:
    best_parameters = copy.deepcopy(mod.model.state_dict())
    # Reset the graph and train a little more:
    mod.model = mod.build_graph()
    mod.max_iter = 1
    mod.fit(X_train, y_train)
    # Make sure the best parameters are still present:
    for key, X in mod.model.state_dict().items():
        assert torch.all(X.eq(best_parameters[key]))


def test_update_no_improvement_count_early_stopping_best_model_used(digits):
    X_train, X_test, y_train, y_test = digits
    mod = SoftmaxClassifier(max_iter=100, warm_start=True, early_stopping=False)
    # This fit call should lead to a good model:
    mod.fit(X_train, y_train)
    # One relevant function call; this should store our good model:
    mod._update_no_improvement_count_early_stopping(X_test, y_test)
    # This will reset the actual graph to random:
    mod.model = mod.build_graph()
    mod.model.to(mod.device)
    # Make sure the best parameters are still present:
    for key, X in mod.model.state_dict().items():
        assert not torch.all(X.eq(mod.best_parameters[key]))


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_parameter_setting(param, expected):
    mod = TorchModelBase(amsgrad=0.5)
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


def test_no_setting_of_missing_param():
    mod = TorchModelBase(amsgrad=0.5)
    with pytest.raises(ValueError):
        mod.set_params(**{'NON_EXISTENT_PARAM': False})
