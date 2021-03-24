import pytest
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.model_selection import train_test_split
import utils

from np_sgd_classifier import BasicSGDClassifier
from np_sgd_classifier import simple_example

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


PARAMS_WITH_TEST_VALUES = [
    ['max_iter', 10],
    ['max_iter', 0],
    ['eta', 0.02]]


@pytest.fixture
def digits():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def test_model():
    f1 = simple_example()
    assert f1 >= 0.89


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_params(param, expected):
    mod = BasicSGDClassifier(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_simple_example_params(digits, param, expected):
    X_train, X_test, y_train, y_test = digits
    mod = BasicSGDClassifier(**{param: expected})
    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if not (param  == "max_iter" and expected <= 1):
        assert acc >= 0.90


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_parameter_setting(param, expected):
    mod = BasicSGDClassifier()
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


def test_hyperparameter_selection(digits):
    X_train, X_test, y_train, y_test = digits
    param_grid = {'eta': [0.02, 0.03]}
    mod = BasicSGDClassifier(max_iter=5)
    xval = RandomizedSearchCV(mod, param_grid, cv=2)
    xval.fit(X_train, y_train)


def test_cross_validation_sklearn(digits):
    X_train, X_test, y_train, y_test = digits
    mod = BasicSGDClassifier(max_iter=5)
    xval = cross_validate(mod, X_train, y_train, cv=2)


def test_cross_validation_nlu(digits):
    X_train, X_test, y_train, y_test = digits
    param_grid = {'eta': [0.02, 0.03]}
    mod = BasicSGDClassifier(max_iter=2)
    best_mod = utils.fit_classifier_with_hyperparameter_search(
        X_train, y_train, mod, cv=2, param_grid=param_grid)
