import numpy as np
import os
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, cross_validate
import string
import tempfile
import torch
import torch.nn as nn
import utils

from test_torch_model_base import PARAMS_WITH_TEST_VALUES as BASE_PARAMS
from np_rnn_classifier import RNNClassifier
from np_rnn_classifier import simple_example as np_simple_example
from torch_rnn_classifier import TorchRNNClassifier, simple_example

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


PARAMS_WITH_TEST_VALUES = [
    ["hidden_dim", 10],
    ["embedding", np.ones((10,10))],
    ["use_embedding", False],
    ["embed_dim", 5],
    ["rnn_cell_class", nn.GRU],
    ["bidirectional", True],
    ['freeze_embedding', True]]


PARAMS_WITH_TEST_VALUES += BASE_PARAMS


@pytest.fixture
def X_sequence():
    vocab = ['a', 'b', '$UNK']

    # No b before an a
    train = [
        [list('ab'), 'good'],
        [list('aab'), 'good'],
        [list('abb'), 'good'],
        [list('aabb'), 'good'],
        [list('ba'), 'bad'],
        [list('baa'), 'bad'],
        [list('bba'), 'bad'],
        [list('bbaa'), 'bad'],
        [list('aba'), 'bad']]

    test = [
        [list('baaa'), 'bad'],
        [list('abaa'), 'bad'],
        [list('bbaa'), 'bad'],
        [list('aaab'), 'good'],
        [list('aaabb'), 'good']]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return X_train, X_test, y_train, y_test, vocab


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


def test_model(X_sequence):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X_train, X_test, y_train, y_test, vocab = X_sequence
    mod = TorchRNNClassifier(vocab=vocab, max_iter=100)
    mod.fit(X_train, y_train)
    mod.predict(X_test)
    mod.predict_proba(X_test)


def test_np_model(X_sequence):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    X_train, X_test, y_train, y_test, vocab = X_sequence
    embedding = np.array([utils.randvec(10) for _ in vocab])
    mod = RNNClassifier(
        vocab=vocab,
        embedding=embedding,
        hidden_dim=20,
        max_iter=100)
    mod.fit(X_train, y_train)
    mod.predict(X_test)
    mod.predict_proba(X_test)
    mod.predict_one(X_test[0])
    mod.predict_one_proba(X_test[0])


def test_cheese_disease(cheese_disease_dataset):
    vocab = cheese_disease_dataset['vocab']
    X_train = cheese_disease_dataset['X_train']
    y_train = cheese_disease_dataset['y_train']
    mod = TorchRNNClassifier(
        vocab=vocab,
        embed_dim=50,
        hidden_dim=50,
        max_iter=200)
    mod.fit(X_train, y_train)
    X_test = cheese_disease_dataset['X_train']
    y_test = cheese_disease_dataset['y_train']
    pred = mod.predict(X_test)
    acc = accuracy_score(y_test, pred)
    assert acc > 0.80


def test_simple_example():
    acc = simple_example()
    assert acc >= 0.95


def test_np_simple_example():
    acc = np_simple_example()
    assert acc >= 0.95


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_simple_example_params(X_sequence, param, expected):
    X_train, X_test, y_train, y_test, vocab = X_sequence
    mod = TorchRNNClassifier(vocab, **{param: expected})

    if param == "use_embedding" and expected == False:
        embedding = np.random.uniform(
            low=-1.0, high=1.0, size=(len(vocab), 60))
        X_train = [[embedding[vocab.index(w)] for w in ex] for ex in X_train]
        X_test = [[embedding[vocab.index(w)] for w in ex] for ex in X_test]

    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if not (param == "max_iter" and expected == 0):
        assert acc >= 0.60


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_params(param, expected):
    vocab = []
    mod = TorchRNNClassifier(vocab, **{param: expected})
    result = getattr(mod, param)
    if param == "embedding":
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_parameter_setting(param, expected):
    vocab = []
    mod = TorchRNNClassifier(vocab)
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    if param == "embedding":
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize("param, expected", [
    ['hidden_dim', 10],
    ['eta', 1.0],
    ['max_iter', 100]
])
def test_np_parameter_setting(param, expected):
    vocab = []
    mod = TorchRNNClassifier(vocab)
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    assert result == expected


def test_np_set_embed_dim():
    value = 26
    vocab = []
    mod = RNNClassifier(vocab, embed_dim=5)
    mod.embed_dim = value
    assert mod.embedding.shape[1] == value


@pytest.mark.parametrize("with_y, expected", [
    [True, 3],
    [False, 2]
])
def test_build_dataset(cheese_disease_dataset, with_y, expected):
    vocab = cheese_disease_dataset['vocab']
    X = cheese_disease_dataset['X_train']
    y = cheese_disease_dataset['y_train']
    mod = TorchRNNClassifier(vocab)
    if with_y:
        dataset = mod.build_dataset(X, y)
    else:
        dataset = mod.build_dataset(X)
    result = next(iter(dataset))
    assert len(result) == expected


@pytest.mark.parametrize("mod_attr, graph_attr", [
    ["hidden_dim", "hidden_size"],
    ["embed_dim", "input_size"],
    ["bidirectional", "bidirectional"]
])
def test_model_graph_dimensions(X_sequence, mod_attr, graph_attr):
    X_train, X_test, y_train, y_test, vocab = X_sequence
    mod = TorchRNNClassifier(vocab, max_iter=1)
    mod.fit(X_train, y_train)
    mod_attr_val = getattr(mod, mod_attr)
    graph_attr_val = getattr(mod.model.rnn.rnn, graph_attr)
    assert mod_attr_val == graph_attr_val


def test_pretrained_embedding(X_sequence):
    X_train, X_test, y_train, y_test, vocab = X_sequence
    embed_dim = 5
    embedding = np.ones((len(vocab), embed_dim))
    mod = TorchRNNClassifier(
        vocab,
        max_iter=1,
        embedding=embedding,
        freeze_embedding=True)
    mod.fit(X_train, y_train)
    graph_emb = mod.model.rnn.embedding.weight.detach().cpu().numpy()
    assert np.array_equal(embedding, graph_emb)


@pytest.mark.parametrize("freeze, outcome", [
    [True, True],
    [False, False]
])
def test_embedding_update_control(X_sequence, freeze, outcome):
    X_train, X_test, y_train, y_test, vocab = X_sequence
    embed_dim = 5
    embedding = np.ones((len(vocab), embed_dim))
    mod = TorchRNNClassifier(
        vocab,
        max_iter=10,
        embedding=embedding,
        freeze_embedding=freeze)
    mod.fit(X_train, y_train)
    graph_emb = mod.model.rnn.embedding.weight.detach().cpu().numpy()
    assert np.array_equal(embedding, graph_emb) == outcome


@pytest.mark.parametrize("model_class", [
    TorchRNNClassifier,
    RNNClassifier
])
def test_predict_proba(cheese_disease_dataset, model_class):
    vocab = cheese_disease_dataset['vocab']
    X = cheese_disease_dataset['X_train']
    y = cheese_disease_dataset['y_train']
    mod = model_class(vocab, max_iter=1)
    mod.fit(X, y)
    probs = mod.predict_proba(X)
    assert all(np.round(x.sum(), 6) == 1.0 for x in probs)


@pytest.mark.parametrize("model_class", [
    TorchRNNClassifier,
    RNNClassifier
])
def test_hyperparameter_selection(cheese_disease_dataset, model_class):
    vocab = cheese_disease_dataset['vocab']
    X = cheese_disease_dataset['X_train']
    y = cheese_disease_dataset['y_train']
    param_grid = {'hidden_dim': [10, 20]}
    mod = model_class(vocab, max_iter=5)
    xval = RandomizedSearchCV(mod, param_grid, cv=2)
    xval.fit(X, y)


@pytest.mark.parametrize("model_class", [
    TorchRNNClassifier,
    RNNClassifier
])
def test_cross_validation_sklearn(cheese_disease_dataset, model_class):
    vocab = cheese_disease_dataset['vocab']
    X = cheese_disease_dataset['X_train']
    y = cheese_disease_dataset['y_train']
    mod = TorchRNNClassifier(vocab, max_iter=5)
    xval = cross_validate(mod, X, y, cv=2)


@pytest.mark.parametrize("model_class", [
    TorchRNNClassifier,
    RNNClassifier
])
def test_cross_validation_nlu(X_sequence, model_class):
    X_train, X_test, y_train, y_test, vocab = X_sequence
    mod = model_class(vocab, max_iter=2)
    best_mod = utils.fit_classifier_with_hyperparameter_search(
        X_train, y_train, mod, cv=2, param_grid={'hidden_dim': [10, 20]})


def test_torch_rnn_classifier_save_load(X_sequence):
    X_train, X_test, y_train, y_test, vocab = X_sequence
    mod = TorchRNNClassifier(vocab=vocab, max_iter=2)
    mod.fit(X_train, y_train)
    mod.predict(X_test)
    with tempfile.NamedTemporaryFile(mode='wb') as f:
        name = f.name
        mod.to_pickle(name)
        mod2 = TorchRNNClassifier.from_pickle(name)
        mod2.predict(X_test)
        mod2.fit(X_test, y_test)

@pytest.mark.parametrize("func", ["predict", "predict_proba"])
def test_predict_functions_honor_device(X_sequence, func):
    X_train, X_test, y_train, y_test, vocab = X_sequence
    mod = TorchRNNClassifier(vocab, max_iter=2)
    mod.fit(X_train, y_train)
    prediction_func = getattr(mod, func)
    with pytest.raises(RuntimeError):
        prediction_func(X_test, device="FAKE_DEVICE")


@pytest.mark.parametrize("func", ["predict", "predict_proba"])
def test_predict_restores_device(X_sequence, func):
    X_train, X_test, y_train, y_test, vocab = X_sequence
    mod = TorchRNNClassifier(vocab, max_iter=2)
    mod.fit(X_train, y_train)
    current_device = mod.device
    assert current_device != torch.device("cpu:0")
    prediction_func = getattr(mod, func)
    prediction_func(X_test, device="cpu:0")
    assert mod.device == current_device
