import numpy as np
import pytest
from sklearn.model_selection import RandomizedSearchCV, cross_validate
import tempfile
import torch
import utils

from test_torch_model_base import PARAMS_WITH_TEST_VALUES as BASE_PARAMS
from torch_color_describer import ContextualColorDescriber
from torch_color_describer import create_example_dataset
from torch_color_describer import simple_example

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


PARAMS_WITH_TEST_VALUES = [
    ["hidden_dim", 10],
    ["embedding", np.ones((10,10))],
    ["embed_dim", 5],
    ["freeze_embedding", True]]


PARAMS_WITH_TEST_VALUES += BASE_PARAMS


MINIMAL_VOCAB = [utils.START_SYMBOL, utils.END_SYMBOL, utils.UNK_SYMBOL]


@pytest.fixture
def dataset():
    color_seqs, word_seqs, vocab = create_example_dataset(
        group_size=50, vec_dim=2)
    return color_seqs, word_seqs, vocab


def test_simple_example():
    acc = simple_example()
    assert acc > 0.98


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_params(param, expected):
    mod = ContextualColorDescriber(MINIMAL_VOCAB, **{param: expected})
    result = getattr(mod, param)
    if param == "embedding":
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize("param, expected", PARAMS_WITH_TEST_VALUES)
def test_parameter_setting(param, expected):
    mod = ContextualColorDescriber(MINIMAL_VOCAB)
    mod.set_params(**{param: expected})
    result = getattr(mod, param)
    if param == "embedding":
        assert np.array_equal(result, expected)
    else:
        assert result == expected


def test_build_dataset(dataset):
    color_seqs, word_seqs, vocab = dataset
    mod = ContextualColorDescriber(vocab)
    dataset = mod.build_dataset(color_seqs, word_seqs)
    result = next(iter(dataset))
    assert len(result) == 3


def test_pretrained_embedding(dataset):
    color_seqs, word_seqs, vocab = dataset
    embed_dim = 5
    embedding = np.ones((len(vocab), embed_dim))
    mod = ContextualColorDescriber(
        vocab,
        max_iter=1,
        embedding=embedding,
        freeze_embedding=True)
    mod.fit(color_seqs, word_seqs)
    graph_emb = mod.model.decoder.embedding.weight.detach().cpu().numpy()
    assert np.array_equal(embedding, graph_emb)


@pytest.mark.parametrize("freeze, outcome", [
    [True, True],
    [False, False]
])
def test_embedding_update_control(dataset, freeze, outcome):
    color_seqs, word_seqs, vocab = dataset
    embed_dim = 5
    embedding = np.ones((len(vocab), embed_dim))
    mod = ContextualColorDescriber(
        vocab,
        max_iter=10,
        embedding=embedding,
        freeze_embedding=freeze)
    mod.fit(color_seqs, word_seqs)
    graph_emb = mod.model.decoder.embedding.weight.detach().cpu().numpy()
    assert np.array_equal(embedding, graph_emb) == outcome


@pytest.mark.parametrize("mod_attr, graph_attr", [
    ["hidden_dim", "hidden_size"],
    ["color_dim", "input_size"]
])
def test_encoder_graph_dimensions(dataset, mod_attr, graph_attr):
    color_seqs, word_seqs, vocab = dataset
    mod = ContextualColorDescriber(
        vocab,
        hidden_dim=5,
        max_iter=1)
    mod.fit(color_seqs, word_seqs)
    mod_attr_val = getattr(mod, mod_attr)
    graph_attr_val = getattr(mod.model.encoder.rnn, graph_attr)
    assert mod_attr_val == graph_attr_val


@pytest.mark.parametrize("mod_attr, graph_attr", [
    ["hidden_dim", "hidden_size"],
    ["embed_dim", "input_size"]
])
def test_decoder_graph_dimensions(dataset, mod_attr, graph_attr):
    color_seqs, word_seqs, vocab = dataset
    mod = ContextualColorDescriber(
        vocab,
        hidden_dim=5,
        max_iter=1)
    mod.fit(color_seqs, word_seqs)
    mod_attr_val = getattr(mod, mod_attr)
    graph_attr_val = getattr(mod.model.decoder.rnn, graph_attr)
    assert mod_attr_val == graph_attr_val


def test_predict_proba(dataset):
    color_seqs, word_seqs, vocab = dataset
    mod = ContextualColorDescriber(vocab, max_iter=1)
    mod.fit(color_seqs, word_seqs)
    probs = mod.predict_proba(color_seqs, word_seqs)
    assert all(np.round(t.sum(), 6) == 1.0 for seq in probs for t in seq)


def test_hyperparameter_selection(dataset):
    color_seqs, word_seqs, vocab = dataset
    param_grid = {'hidden_dim': [10, 20]}
    mod = ContextualColorDescriber(vocab, max_iter=5)
    xval = RandomizedSearchCV(mod, param_grid, cv=2)
    xval.fit(color_seqs, word_seqs)


def test_cross_validation_sklearn(dataset):
    color_seqs, word_seqs, vocab = dataset
    param_grid = {'hidden_dim': [10, 20]}
    mod = ContextualColorDescriber(vocab, max_iter=5)
    xval = cross_validate(mod, color_seqs, word_seqs, cv=2)


def test_torch_color_describer_save_load(dataset):
    color_seqs, word_seqs, vocab = dataset
    mod = ContextualColorDescriber(
        vocab,
        embed_dim=10,
        hidden_dim=10,
        max_iter=100,
        embedding=None)
    mod.fit(color_seqs, word_seqs)
    mod.predict(color_seqs)
    with tempfile.NamedTemporaryFile(mode='wb') as f:
        name = f.name
        mod.to_pickle(name)
        mod2 = ContextualColorDescriber.from_pickle(name)
        mod2.predict(color_seqs)
        mod2.fit(color_seqs, word_seqs)


@pytest.mark.parametrize("func", [
    "predict",
    "predict_proba",
    "perplexities",
    "listener_accuracy",
    "score",
    "evaluate"
])
def test_predict_functions_honor_device(dataset, func):
    color_seqs, word_seqs, vocab = dataset
    mod = ContextualColorDescriber(vocab, max_iter=2)
    mod.fit(color_seqs, word_seqs)
    prediction_func = getattr(mod, func)
    with pytest.raises(RuntimeError):
        if func == "predict":
            prediction_func(color_seqs, device="FAKE_DEVICE")
        else:
            prediction_func(color_seqs, word_seqs, device="FAKE_DEVICE")


@pytest.mark.parametrize("func", [
    "predict",
    "predict_proba",
    "perplexities",
    "listener_accuracy",
    "score",
    "evaluate"
])
def test_predict_restores_device(dataset, func):
    color_seqs, word_seqs, vocab = dataset
    mod = ContextualColorDescriber(vocab, max_iter=2)
    mod.fit(color_seqs, word_seqs)
    current_device = mod.device
    assert current_device != torch.device("cpu:0")
    prediction_func = getattr(mod, func)
    if func == "predict":
        prediction_func(color_seqs, device="cpu:0")
    else:
        prediction_func(color_seqs, word_seqs, device="cpu:0")
    assert mod.device == current_device
