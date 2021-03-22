import numpy as np
import os
import pandas as pd
import pytest
import torch
from transformers import BertModel, BertTokenizer

import vsm
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


DATA_HOME = os.path.join('data', 'vsmdata')
REL_HOME = os.path.join('data', 'wordrelatedness')


@pytest.fixture
def df():
    vocab = ['ab', 'bc', 'cd', 'de']
    df = pd.DataFrame([
        [  4.,   4.,   2.,   0.],
        [  4.,  61.,   8.,  18.],
        [  2.,   8.,  10.,   0.],
        [  0.,  18.,   0.,   5.]],
        index=vocab, columns=vocab)
    return df


@pytest.fixture
def count_dfs():
    dfs = {}
    basenames = (
         'yelp_window5-scaled.csv.gz',
         'yelp_window20-flat.csv.gz',
         'giga_window5-scaled.csv.gz',
         'giga_window20-flat.csv.gz')
    for basename in basenames:
        dfs[basename] = pd.read_csv(
            os.path.join(DATA_HOME, basename), index_col=0)
    return dfs


@pytest.mark.parametrize("arg, expected", [
    [
        np.array([[34.0, 11.0], [ 47.0, 7.0]]),
        np.array([[0.92345679, 1.34444444], [1.06378601, 0.71296296]])
    ]
])
def test_observed_over_expected(arg, expected):
    result = vsm.observed_over_expected(arg).round(8)
    assert np.array_equal(result, expected.round(8))


@pytest.mark.parametrize("arg, expected, positive", [
    [
        np.array([[34.0, 11.0], [ 47.0, 7.0]]),
        np.array([[0.0, 0.29598088], [0.06183425, 0.0]]),
        True
    ],
    [
        np.array([[34.0, 11.0], [ 47.0, 7.0]]),
        np.array([[-0.07963127, 0.29598088], [0.06183425, -0.33832581]]),
        False
    ],
    [
        np.array([[1.0, 0.0, 0.0], [1000.0, 1000.0, 4000.0], [1000.0, 2000.0, 999.0]]),
        np.array([[1.60893804, 0., 0.],[0., 0., 0.28788209], [0.22289371, 0.51107566, 0.]]),
        True
    ]
])
def test_pmi(arg, expected, positive):
    result = vsm.pmi(arg, positive=positive).round(8)
    assert np.array_equal(result, expected.round(8))


@pytest.mark.parametrize("arg, expected", [
    [
        np.array([[34.0, 0.0], [0.0, 7.0]]),
        np.array([[0.69314718, 0.0], [0.0, 0.69314718]])
    ],
    [
        np.array([
            [10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0,  0.0],
            [10.0, 10.0,  0.0,  0.0],
            [0.0,   0.0,  0.0,  1.0]]),
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.09589402, 0.09589402, 0.14384104, 0.],
            [0.23104906, 0.23104906, 0., 0.],
            [0., 0., 0., 0.12602676]])
    ]
])
def test_tfidf(arg, expected):
    result = vsm.tfidf(arg).round(8)
    expected = expected.round(8)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("bigram, expected", [
    [
        '<w>a', np.array([4.0, 4.0, 2.0, 0.0])
    ],
    [
        '<w>b',  np.array([4.0, 61.0, 8.0, 18.0])
    ],
])
def test_ngram_vsm(df, bigram, expected):
    X = vsm.ngram_vsm(df)
    result = X.loc[bigram]
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("word, n, expected", [
    ['abc', 1, ['a', 'b', 'c']],
    ['abc', 2, ['<w>a', 'ab', 'bc', 'c</w>']]
])
def test_get_character_ngrams(word, n, expected):
    result = vsm.get_character_ngrams(word, n=n)
    assert result == expected


def test_tsne_viz(df):
    vsm.tsne_viz(df)


def test_lsa(df):
    vsm.lsa(df, k=2)


@pytest.mark.parametrize("basename1, basename2", [
    ['yelp_window5-scaled.csv.gz', 'yelp_window20-flat.csv.gz'],
    ['yelp_window20-flat.csv.gz', 'giga_window5-scaled.csv.gz'],
    ['giga_window5-scaled.csv.gz', 'giga_window20-flat.csv.gz']
])
def test_count_matrix_index_identity(basename1, basename2, count_dfs):
    df1 = count_dfs[basename1]
    df2 = count_dfs[basename2]
    assert df1.index.equals(df2.index)


@pytest.mark.parametrize("batch_ids, expected_shape", [
    (
        [[101, 14324, 102]], (1, 3, 768)
    )
])
def test_hf_represent(batch_ids, expected_shape):
    """Really just tests that the function works."""
    batch_ids = torch.LongTensor(batch_ids)
    model = BertModel.from_pretrained('bert-base-uncased')
    reps = vsm.hf_represent(batch_ids, model, layer=-1)
    assert reps.shape == expected_shape


@pytest.mark.parametrize("text, add_special_tokens, expected_len", [
    ("the cat", True, 4),
    ("the cat", False, 2)
])
def test_hf_encode(text, add_special_tokens, expected_len):
    """Really just tests that the function works."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = vsm.hf_encode(
        text, tokenizer, add_special_tokens=add_special_tokens)
    assert encoding.shape == (1, expected_len)


@pytest.mark.parametrize("X, expected", [
    ([[[1., 2, 3], [4., 5, 6]]], [[2.5, 3.5, 4.5]])
])
def test_mean_pooling(X, expected):
    X = torch.tensor(X)
    expected = torch.tensor(expected)
    result = vsm.mean_pooling(X)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("X, expected", [
    [
        [[[1., 4, 3], [4., 2, 6]]],
        [[4., 4, 6]]
    ],
    [
        [[[1., 4, 3], [4., 2, 6]], [[1., 4, 3], [4., 2, 6]]],
        [[4., 4, 6], [4., 4, 6]]
    ],
])
def test_max_pooling(X, expected):
    X = torch.tensor(X)
    expected = torch.tensor(expected)
    result = vsm.max_pooling(X)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("X, expected", [
    [
        [[[1., 4, 3], [4., 2, 6]]],
        [[1., 2, 3]]
    ],
    [
        [[[1., 4, 3], [4., 2, 6]], [[4, 3, 2], [2, 3, 4.]]],
        [[1., 2, 3], [2., 3, 2]]
    ]
])
def test_min_pooling(X, expected):
    X = torch.tensor(X)
    expected = torch.tensor(expected)
    result = vsm.min_pooling(X)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("X, expected", [
    [
        [[[1., 4, 3], [4., 2, 6]]],
        [[4., 2, 6]]
    ],
    [
        [[[1., 4, 3], [4., 2, 6]], [[4, 3, 2], [2, 3, 4.]]],
        [[4., 2, 6], [2., 3, 4]]
    ]
])
def test_last_pooling(X, expected):
    X = torch.tensor(X)
    expected = torch.tensor(expected)
    result = vsm.last_pooling(X)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("pool_func", [
    vsm.mean_pooling,
    vsm.max_pooling,
    vsm.min_pooling,
    vsm.last_pooling
])
def test_pool_func_shape_check_raise(pool_func):
    hidden_states = torch.Tensor([1,2])
    with pytest.raises(ValueError):
        pool_func(hidden_states)


def test_create_subword_pooling_vsm():
    """Really just tests that the function works."""
    vocab = ["puppy", "snuffleupagus"]
    bert_weights_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_weights_name)
    model = BertModel.from_pretrained(bert_weights_name)
    df = vsm.create_subword_pooling_vsm(
        vocab, tokenizer, model,
        layer=1, pool_func=vsm.mean_pooling)
    assert list(df.index) == vocab


def test_word_relatedness_evaluation():
    """Really just tests that the function works."""
    dev_df = pd.read_csv(
        os.path.join(REL_HOME, "cs224u-wordrelatedness-dev.csv"))
    count_df = pd.read_csv(
        os.path.join(DATA_HOME, "giga_window5-scaled.csv.gz"), index_col=0)
    count_pred_df, count_rho = vsm.word_relatedness_evaluation(dev_df, count_df)
    assert isinstance(count_rho, float)
    assert 'prediction' in count_pred_df.columns
