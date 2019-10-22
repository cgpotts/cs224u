import numpy as np
import pandas as pd
import pytest
import vsm

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


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


def test_glove(df):
    """Just makes sure that this code will run; it doesn't check that
    it is creating good models.
    """
    vsm.glove(df, n=100, xmax=100, alpha=0.75, max_iter=100, eta=0.05,
        tol=1e-4, display_progress=True)
