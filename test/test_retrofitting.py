import numpy as np
import pandas as pd
import pytest
import retrofitting
from retrofitting import Retrofitter

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


@pytest.fixture
def retrofitter():
    return Retrofitter(
        max_iter=100,
        alpha=None,
        beta=None,
        tol=1e-2,
        verbose=False,
        introspecting=False)


def test_identical_vectors(retrofitter):
    X = pd.DataFrame([
        [0.1, 0.3],
        [0.1, 0.3]])
    edges = {0: {1}, 1: {0}}
    Y = retrofitter.fit(X, edges)
    assert X.loc[0].equals(Y.loc[0])


def test_average_with_neighbor(retrofitter):
    X = pd.DataFrame([
        [0.5, 0.0],
        [0.0, 0.5]])
    edges = {0: {1}, 1:set()}
    Y = retrofitter.fit(X, edges)
    assert np.array_equal(Y.loc[0], np.array([0.25, 0.25]))
    assert X.loc[1].equals(Y.loc[1])


def test_mutual_averaging(retrofitter):
    retrofitter.tol = 1e-10
    X = pd.DataFrame([
        [0.5, 0.0],
        [0.0, 0.5]])
    edges = {0: {1}, 1: {0}}
    Y = retrofitter.fit(X, edges)
    Y = Y.round(6)
    assert np.array_equal(Y.loc[0], np.array([0.333333, 0.166667]))
    assert np.array_equal(Y.loc[1], np.array([0.166667, 0.333333]))


def test_alpha_setting(retrofitter):
    retrofitter.alpha = lambda x: 0.0
    X = pd.DataFrame([
        [0.5, 0.0],
        [0.0, 0.5]])
    edges = {0: {1}, 1:set()}
    Y = retrofitter.fit(X, edges)
    assert np.array_equal(Y.loc[0], np.array([0.0, 0.5]))


def test_plot_retro_path():
    Q_hat = pd.DataFrame([
        [0.0, 0.0],
        [0.0, 0.5],
        [0.5, 0.0]],
        columns=['x', 'y'])
    edges = {0: {1, 2}, 1: set(), 2: set()}
    retrofitting.plot_retro_path(Q_hat, edges)
