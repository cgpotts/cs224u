from collections import Counter
import os
from sklearn.linear_model import LogisticRegression
import sst
from torch_rnn_classifier import TorchRNNClassifier
import pytest
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


utils.fix_random_seeds()


sst_home = os.path.join('data', 'sentiment')


@pytest.mark.parametrize("split_df, expected_count", [
    [sst.train_reader(sst_home, include_subtrees=True, dedup=False), 318582],
    [sst.train_reader(sst_home, include_subtrees=True, dedup=True), 159274],
    [sst.train_reader(sst_home, include_subtrees=False, dedup=False), 8544],
    [sst.train_reader(sst_home, include_subtrees=False, dedup=True), 8534],
    [sst.dev_reader(sst_home, include_subtrees=True, dedup=False), 1101],
    [sst.dev_reader(sst_home, include_subtrees=True, dedup=True), 1100],
    [sst.dev_reader(sst_home, include_subtrees=False, dedup=False), 1101],
    [sst.dev_reader(sst_home, include_subtrees=False, dedup=True), 1100]
])
def test_readers(split_df, expected_count):
    result = split_df.shape[0]
    assert result == expected_count


def test_build_dataset_vectorizing():
    phi = lambda text: Counter(text.split())
    split_df = sst.dev_reader(sst_home)
    dataset = sst.build_dataset(
        split_df,
        phi,
        vectorizer=None,
        vectorize=True)
    assert len(dataset['X']) == split_df.shape[0]
    assert len(dataset['y']) == len(dataset['X'])
    assert len(dataset['raw_examples']) == len(dataset['X'])


def test_build_dataset_not_vectorizing():
    phi = lambda text: text
    split_df = sst.dev_reader(sst_home)
    dataset = sst.build_dataset(
        split_df,
        phi,
        vectorizer=None,
        vectorize=False)
    assert len(dataset['X']) == split_df.shape[0]
    assert dataset['X'] == dataset['raw_examples']
    assert len(dataset['y']) == len(dataset['X'])


def test_build_rnn_dataset():
    split_df = sst.dev_reader(sst_home)
    X, y = sst.build_rnn_dataset(split_df)
    assert len(X) == 1101
    assert len(y) == 1101


@pytest.mark.parametrize("assess_dataframe", [
    None,
    sst.dev_reader(sst_home)
])
def test_experiment(assess_dataframe):
    def fit_maxent(X, y):
        mod = LogisticRegression(solver='liblinear', multi_class='auto')
        mod.fit(X, y)
        return mod
    sst.experiment(
        sst.train_reader(sst_home, include_subtrees=False),
        phi=lambda x: {"$UNK": 1},
        train_func=fit_maxent,
        assess_dataframes=assess_dataframe,
        random_state=42)
