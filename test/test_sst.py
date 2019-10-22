from collections import Counter
import os
from sklearn.linear_model import LogisticRegression
import sst
from torch_rnn_classifier import TorchRNNClassifier
import pytest

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


sst_home = os.path.join('data', 'trees')


@pytest.mark.parametrize("reader, count", [
    [sst.train_reader(sst_home, class_func=None), 8544],
    [sst.train_reader(sst_home, class_func=sst.binary_class_func), 6920],
    [sst.train_reader(sst_home, class_func=sst.ternary_class_func), 8544],
    [sst.dev_reader(sst_home, class_func=None), 1101],
    [sst.dev_reader(sst_home, class_func=sst.binary_class_func), 872],
    [sst.dev_reader(sst_home, class_func=sst.ternary_class_func), 1101],

])
def test_readers(reader, count):
    result = len(list(reader))
    assert result == count


def test_reader_labeling():
    tree, label = next(sst.train_reader(sst_home, class_func=sst.ternary_class_func))
    for subtree in tree.subtrees():
        assert subtree.label() in {'negative', 'neutral', 'positive'}


def test_build_dataset_vectorizing():
    phi = lambda tree: Counter(tree.leaves())
    class_func = None
    reader = sst.train_reader
    dataset = sst.build_dataset(
        sst_home,
        reader,
        phi,
        class_func,
        vectorizer=None,
        vectorize=True)
    assert len(dataset['X']) == len(list(reader(sst_home)))
    assert len(dataset['y']) == len(dataset['X'])
    assert len(dataset['raw_examples']) == len(dataset['X'])


def test_build_dataset_not_vectorizing():
    phi = lambda tree: tree
    class_func = None
    reader = sst.train_reader
    dataset = sst.build_dataset(
        sst_home,
        reader,
        phi,
        class_func,
        vectorizer=None,
        vectorize=False)
    assert len(dataset['X']) == len(list(reader(sst_home)))
    assert dataset['X'] == dataset['raw_examples']
    assert len(dataset['y']) == len(dataset['X'])


def test_build_rnn_dataset():
    X, y = sst.build_rnn_dataset(
        sst_home, sst.train_reader, class_func=sst.binary_class_func)
    assert len(X) == 6920
    assert len(y) == 6920


@pytest.mark.parametrize("assess_reader", [
    None,
    sst.dev_reader
])
def test_experiment(assess_reader):
    def fit_maxent(X, y):
        mod = LogisticRegression(solver='liblinear', multi_class='auto')
        mod.fit(X, y)
        return mod
    sst.experiment(
        sst_home,
        train_reader=sst.train_reader,
        phi=lambda x: {"$UNK": 1},
        train_func=fit_maxent,
        assess_reader=assess_reader,
        random_state=42)
