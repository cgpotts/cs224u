import json
import nli
from nltk.tree import Tree
import numpy as np
import os
import pytest
from sklearn.linear_model import LogisticRegression
from tf_shallow_neural_classifier import TfShallowNeuralClassifier


@pytest.fixture
def wordentail_data():
    nlidata_home = 'nlidata'
    wordentail_filename = os.path.join(
        nlidata_home, 'nli_wordentail_bakeoff_data.json')
    with open(wordentail_filename) as f:
        data = json.load(f)
    return data


@pytest.mark.parametrize("key", nli.BAKEOFF_CONDITION_NAMES)
def test_build_bakeoff_dataset(wordentail_data, key):
    dataset = nli.build_bakeoff_dataset(
        wordentail_data,
        vector_func=lambda x: np.ones(10),
        vector_combo_func=lambda u, v: np.concatenate((u, v)))
    assert key in dataset


@pytest.mark.parametrize("split, count", [
    ["edge_disjoint", 0],
    ["word_disjoint", 0],
    ["word_disjoint_balanced", 0]
])
def test_edge_overlap_size(wordentail_data, split, count):
    result = nli.get_edge_overlap_size(wordentail_data, split)
    assert result == count


@pytest.mark.parametrize("split, count", [
    ["edge_disjoint", 4769],
    ["word_disjoint", 0],
    ["word_disjoint_balanced", 0]
])
def test_vocab_overlap_size(wordentail_data, split, count):
    result = nli.get_vocab_overlap_size(wordentail_data, split)
    assert result == count


def test_bakeoff_experiment(wordentail_data):
    dataset = nli.build_bakeoff_dataset(
        wordentail_data,
        vector_func=lambda x: np.ones(10),
        vector_combo_func=lambda u, v: np.concatenate((u, v)))
    net = TfShallowNeuralClassifier(hidden_dim=5, max_iter=1)
    nli.bakeoff_experiment(dataset, net)


@pytest.mark.parametrize("s, expected", [
    [
        "( ( ( A person ) ) ( eats pizza ) )",
        Tree.fromstring("(X (X (X A person ) ) (X eats pizza ) )")
    ],
    [
        "non-tree",
         Tree.fromstring("(X non-tree )")
    ]
])
def test_str2tree(s, expected):
    result = nli.str2tree(s, binarize=True)
    assert result == expected


@pytest.mark.parametrize("reader_class, count", [
    [nli.SNLITrainReader, 550152],
    [nli.SNLIDevReader, 10000],
    [nli.MultiNLITrainReader, 392702],
    [nli.MultiNLIMatchedDevReader, 10000],
    [nli.MultiNLIMismatchedDevReader, 10000]

])
@pytest.mark.slow
def test_nli_readers(reader_class, count):
    reader = reader_class(samp_percentage=None, filter_unlabeled=False)
    result = len([1 for _ in reader.read()])
    assert result == count


@pytest.mark.parametrize("src_filename", [
    "multinli_1.0_matched_annotations.txt",
    "multinli_1.0_mismatched_annotations.txt"
])
def test_read_annotated_subset(src_filename):
    annotations_home = os.path.join(
        "nlidata", "multinli_1.0_annotations")
    src_filename = os.path.join(
        annotations_home, src_filename)
    data = nli.read_annotated_subset(src_filename)
    assert len(data) == 495


def test_build_dataset():
    nli.build_dataset(
        reader=nli.SNLITrainReader(samp_percentage=0.01),
        phi=lambda x, y: {"$UNK": 1},
        vectorizer=None,
        vectorize=True)


def test_experiment():
    def fit_maxent(X, y):
        mod = LogisticRegression()
        mod.fit(X, y)
        return mod
    nli.experiment(
        train_reader=nli.SNLITrainReader(samp_percentage=0.01),
        phi=lambda x, y: {"$UNK": 1},
        train_func=fit_maxent,
        assess_reader=None,
        random_state=42)
