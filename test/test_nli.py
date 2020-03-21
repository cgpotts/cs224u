import json
import nli
from nltk.tree import Tree
import numpy as np
import os
import pytest
from sklearn.linear_model import LogisticRegression
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


@pytest.fixture
def wordentail_data():
    nlidata_home = os.path.join('data', 'nlidata')
    wordentail_filename = os.path.join(
        nlidata_home, 'nli_wordentail_bakeoff_data.json')
    with open(wordentail_filename, encoding='utf8') as f:
        data = json.load(f)
    return data


@pytest.mark.parametrize("condition, split",[
    ("edge_disjoint", "train"), ("edge_disjoint", "dev"),
    ("word_disjoint", "train"), ("word_disjoint", "dev")
])
def test_word_entail_featurize(wordentail_data, condition, split):
    data = wordentail_data[condition][split]
    nli.word_entail_featurize(
        data,
        vector_func=lambda x: np.ones(10),
        vector_combo_func=lambda u, v: np.concatenate((u, v)))



@pytest.mark.parametrize("split, count", [
    ["edge_disjoint", 0],
    ["word_disjoint", 0]
])
def test_edge_overlap_size(wordentail_data, split, count):
    result = nli.get_edge_overlap_size(wordentail_data, split)
    assert result == count


@pytest.mark.parametrize("split, count", [
    ["edge_disjoint", 2916],
    ["word_disjoint", 0]
])
def test_vocab_overlap_size(wordentail_data, split, count):
    result = nli.get_vocab_overlap_size(wordentail_data, split)
    assert result == count


@pytest.mark.parametrize("condition", [
    "edge_disjoint", "word_disjoint"
])
def test_wordentail_experiment(wordentail_data, condition):
    nli.wordentail_experiment(
        train_data=wordentail_data[condition]['train'],
        assess_data=wordentail_data[condition]['dev'],
        vector_func=lambda x: np.ones(10),
        vector_combo_func=lambda u, v: np.concatenate((u, v)),
        model=TorchShallowNeuralClassifier(hidden_dim=5, max_iter=1))


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


data_home = os.path.join("data", "nlidata")

snli_home = os.path.join(data_home, "snli_1.0")

multinli_home = os.path.join(data_home, "multinli_1.0")

annotations_home = os.path.join(data_home, "multinli_1.0_annotations")

anli_home = os.path.join(data_home, "anli_v0.1")


@pytest.mark.parametrize("reader_class, corpus_home, count", [
    [nli.SNLITrainReader, snli_home, 550152],
    [nli.SNLIDevReader, snli_home, 10000],
    [nli.MultiNLITrainReader, multinli_home, 392702],
    [nli.MultiNLIMatchedDevReader, multinli_home, 10000],
    [nli.MultiNLIMismatchedDevReader, multinli_home, 10000],
    [nli.ANLITrainReader, anli_home, 162865],
    [nli.ANLIDevReader, anli_home, 3200],

])
@pytest.mark.slow
def test_nli_readers(reader_class, corpus_home, count):
    reader = reader_class(
        corpus_home, samp_percentage=None, filter_unlabeled=False)
    result = len([1 for _ in reader.read()])
    assert result == count


@pytest.mark.parametrize("reader_class, rounds, count", [
    [nli.ANLITrainReader, (1,2,3), 162865],
    [nli.ANLITrainReader, (1,), 16946],
    [nli.ANLITrainReader, (2,), 45460],
    [nli.ANLITrainReader, (3,), 100459],
    [nli.ANLIDevReader, (1,2,3), 3200],
    [nli.ANLIDevReader, (1,), 1000],
    [nli.ANLIDevReader, (2,), 1000],
    [nli.ANLIDevReader, (3,), 1200],
])
@pytest.mark.slow
def test_anli_readers_by_rounds(reader_class, rounds, count):
    reader = reader_class(anli_home, rounds=rounds)
    result = len([1 for _ in reader.read()])
    assert result == count


@pytest.mark.parametrize("src_filename", [
    "multinli_1.0_matched_annotations.txt",
    "multinli_1.0_mismatched_annotations.txt"
])
def test_read_annotated_subset(src_filename):
    src_filename = os.path.join(
        annotations_home, src_filename)
    data = nli.read_annotated_subset(src_filename, multinli_home)
    assert len(data) == 495


def test_build_dataset():
    nli.build_dataset(
        reader=nli.SNLITrainReader(snli_home, samp_percentage=0.01),
        phi=lambda x, y: {"$UNK": 1},
        vectorizer=None,
        vectorize=True)


@pytest.mark.parametrize("assess_reader", [
    None,
    nli.SNLIDevReader(snli_home)
])
def test_experiment(assess_reader):
    def fit_maxent(X, y):
        mod = LogisticRegression(solver='liblinear', multi_class='auto')
        mod.fit(X, y)
        return mod
    nli.experiment(
        train_reader=nli.SNLITrainReader(snli_home, samp_percentage=0.01),
        phi=lambda x, y: {"$UNK": 1},
        train_func=fit_maxent,
        assess_reader=assess_reader,
        random_state=42)
