from datasets import load_dataset
import json
from nltk.tree import Tree
import numpy as np
import os
import pytest
from sklearn.linear_model import LogisticRegression

import nli
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import utils


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2022"


utils.fix_random_seeds()


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

annotations_home = os.path.join(data_home, "multinli_1.0_annotations")

snli = load_dataset("snli")
mnli = load_dataset("multi_nli")
anli = load_dataset("anli")


@pytest.mark.parametrize("dataset, split, count", [
    [snli, "train", 550152],
    [snli, "validation", 10000],
    [mnli, "train", 392702],
    [mnli, "validation_matched", 9815],
    [mnli, "validation_mismatched", 9832]
])
@pytest.mark.slow
def test_nli_readers(dataset, split, count):
    reader = nli.NLIReader(
        dataset[split], samp_percentage=None, filter_unlabeled=False)
    result = len([1 for _ in reader.read()])
    assert result == count


@pytest.mark.parametrize("split, rounds, count", [
    ["train", (1,2,3), 162865],
    ["train", (1,), 16946],
    ["train", (2,), 45460],
    ["train", (3,), 100459],
    ["dev", (1,2,3), 3200],
    ["dev", (1,), 1000],
    ["dev", (2,), 1000],
    ["dev", (3,), 1200],
])
@pytest.mark.slow
def test_anli_readers_by_rounds(split, rounds, count):
    splits = [anli['{}_r{}'.format(split, i)] for i in rounds]
    reader = nli.NLIReader(*splits)
    result = len([1 for _ in reader.read()])
    assert result == count


@pytest.mark.parametrize("src_filename", [
    "multinli_1.0_matched_annotations.txt",
    "multinli_1.0_mismatched_annotations.txt"
])
def test_read_annotated_subset(src_filename):
    src_filename = os.path.join(
        annotations_home, src_filename)
    if 'mismatched' in src_filename:
        split = 'validation_mismatched'
    else:
        split = 'validation_matched'
    data = nli.read_annotated_subset(src_filename, mnli[split])
    assert len(data) == 495


def test_build_dataset():
    nli.build_dataset(
        reader=nli.NLIReader(snli['train'], samp_percentage=0.01),
        phi=lambda ex: {"$UNK": 1},
        vectorizer=None,
        vectorize=True)


@pytest.mark.parametrize("assess_reader", [
    None,
    nli.NLIReader(snli['validation'])
])
def test_experiment(assess_reader):
    def fit_maxent(X, y):
        mod = LogisticRegression(solver='liblinear', multi_class='auto')
        mod.fit(X, y)
        return mod
    nli.experiment(
        train_reader=nli.NLIReader(snli['train'], samp_percentage=0.01),
        phi=lambda ex: {"$UNK": 1},
        train_func=fit_maxent,
        assess_reader=assess_reader,
        random_state=42)
