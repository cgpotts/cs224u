import os
import pytest
import rel_ext
from sklearn.linear_model import LogisticRegression
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


@pytest.fixture
def corpus():
    rel_ext_data_home =  os.path.join('data', 'rel_ext_data')
    src_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        rel_ext_data_home, 'corpus.tsv.gz')
    return rel_ext.Corpus(src_filename)


@pytest.fixture
def kb():
    rel_ext_data_home = os.path.join('data', 'rel_ext_data')
    src_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '..',
        rel_ext_data_home, 'kb.tsv.gz')
    return rel_ext.KB(src_filename)


def dummy_vectorizing_feature_function(kbt, corpus, feature_counter):
    return {"bias": 1}


def dummy_nonvectorizing_feature_function(kbt, corpus):
    return utils.randvec(10)


def test_corpus_length(corpus):
    assert len(corpus) == 331696


def test_kb_length(kb):
    assert len(kb) == 45884


def test_dataset_build_dataset(corpus, kb):
    dataset = rel_ext.Dataset(corpus, kb)
    dat = dataset.build_dataset(include_positive=True, sampling_rate=0.1)


def test_dataset_build_splits(corpus, kb):
    dataset = rel_ext.Dataset(corpus, kb)
    dat = dataset.build_splits(seed=1)


def test_dataset_featurize_vectorize(corpus, kb):
    dataset = rel_ext.Dataset(corpus, kb)
    kbts_by_rel, _ = dataset.build_dataset(sampling_rate=0.1)
    featurizers = [lambda kbt, corpus, feature_counter: {"bias": 1}]
    dataset.featurize(kbts_by_rel, featurizers)


def test_dataset_featurize_no_vectorize(corpus, kb):
    dataset = rel_ext.Dataset(corpus, kb)
    kbts_by_rel, _ = dataset.build_dataset(sampling_rate=0.1)
    def featurizer(kbt, corpus):
        return utils.randvec(10)
    dataset.featurize(kbts_by_rel, [featurizer], vectorize=False)


@pytest.mark.parametrize("featurizer, vectorize", [
    [dummy_nonvectorizing_feature_function, False],
    [dummy_vectorizing_feature_function, True]
])
def test_experiment(featurizer, vectorize, corpus, kb):
    dataset = rel_ext.Dataset(corpus, kb)
    splits = dataset.build_splits(
        split_names=['tiny_train', 'tiny_dev', 'rest'],
        split_fracs=[0.05, 0.05, 0.90],
        seed=1)
    results = rel_ext.experiment(
        splits,
        train_split='tiny_train',
        test_split='tiny_dev',
        featurizers=[featurizer],
        train_sampling_rate=0.2,
        test_sampling_rate=0.2,
        vectorize=vectorize,
        verbose=False)


@pytest.mark.parametrize("featurizer, vectorize", [
    [dummy_nonvectorizing_feature_function, False],
    [dummy_vectorizing_feature_function, True]
])
def test_examine_model_weights(featurizer, vectorize, corpus, kb):
    dataset = rel_ext.Dataset(corpus, kb)
    splits = dataset.build_splits(
        split_names=['tiny_train', 'tiny_dev', 'rest'],
        split_fracs=[0.05, 0.05, 0.90],
        seed=1)
    results = rel_ext.experiment(
        splits,
        train_split='tiny_train',
        test_split='tiny_dev',
        featurizers=[featurizer],
        vectorize=vectorize,
        verbose=False)
    rel_ext.examine_model_weights(results)


@pytest.mark.parametrize("featurizer, vectorize", [
    [dummy_nonvectorizing_feature_function, False],
    [dummy_vectorizing_feature_function, True]
])
def test_find_new_relation_instances(corpus, kb, featurizer, vectorize):
    dataset = rel_ext.Dataset(corpus, kb)
    rel_ext.find_new_relation_instances(
        dataset,
        [featurizer],
        train_split='train',
        test_split='dev',
        model_factory=lambda: LogisticRegression(solver='liblinear'),
        k=10,
        vectorize=vectorize,
        verbose=False)
