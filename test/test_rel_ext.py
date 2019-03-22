import os
import pytest
import rel_ext

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


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


def test_dataset_featurize(corpus, kb):
    dataset = rel_ext.Dataset(corpus, kb)
    kbts_by_rel, _ = dataset.build_dataset()
    featurizers = [lambda kbt, corpus, feature_counter: {"bias": 1}]
    dataset.featurize(kbts_by_rel, featurizers)
