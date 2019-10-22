from collections import Counter, namedtuple
from nltk.tree import Tree
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import scipy.stats
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


def sentiment_treebank_reader(src_filename, class_func=None):
    """Iterator for the Penn-style distribution of the Stanford
    Sentiment Treebank. The iterator yields (tree, label) pairs.

    The labels are strings. They do not make sense as a linear order
    because negative ('0', '1'), neutral ('2'), and positive ('3','4')
    do not form a linear order conceptually, and because '0' is
    stronger than '1' but '4' is stronger than '3'.

    Parameters
    ----------
    src_filename : str
        Full path to the file to be read.
    class_func : None, or function mapping labels to labels or None
        If this is None, then the original 5-way labels are returned.
        Other options: `binary_class_func` and `ternary_class_func`
        (or you could write your own).

    Yields
    ------
    (tree, label)
        nltk.Tree, str in {'0','1','2','3','4'}

    """
    if class_func is None:
        class_func = lambda x: x
    with open(src_filename, encoding='utf8') as f:
        for line in f:
            tree = Tree.fromstring(line)
            label = class_func(tree.label())
            # As in the paper, if the root node doesn't fall into any
            # of the classes for this version of the problem, then
            # we drop the example:
            if label:
                for subtree in tree.subtrees():
                    subtree.set_label(class_func(subtree.label()))
                yield (tree, label)


def binary_class_func(y):
    """Define a binary SST task.

    Parameters
    ----------
    y : str
        Assumed to be one of the SST labels.

    Returns
    -------
    str or None
        None values are ignored by `build_dataset` and thus left out of
        the experiments.

    """
    if y in ("0", "1"):
        return "negative"
    elif y in ("3", "4"):
        return "positive"
    else:
        return None


def ternary_class_func(y):
    """Define a binary SST task. Just like `binary_class_func` except
    input '2' returns 'neutral'."""
    if y in ("0", "1"):
        return "negative"
    elif y in ("3", "4"):
        return "positive"
    else:
        return "neutral"


def train_reader(sst_home, **kwargs):
    """Convenience function for reading the train file, full-trees only."""
    src = os.path.join(sst_home, 'train.txt')
    return sentiment_treebank_reader(src, **kwargs)


def dev_reader(sst_home, **kwargs):
    """Convenience function for reading the dev file, full-trees only."""
    src = os.path.join(sst_home, 'dev.txt')
    return sentiment_treebank_reader(src, **kwargs)


def test_reader(sst_home, **kwargs):
    """Convenience function for reading the test file, full-trees only.
    This function should be used only for the final stages of a project,
    to obtain final results.
    """
    src = os.path.join(sst_home, 'test.txt')
    return sentiment_treebank_reader(src, **kwargs)


def build_dataset(sst_home, reader, phi, class_func, vectorizer=None, vectorize=True):
    """Core general function for building experimental datasets.

    Parameters
    ----------
    sst_home : str
        Full path to the 'trees' directory for SST.
    reader : iterator
       Should be `train_reader`, `dev_reader`, or another function
       defined in those terms. This is the dataset we'll be
       featurizing.
    phi : feature function
       Any function that takes an `nltk.Tree` instance as input
       and returns a bool/int/float-valued dict as output.
    class_func : function on the SST labels
       Any function like `binary_class_func` or `ternary_class_func`.
       This modifies the SST labels based on the experimental
       design. If `class_func` returns None for a label, then that
       item is ignored.
    vectorizer : sklearn.feature_extraction.DictVectorizer
       If this is None, then a new `DictVectorizer` is created and
       used to turn the list of dicts created by `phi` into a
       feature matrix. This happens when we are training.
       If this is not None, then it's assumed to be a `DictVectorizer`
       and used to transform the list of dicts. This happens in
       assessment, when we take in new instances and need to
       featurize them as we did in training.
    vectorize : bool
       Whether to use a DictVectorizer. Set this to False for
       deep learning models that process their own input.

    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels), 'vectorizer' (the `DictVectorizer`), and
        'raw_examples' (the `nltk.Tree` objects, for error analysis).

    """
    labels = []
    feat_dicts = []
    raw_examples = []
    for tree, label in reader(sst_home, class_func=class_func):
        labels.append(label)
        feat_dicts.append(phi(tree))
        raw_examples.append(tree)
    feat_matrix = None
    if vectorize:
        # In training, we want a new vectorizer:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=False)
            feat_matrix = vectorizer.fit_transform(feat_dicts)
        # In assessment, we featurize using the existing vectorizer:
        else:
            feat_matrix = vectorizer.transform(feat_dicts)
    else:
        feat_matrix = feat_dicts
    return {'X': feat_matrix,
            'y': labels,
            'vectorizer': vectorizer,
            'raw_examples': raw_examples}


def experiment(
        sst_home,
        phi,
        train_func,
        train_reader=train_reader,
        assess_reader=None,
        train_size=0.7,
        class_func=ternary_class_func,
        score_func=utils.safe_macro_f1,
        vectorize=True,
        verbose=True,
        random_state=None):
    """Generic experimental framework for SST. Either assesses with a
    random train/test split of `train_reader` or with `assess_reader` if
    it is given.

    Parameters
    ----------
    sst_home : str
        Full path to the 'trees' directory for SST.
    phi : feature function
        Any function that takes an `nltk.Tree` instance as input
        and returns a bool/int/float-valued dict as output.
    train_func : model wrapper
        Any function that takes a feature matrix and a label list
        as its values and returns a fitted model with a `predict`
        function that operates on feature matrices.
    train_reader : SST iterator (default: `train_reader`)
        Iterator for training data.
    assess_reader : iterator or None (default: None)
        If None, then the data from `train_reader` are split into
        a random train/test split, with the the train percentage
        determined by `train_size`. If not None, then this should
        be an iterator for assessment data (e.g., `dev_reader`).
    train_size : float (default: 0.7)
        If `assess_reader` is None, then this is the percentage of
        `train_reader` devoted to training. If `assess_reader` is
        not None, then this value is ignored.
    class_func : function on the SST labels
        Any function like `binary_class_func` or `ternary_class_func`.
        This modifies the SST labels based on the experimental
        design. If `class_func` returns None for a label, then that
        item is ignored.
    score_metric : function name (default: `utils.safe_macro_f1`)
        This should be an `sklearn.metrics` scoring function. The
        default is weighted average F1 (macro-averaged F1). For
        comparison with the SST literature, `accuracy_score` might
        be used instead. For micro-averaged F1, use
          (lambda y, y_pred : f1_score(y, y_pred, average='micro', pos_label=None))
        For other metrics that can be used here, see
        see http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    vectorize : bool
       Whether to use a DictVectorizer. Set this to False for
       deep learning models that process their own input.
    verbose : bool (default: True)
        Whether to print out the model assessment to standard output.
        Set to False for statistical testing via repeated runs.
    random_state : int or None
        Optionally set the random seed for consistent sampling.

    Prints
    -------
    To standard output, if `verbose=True`
        Model precision/recall/F1 report. Accuracy is micro-F1 and is
        reported because many SST papers report that figure, but macro
        precision/recall/F1 is better given the class imbalances and the
        fact that performance across the classes can be highly variable.

    Returns
    -------
    dict with keys
        'model': trained model
        'phi': the function used for featurization
        'train_dataset': a dataset as returned by `build_dataset`
        'assess_dataset': a dataset as returned by `build_dataset`
        'predictions': predictions on the assessment data
        'metric': `score_func.__name__`
        'score': the `score_func` score on the assessment data

    """
    # Train dataset:
    train = build_dataset(
        sst_home,
        train_reader,
        phi,
        class_func,
        vectorizer=None,
        vectorize=vectorize)
    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    raw_train = train['raw_examples']
    if assess_reader == None:
        X_train, X_assess, y_train, y_assess, raw_train, raw_assess = train_test_split(
            X_train, y_train, raw_train,
            train_size=train_size, test_size=None, random_state=random_state)
        assess = {
            'X': X_assess,
            'y': y_assess,
            'vectorizer': train['vectorizer'],
            'raw_examples': raw_assess}
    else:
        # Assessment dataset using the training vectorizer:
        assess = build_dataset(
            sst_home,
            assess_reader,
            phi,
            class_func,
            vectorizer=train['vectorizer'],
            vectorize=vectorize)
        X_assess, y_assess = assess['X'], assess['y']
    # Train:
    mod = train_func(X_train, y_train)
    # Predictions:
    predictions = mod.predict(X_assess)
    # Report:
    if verbose:
        print(classification_report(y_assess, predictions, digits=3))
    # Return the overall score and experimental info:
    return {
        'model': mod,
        'phi': phi,
        'train_dataset': train,
        'assess_dataset': assess,
        'predictions': predictions,
        'metric': score_func.__name__,
        'score': score_func(y_assess, predictions)}


def compare_models(
        sst_home,
        phi1,
        train_func1,
        phi2=None,
        train_func2=None,
        vectorize1=True,
        vectorize2=True,
        stats_test=scipy.stats.wilcoxon,
        trials=10,
        reader=train_reader,
        train_size=0.7,
        class_func=ternary_class_func,
        score_func=utils.safe_macro_f1):
    """Wrapper for comparing models. The parameters are like those of
    `experiment`, with the same defaults, except

    Parameters
    ----------
    sst_home : str
        Full path to the 'trees' directory for SST.
    phi1, phi2
        Just like `phi` for `experiment`. `phi1` defaults to
        `unigrams_phi`. If `phi2` is None, then it is set equal
        to `phi1`.
    train_func1, train_func2
        Just like `train_func` for `experiment`. If `train_func2`
        is None, then it is set equal to `train_func`.
    vectorize1, vectorize1 : bool
        Whether to vectorize the respective inputs. Use `False` for
        deep learning models that featurize their own input.
    stats_test : scipy.stats function
        Defaults to `scipy.stats.wilcoxon`, a non-parametric version
        of the paired t-test.
    trials : int (default: 10)
        Number of runs on random train/test splits of `reader`,
        with `train_size` controlling the amount of training data.
    train_size : float
        Percentage of data o use for training.
    class_func : function on the SST labels
        Any function like `binary_class_func` or `ternary_class_func`.
        This modifies the SST labels based on the experimental
        design. If `class_func` returns None for a label, then that
        item is ignored.

    Prints
    ------
    To standard output
        A report of the assessment.

    Returns
    -------
    (np.array, np.array, float)
        The first two are the scores from each model (length `trials`),
        and the third is the p-value returned by stats_test.

    """
    if phi2 == None:
        phi2 = phi1
    if train_func2 == None:
        train_func2 = train_func1
    experiments1 = [experiment(sst_home,
        train_reader=reader,
        phi=phi1,
        train_func=train_func1,
        class_func=class_func,
        score_func=score_func,
        vectorize=vectorize1,
        verbose=False) for _ in range(trials)]
    experiments2 = [experiment(sst_home,
        train_reader=reader,
        phi=phi2,
        train_func=train_func2,
        class_func=class_func,
        score_func=score_func,
        vectorize=vectorize2,
        verbose=False) for _ in range(trials)]
    scores1 = np.array([d['score'] for d in experiments1])
    scores2 = np.array([d['score'] for d in experiments2])
    # stats_test returns (test_statistic, p-value). We keep just the p-value:
    pval = stats_test(scores1, scores2)[1]
    # Report:
    print('Model 1 mean: %0.03f' % scores1.mean())
    print('Model 2 mean: %0.03f' % scores2.mean())
    print('p = %0.03f' % pval if pval >= 0.001 else 'p < 0.001')
    # Return the scores for later analysis, and the p value:
    return (scores1, scores2, pval)


def build_rnn_dataset(sst_home, reader, class_func=ternary_class_func):
    """Given an SST reader, return the `class_func` version of the
    dataset as  (X, y) training pair.

    Parameters
    ----------
    sst_home : str
        Full path to the 'trees' directory for SST.
    reader : train_reader or dev_reader
    class_func : function on the SST labels

    Returns
    -------
    X, y
       Where X is a list of list of str, and y is the output label list.

    """
    r = reader(sst_home, class_func=class_func)
    data = [(tree.leaves(), label) for tree, label in r]
    X, y = zip(*data)
    return list(X), list(y)


def build_tree_dataset(sst_home, reader, class_func=ternary_class_func):
    """Given an SST reader, return the `class_func` version of the
    dataset. The root node of each tree (`tree.label()`) is set to
    the class for that tree. We also return the label vector for
    assessment.

    Parameters
    ----------
    sst_home : str
        Full path to the 'trees' directory for SST.
    reader : train_reader or dev_reader
    class_func : function on the SST labels

    Returns
    -------
    X, y
        Where X is a list of `nltk.tree.Tree`, and y is the output
        label list.

    """
    data = []
    labels = []
    for (tree, label) in reader(sst_home, class_func=class_func):
        tree.set_label(label)
        data.append(tree)
        labels.append(label)
    return data, labels
