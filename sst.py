from collections import Counter
from nltk.tree import Tree
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import scipy.stats
import utils


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2018 term"


SST_HOME = 'trees'


def sentiment_treebank_reader(
        src_filename,
        include_subtrees=False,
        replace_root_score=True,
        class_func=None):
    """Iterator for the Penn-style distribution of the Stanford
    Sentiment Treebank. The iterator yields (tree, label) pairs.

    The root node of the tree is the label, so the root node itself is
    replaced with a string to ensure that it doesn't get used as a
    predictor. The subtree labels are retained. If they are used, it can
    feel like cheating (see `root_daughter_scores_phis` below), so take
    care!

    The labels are strings. They do not make sense as a linear order
    because negative ('0', '1'), neutral ('2'), and positive ('3','4')
    do not form a linear order conceptually, and because '0' is
    stronger than '1' but '4' is stronger than '3'.

    Parameters
    ----------
    src_filename : str
        Full path to the file to be read.
    include_subtrees : boolean (default: False)
        Whether to yield all the subtrees with labels or just the full
        tree. In both cases, the label is the root of the subtree.
    replace_root_score : boolean (default: True)
        The root node of the tree is the label, so, by default, the root
        node itself is replaced with a string to ensure that it doesn't
        get used as a predictor.
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
    with open(src_filename) as f:
        for line in f:
            tree = Tree.fromstring(line)
            if include_subtrees:
                for subtree in tree.subtrees():
                    label = subtree.label()
                    label = class_func(label)
                    if label:
                        if replace_root_score:
                            subtree.set_label("X")
                        yield (subtree, label)
            else:
                label = tree.label()
                label = class_func(label)
                if label:
                    if replace_root_score:
                        tree.set_label("S")
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


def train_reader(**kwargs):
    """Convenience function for reading the train file, full-trees only."""
    src = os.path.join(SST_HOME, 'train.txt')
    return sentiment_treebank_reader(src,**kwargs)


def dev_reader(**kwargs):
    """Convenience function for reading the dev file, full-trees only."""
    src = os.path.join(SST_HOME, 'dev.txt')
    return sentiment_treebank_reader(src, **kwargs)


def test_reader(**kwargs):
    """Convenience function for reading the test file, full-trees only.
    This function should be used only for the final stages of a project,
    to obtain final results.
    """
    src = os.path.join(SST_HOME, 'test.txt')
    return sentiment_treebank_reader(src, **kwargs)


def allnodes_train_reader(**kwargs):
    """Convenience function for reading the train file, all nodes."""
    src = os.path.join(SST_HOME, 'train.txt')
    return sentiment_treebank_reader(src, include_subtrees=True, **kwargs)


def allnodes_dev_reader():
    """Convenience function for reading the dev file, all nodes."""
    src = os.path.join(SST_HOME, 'dev.txt')
    return sentiment_treebank_reader(src, include_subtrees=True, **kwargs)


def build_dataset(reader, phi, class_func, vectorizer=None, vectorize=True):
    """Core general function for building experimental datasets.

    Parameters
    ----------
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
    for tree, label in reader(class_func=class_func):
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
        phi,
        train_func,
        train_reader=train_reader,
        assess_reader=None,
        train_size=0.7,
        class_func=binary_class_func,
        score_func=utils.safe_macro_f1,
        vectorize=True,
        verbose=True):
    """Generic experimental framework for SST. Either assesses with a
    random train/test split of `train_reader` or with `assess_reader` if
    it is given.

    Parameters
    ----------
    phi : feature function
        Any function that takes an `nltk.Tree` instance as input
        and returns a bool/int/float-valued dict as output.
    train_func : model wrapper (default: `fit_maxent_classifier`)
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

    Prints
    -------
    To standard output, if `verbose=True`
        Model accuracy and a model precision/recall/F1 report. Accuracy is
        reported because many SST papers report that figure, but the
        precision/recall/F1 is better given the class imbalances and the
        fact that performance across the classes can be highly variable.

    Returns
    -------
    float
        The overall scoring metric as determined by `score_metric`.

    """
    # Train dataset:
    train = build_dataset(
        train_reader, phi, class_func, vectorizer=None, vectorize=vectorize)
    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    X_assess = None
    y_assess = None
    if assess_reader == None:
         X_train, X_assess, y_train, y_assess = train_test_split(
                X_train, y_train, train_size=train_size, test_size=None)
    else:
        # Assessment dataset using the training vectorizer:
        assess = build_dataset(
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
        print('Accuracy: %0.03f' % accuracy_score(y_assess, predictions))
        print(classification_report(y_assess, predictions, digits=3))
    # Return the overall score:
    return score_func(y_assess, predictions)


def fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid, scoring='f1_macro'):
    """Fit a classifier with hyperparmaters set via cross-validation.

    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
    y : list
        The list of labels for rows in `X`.
    basemod : an sklearn model class instance
        This is the basic model-type we'll be optimizing.
    cv : int
        Number of cross-validation folds.
    param_grid : dict
        A dict whose keys name appropriate parameters for `basemod` and
        whose values are lists of values to try.
    scoring : value to optimize for (default: f1_macro)
        Other options include 'accuracy' and 'f1_micro'. See
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    Prints
    ------
    To standard output:
        The best parameters found.
        The best macro F1 score obtained.

    Returns
    -------
    An instance of the same class as `basemod`.
        A trained model instance, the best model found.

    """
    # Find the best model within param_grid:
    crossvalidator = GridSearchCV(basemod, param_grid, cv=cv, scoring=scoring)
    crossvalidator.fit(X, y)
    # Report some information:
    print("Best params", crossvalidator.best_params_)
    print("Best score: %0.03f" % crossvalidator.best_score_)
    # Return the best model found:
    return crossvalidator.best_estimator_


def compare_models(
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

    Prints
    ------
    To standard output
        A report of the assessment.

    Returns
    -------
    (np.array, np.array, float)
        The first two are the scores from each model (length `trials`),
        and the third is the p-value returned by stats_test.

    TODO
    ----
    This function can easily be parallelized. The ParallelPython
    makes this easy:http://www.parallelpython.com

    """
    if phi2 == None:
        phi2 = phi1
    if train_func2 == None:
        train_func2 = train_func1
    scores1 = np.array([experiment(train_reader=reader,
        phi=phi1,
        train_func=train_func1,
        class_func=class_func,
        score_func=score_func,
        vectorize=vectorize1,
        verbose=False) for _ in range(trials)])
    scores2 = np.array([experiment(train_reader=reader,
        phi=phi2,
        train_func=train_func2,
        class_func=class_func,
        score_func=score_func,
        vectorize=vectorize2,
        verbose=False) for _ in range(trials)])
    # stats_test returns (test_statistic, p-value). We keep just the p-value:
    pval = stats_test(scores1, scores2)[1]
    # Report:
    print('Model 1 mean: %0.03f' % scores1.mean())
    print('Model 2 mean: %0.03f' % scores2.mean())
    print('p = %0.03f' % pval if pval >= 0.001 else 'p < 0.001')
    # Return the scores for later analysis, and the p value:
    return (scores1, scores2, pval)


def get_vocab(X, n_words=None):
    """Get the vocabulary for an RNN example matrix `X`,
    adding $UNK$ if it isn't already present.

    Parameters
    ----------
    X : list of lists of str
    n_words : int or None
        If this is `int > 0`, keep only the top `n_words` by frequency.

    Returns
    -------
    list of str

    """
    wc = Counter([w for ex in X for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    vocab = {w for w, c in wc}
    vocab.add("$UNK")
    return sorted(vocab)


def build_binary_rnn_dataset(reader):
    """Given an SST reader, return the binary version of the dataset
    as  (X, y) training pair.

    Parameters
    ----------
    reader : train_reader or dev_reader

    Returns
    -------
    X, y
       Where X is a list of list of str, and y is the output label list.

    """
    data = [(tree.leaves(), label) for tree, label in reader(class_func=binary_class_func)]
    X, y = zip(*data)
    return list(X), list(y)


def get_sentence_embedding_from_rnn(rnn, X):
    """Given a trained model `rnn` and a set of RNN examples `X` create
    a DataFrame of the final hidden representations.

    Parameters
    ----------
    rnn : `TfRNNClassifier` instance
    X : list of list of str
        With a vocab appropriate for `rnn`. This should probably be
        the same dataset as `rnn` was trained on.

    Returns
    -------
    pd.DataFrame

    """
    X_indexed, ex_lengths = rnn._convert_X(X)
    S = rnn.sess.run(
        rnn.last,
        {rnn.inputs: X_indexed, rnn.ex_lengths: ex_lengths})
    S = pd.DataFrame(S)
    return S
