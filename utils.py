from collections import Counter
import csv
import logging
import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import sys
import os

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


def glove2dict(src_filename):
    """GloVe Reader.
    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.
    Returns
    -------
    dict
        Mapping words to their GloVe vectors.
    """
    data = {}
    with open(src_filename, encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data


def d_tanh(z):
    """The derivative of np.tanh. z should be a float or np.array."""
    return 1.0 - z**2


def softmax(z):
    """Softmax activation function. z should be a float or np.array."""
    # Increases numerical stability:
    t = np.exp(z - np.max(z))
    return t / np.sum(t)


def relu(z):
    return np.maximum(0, z)


def d_relu(z):
    return np.where(z > 0, 1, 0)


def randvec(n=50, lower=-0.5, upper=0.5):
    """Returns a random vector of length `n`. `w` is ignored."""
    return np.array([random.uniform(lower, upper) for i in range(n)])


def randmatrix(m, n, lower=-0.5, upper=0.5):
    """Creates an m x n matrix of random values in [lower, upper]"""
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)


def safe_macro_f1(y, y_pred):
    """Macro-averaged F1, forcing `sklearn` to report as a multiclass
    problem even when there are just two classes. `y` is the list of
    gold labels and `y_pred` is the list of predicted labels."""
    return f1_score(y, y_pred, average='macro', pos_label=None)


def progress_bar(msg):
    """Simple over-writing progress bar."""
    sys.stderr.write('\r')
    sys.stderr.write(msg)
    sys.stderr.flush()


def log_of_array_ignoring_zeros(M):
    """Returns an array containing the logs of the nonzero
    elements of M. Zeros are left alone since log(0) isn't
    defined.
    """
    log_M = M.copy()
    mask = log_M > 0
    log_M[mask] = np.log(log_M[mask])
    return log_M


def mcnemar(y_true, pred_a, pred_b):
    """McNemar's test using the chi2 distribution.

    Parameters
    ----------
    y_true : list of actual labels
    pred_a, pred_b : lists
        Predictions from the two systems being evaluated.
        Assumed to have the same length as `y_true`.

    Returns
    -------
    float, float (the test statistic and p value)

    """
    c01 = 0
    c10 = 0
    for y, a, b in zip(y_true, pred_a, pred_b):
        if a == y and b != y:
            c01 += 1
        elif a != y and b == y:
            c10 += 1
    stat = ((np.abs(c10 - c01) - 1.0)**2) / (c10 + c01)
    df = 1
    pval = stats.chi2.sf(stat, df)
    return stat, pval


def fit_classifier_with_crossvalidation(
        X, y, basemod, cv, param_grid, scoring='f1_macro', verbose=True):
    """Fit a classifier with hyperparameters set via cross-validation.

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
    verbose : bool
        Whether to print some summary information to standard output.

    Prints
    ------
    To standard output (if `verbose=True`)
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
    if verbose:
        print("Best params: {}".format(crossvalidator.best_params_))
        print("Best score: {0:0.03f}".format(crossvalidator.best_score_))
    # Return the best model found:
    return crossvalidator.best_estimator_


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


def create_pretrained_embedding(lookup, vocab):
    """Create an embedding matrix from a lookup and a specified vocab.

    Parameters
    ----------
    lookup : dict
        Must map words to their vector representations.
    vocab : list of str
        Words to create embeddings for.

    Returns
    -------
    np.array, list
        The np.array is an embedding for `vocab`, restricted to words
        that are in in `lookup`, and sorted alphabetically. The last
        vector is for $UNK if it is not already in both `lookup`
        and `vocab`. The list is the updated vocabulary. The words are
        sorted alphabetically, to align with the embedding, and $UNK is
        appended the end if it was not already in in both `lookup` and
        `vocab`.

    """
    vocab = sorted(set(lookup) & set(vocab))
    embedding = np.array([lookup[w] for w in vocab])
    if '$UNK' not in vocab:
        vocab.append("$UNK")
        embedding = np.vstack((embedding, randvec(embedding.shape[1])))
    return embedding, vocab


def tf_train_progress_logging():
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    def info_filter(logrec):
        return int('loss' in logrec.getMessage().lower())
    logging.getLogger('tensorflow').addFilter(info_filter)


def fix_random_seeds(
        seed=42,
        set_system=True,
        set_torch=True,
        set_tensorflow=True,
        set_torch_cudnn=True):
    """Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed to be set.
    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`
    set_tensorflow : bool
        Whether to set `tf.random.set_random_seed(seed)`
    set_torch : bool
        Whether to set `torch.manual_seed(seed)`
    set_torch_cudnn: bool
        Flag for whether to enable cudnn deterministic mode.
        Note that deterministic mode can have a performance impact, depending on your model.
        https://pytorch.org/docs/stable/notes/randomness.html

    Notes
    -----
    Even though the random seeds are explicitly set,
    the behavior may still not be deterministic (especially when a
    GPU is enabled), due to:

    * CUDA: There are some PyTorch functions that use CUDA functions
    that can be a source of non-determinism:
    https://pytorch.org/docs/stable/notes/randomness.html

    * PYTHONHASHSEED: On Python 3.3 and greater, hash randomization is
    turned on by default. This seed could be fixed before calling the
    python interpreter (PYTHONHASHSEED=0 python test.py). However, it
    seems impossible to set it inside the python program:
    https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program

    """
    import torch
    import tensorflow as tf
    # set system seed
    if set_system:
        np.random.seed(seed)
        random.seed(seed)

    # set torch seed
    if set_torch:
        torch.manual_seed(seed)

    # set torch cudnn backend
    if set_torch_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # set tf seed
    if set_tensorflow:
        tf.random.set_random_seed(seed)
