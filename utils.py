from collections import Counter
import csv
import logging
import numpy as np
import random
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import sys

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):
    """Reads in matrices from CSV or space-delimited files.

    Parameters
    ----------
    src_filename : str
        Full path to the file to read.

    delimiter : str (default: ',')
        Delimiter for fields in src_filename. Use delimter=' '
        for GloVe files.

    header : bool (default: True)
        Whether the file's first row contains column names.
        Use header=False for GloVe files.

    quoting : csv style (default: QUOTE_MINIMAL)
        Use the default for normal csv files and csv.QUOTE_NONE for
        GloVe files.

    Returns
    -------
    (np.array, list of str, list of str)
       The first member is a dense 2d Numpy array, and the second
       and third are lists of strings (row names and column names,
       respectively). The third (column names) is None if the
       input file has no header. The row names are assumed always
       to be present in the leftmost column.
    """
    reader = csv.reader(open(src_filename), delimiter=delimiter, quoting=quoting)
    colnames = None
    if header:
        colnames = next(reader)
        colnames = colnames[1: ]
    mat = []
    rownames = []
    for line in reader:
        rownames.append(line[0])
        mat.append(np.array(list(map(float, line[1: ]))))
    return (np.array(mat), rownames, colnames)


def build_glove(src_filename):
    """Wrapper for using `build` to read in a GloVe file as a matrix"""
    return build(src_filename, delimiter=' ', header=False, quoting=csv.QUOTE_NONE)


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
    with open(src_filename,  encoding='utf8') as f:
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


def sequence_length_report(X, potential_max_length=50):
    lengths = [len(ex) for ex in X]
    longer = len([x for x in lengths if x > potential_max_length])
    print("Max sequence length: {:,}".format(max(lengths)))
    print("Min sequence length: {:,}".format(min(lengths)))
    print("Mean sequence length: {:0.02f}".format(np.mean(lengths)))
    print("Median sequence length: {:0.02f}".format(np.median(lengths)))
    print("Sequences longer than {:,}: {:,} of {:,}".format(
            potential_max_length, longer, len(lengths)))


def evaluate_rnn(y, preds):
    """Because the RNN sequences get clipped as necessary based
    on the `max_length` parameter, they have to be realigned to
    get a classification report. This method does that, building
    in the assumption that any clipped tokens are assigned an
    incorrect label.

    Parameters
    ----------
    y : list of list of labels
    preds : list of list of labels

    Both of these lists need to have the same length, but the
    sequences they contain can vary in length.
    """
    labels = sorted({c for ex in y for c in ex})
    new_preds = []
    for gold, pred in zip(y, preds):
        delta = len(gold) - len(pred)
        if delta > 0:
            # Make a *wrong* guess for these clipped tokens:
            pred += [random.choice(list(set(labels)-{label}))
                     for label in gold[-delta: ]]
        new_preds.append(pred)
    labels = sorted({cls for ex in y for cls in ex} - {'OTHER'})
    data = {}
    data['classification_report'] = flat_classification_report(y, new_preds)
    data['f1_macro'] = flat_f1_score(y, new_preds, average='macro')
    data['f1_micro'] = flat_f1_score(y, new_preds, average='micro')
    data['f1'] = flat_f1_score(y, new_preds, average=None)
    data['precision_score'] = flat_precision_score(y, new_preds, average=None)
    data['recall_score'] = flat_recall_score(y, new_preds, average=None)
    data['accuracy'] = flat_accuracy_score(y, new_preds)
    data['sequence_accuracy_score'] = sequence_accuracy_score(y, new_preds)
    return data


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
