import codecs
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy
import scipy.spatial.distance
import sys
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


def euclidean(u, v):
    return scipy.spatial.distance.euclidean(u, v)


def vector_length(u):
    return np.sqrt(u.dot(u))


def length_norm(u):
    return u / vector_length(u)


def cosine(u, v):
    return scipy.spatial.distance.cosine(u, v)


def matching(u, v):
    return np.sum(np.minimum(u, v))


def jaccard(u, v):
    return 1.0 - (matching(u, v) / np.sum(np.maximum(u, v)))


def neighbors(word, df, distfunc=cosine):
    """Tool for finding the nearest neighbors of `word` in `df` according
    to `distfunc`. The comparisons are between row vectors.

    Parameters
    ----------
    word : str
        The anchor word. Assumed to be in `rownames`.
    df : pd.DataFrame
        The vector-space model.
    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`,
        `matching`, `jaccard`, as well as any other distance measure
        between 1d vectors.

    Raises
    ------
    ValueError
        If word is not in `df.index`.

    Returns
    -------
    pd.Series
        Ordered by closeness to `word`.

    """
    if word not in df.index:
        raise ValueError('{} is not in this VSM'.format(word))
    w = df.loc[word]
    dists = df.apply(lambda x: distfunc(w, x), axis=1)
    return dists.sort_values()


def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    oe = df / expected
    return oe


def pmi(df, positive=True):
    df = observed_over_expected(df)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df


def tfidf(df):
    # Inverse document frequencies:
    doccount = float(df.shape[1])
    freqs = df.astype(bool).sum(axis=1)
    idfs = np.log(doccount / freqs)
    idfs[np.isinf(idfs)] = 0.0  # log(0) = 0
    # Term frequencies:
    col_totals = df.sum(axis=0)
    tfs = df / col_totals
    return (tfs.T * idfs).T


def ngram_vsm(df, n=2):
    """Create a character-level VSM from `df`.

    Parameters
    ----------
    df : pd.DataFrame
    n : int
        The n-gram size.

    Returns
    -------
    pd.DataFrame
        This will have the same column dimensionality as `df`, but the
        rows will be expanded with representations giving the sum of
        all the original rows in `df` that contain that row's n-gram.

    """
    unigram2vecs = defaultdict(list)
    for w, x in df.iterrows():
        for c in get_character_ngrams(w, n):
            unigram2vecs[c].append(x)
    unigram2vecs = {c: np.array(x).sum(axis=0)
                    for c, x in unigram2vecs.items()}
    cf = pd.DataFrame(unigram2vecs).T
    cf.columns = df.columns
    return cf


def get_character_ngrams(w, n):
    """Map a word to its character-level n-grams, with boundary
    symbols '<w>' and '</w>'.

    Parameters
    ----------
    w : str
    n : int
        The n-gram size.

    Returns
    -------
    list of str

    """
    if n > 1:
        w = ["<w>"] + list(w) + ["</w>"]
    else:
        w = list(w)
    return ["".join(w[i: i+n]) for i in range(len(w)-n+1)]


def character_level_rep(word, cf, n=4):
    """Get a representation for `word` as the sum of all the
    representations of `n`grams that it contains, according to `cf`.

    Parameters
    ----------
    word : str
        The word to represent.
    cf : pd.DataFrame
        The character-level VSM (e.g, the output of `ngram_vsm`).
    n : int
        The n-gram size.

    Returns
    -------
    np.array

    """
    ngrams = get_character_ngrams(word, n)
    ngrams = [n for n in ngrams if n in cf.index]
    reps = cf.loc[ngrams].values
    return reps.sum(axis=0)


def tsne_viz(df, colors=None, output_filename=None, figsize=(40, 50), random_state=None):
    """2d plot of `df` using t-SNE, with the points labeled by `df.index`,
    aligned with `colors` (defaults to all black).

    Parameters
    ----------
    df : pd.DataFrame
        The matrix to visualize.
    colors : list of colornames or None (default: None)
        Optional list of colors for the vocab. The color names just
        need to be interpretable by matplotlib. If they are supplied,
        they need to have the same length as `df.index`. If `colors=None`,
        then all the words are displayed in black.
    output_filename : str (default: None)
        If not None, then the output image is written to this location.
        The filename suffix determines the image type. If `None`, then
        `plt.plot()` is called, with the behavior determined by the
        environment.
    figsize : (int, int) (default: (40, 50))
        Default size of the output in display units.
    random_state : int or None
        Optionally set the `random_seed` passed to `PCA` and `TSNE`.

    """
    # Colors:
    vocab = df.index
    if not colors:
        colors = ['black' for i in vocab]
    # Recommended reduction via PCA or similar:
    n_components = 50 if df.shape[1] >= 50 else df.shape[1]
    dimreduce = PCA(n_components=n_components, random_state=random_state)
    X = dimreduce.fit_transform(df)
    # t-SNE:
    tsne = TSNE(n_components=2, random_state=random_state)
    tsnemat = tsne.fit_transform(X)
    # Plot values:
    xvals = tsnemat[: , 0]
    yvals = tsnemat[: , 1]
    # Plotting:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(xvals, yvals, marker='', linestyle='')
    # Text labels:
    for word, x, y, color in zip(vocab, xvals, yvals, colors):
        try:
            ax.annotate(word, (x, y), fontsize=8, color=color)
        except UnicodeDecodeError:  ## Python 2 won't cooperate!
            pass
    # Output:
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
    else:
        plt.show()


def lsa(df, k=100):
    """Latent Semantic Analysis using pure scipy.

    Parameters
    ----------
    df : pd.DataFrame
       The matrix to operate on.
    k : int (default: 100)
        Number of dimensions to truncate to.

    Returns
    -------
    pd.DataFrame
        The SVD-reduced version of `df` with dimension (m x k), where
        m is the rowcount of mat and `k` is either the user-supplied
        k or the column count of `mat`, whichever is smaller.

    """
    rowmat, singvals, colmat = np.linalg.svd(df, full_matrices=False)
    singvals = np.diag(singvals)
    trunc = np.dot(rowmat[:, 0:k], singvals[0:k, 0:k])
    return pd.DataFrame(trunc, index=df.index)


def glove(df, n=100, xmax=100, alpha=0.75, max_iter=100, eta=0.05,
        tol=1e-4, display_progress=True):
    """Basic GloVe. This is mainly here as a reference implementation.
    We recommend using `mittens.GloVe` instead.

    Parameters
    ----------
    df : pd.DataFrame or np.array
        This must be a square matrix.
    n : int (default: 100)
        The dimensionality of the output vectors.
    xmax : int (default: 100)
        Words with frequency greater than this are given weight 1.0.
        Words with frequency under this are given weight (c/xmax)**alpha
        where c is their count in mat (see the paper, eq. (9)).
    alpha : float (default: 0.75)
        Exponent in the weighting function (see the paper, eq. (9)).
    max_iter : int (default: 100)
        Number of training epochs.
    eta : float (default: 0.05)
        Controls the rate of SGD weight updates.
    tol : float (default: 1e-4)
        Stopping criterion for the loss.
    display_progress : bool (default: True)
        Whether to print iteration number and current error to stdout.

    Returns
    -------
    pd.DataFrame
        With dimension `(df.shape[0], n)`

    """
    X = df.values if isinstance(df, pd.DataFrame) else df
    m = X.shape[0]
    # Parameters:
    W = utils.randmatrix(m, n)  # Word weights.
    C = utils.randmatrix(m, n)  # Context weights.
    B = utils.randmatrix(2, m)  # Word and context biases.
    # Precomputable GloVe values:
    X_log = utils.log_of_array_ignoring_zeros(X)
    X_weights = (np.minimum(X, xmax) / xmax)**alpha  # eq. (9)
    # Learning:
    indices = list(range(m))
    for iteration in range(max_iter):
        error = 0.0
        random.shuffle(indices)
        for i, j in itertools.product(indices, indices):
            if X[i,j] > 0.0:
                weight = X_weights[i,j]
                # Cost is J' based on eq. (8) in the paper:
                diff = W[i].dot(C[j]) + B[0,i] + B[1,j] - X_log[i,j]
                fdiff = diff * weight
                # Gradients:
                wgrad = fdiff * C[j]
                cgrad = fdiff * W[i]
                wbgrad = fdiff
                wcgrad = fdiff
                # Updates:
                W[i] -= eta * wgrad
                C[j] -= eta * cgrad
                B[0,i] -= eta * wbgrad
                B[1,j] -= eta * wcgrad
                # One-half squared error term:
                error += 0.5 * weight * (diff**2)
        error /= m
        if display_progress:
            if error < tol:
                utils.progress_bar(
                    "Stopping at iteration {} with "
                    "error {}".format(iteration, error))
                break
            else:
                utils.progress_bar(
                    "Iteration {}: error {}".format(iteration, error))
    if display_progress:
        sys.stderr.write('\n')
    # Return the sum of the word and context matrices, per the advice
    # in section 4.2:
    G = W + C
    if isinstance(df, pd.DataFrame):
        G = pd.DataFrame(G, index=df.index)
    return G
