from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy
import scipy.spatial.distance
from scipy.stats import spearmanr
import torch
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


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
    """
    Tool for finding the nearest neighbors of `word` in `df` according
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
    """
    2d plot of `df` using t-SNE, with the points labeled by `df.index`,
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
    """
    Latent Semantic Analysis using pure scipy.

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


def hf_represent(batch_ids, model, layer=-1):
    """
    Encode a batch of sequences of ids using a Hugging Face
    Transformer-based model `model`. The model's `forward` method is
    `output_hidden_states=True`, and we get the hidden states from
    `layer`.


    Parameters
    ----------
    batch_ids : iterable, shape (n_examples, n_tokens)
        Sequences of indices into the model vocabulary.

    model : Hugging Face transformer model

    later : int
        The layer to return. This will get all the hidden states at
        this layer. `layer=0` gives the embedding, and `layer=-1`
        gives the final output states.

    Returns
    -------
    Tensor of shape `(n_examples, n_tokens, n_dimensions)`
       where `n_dimensions` is the dimensionality of the
       Transformer model

    """
    with torch.no_grad():
        reps = model(batch_ids, output_hidden_states=True)
        return reps.hidden_states[layer]


def hf_encode(text, tokenizer, add_special_tokens=False):
    """
    Get the indices for the tokens in `text` according to `tokenizer`.
    If no tokens can be obtained from `text`, then the tokenizer.unk_token`
    is used as the only token.

    Parameters
    ----------
    text: str

    tokenizer: Hugging Face tokenizer

    add_special_tokens : bool
        A Hugging Face parameter to the tokenizer.

    Returns
    -------
    torch.Tensor of shape `(1, m)`
        A batch of 1 example of `m` tokens`, where `m` is determined
        by `text` and the nature of `tokenizer`.

    """
    encoding = tokenizer.encode(
        text,
        add_special_tokens=add_special_tokens,
        return_tensors='pt')
    if encoding.shape[1] == 0:
        text = tokenizer.unk_token
        encoding = torch.tensor([[tokenizer.vocab[text]]])
    return encoding


def mean_pooling(hidden_states):
    """
    Get the mean along `axis=1` of a Tensor.

    Parameters
    ----------
    hidden_states : torch.Tensor, shape `(k, m, n)`
        Where `k` is the number of examples, `m` is the number of vectors
        for each example, and `n` is dimensionality of each vector.

    Returns
    -------
    torch.Tensor of dimension `(k, n)`.

    """
    _check_pooling_dimensionality(hidden_states)
    return torch.mean(hidden_states, axis=1)


def max_pooling(hidden_states):
    """
    Get the max values along `axis=1` of a Tensor.

    Parameters
    ----------
    hidden_states : torch.Tensor, shape `(k, m, n)`
        Where `k` is the number of examples, `m` is the number of vectors
        for each example, and `n` is dimensionality of each vector.

    Raises
    ------
    ValueError
        If `hidden_states` does not have 3 dimensions.

    Returns
    -------
    torch.Tensor of dimension `(k, n)`.

    """
    _check_pooling_dimensionality(hidden_states)
    return torch.amax(hidden_states, axis=1)


def min_pooling(hidden_states):
    """
    Get the min values along `axis=1` of a Tensor.

    Parameters
    ----------
    hidden_states : torch.Tensor, shape `(k, m, n)`
        Where `k` is the number of examples, `m` is the number of vectors
        for each example, and `n` is dimensionality of each vector.

    Raises
    ------
    ValueError
        If `hidden_states` does not have 3 dimensions.

    Returns
    -------
    torch.Tensor of dimension `(k, n)`.

    """
    _check_pooling_dimensionality(hidden_states)
    return torch.amin(hidden_states, axis=1)


def last_pooling(hidden_states):
    """Get the final vector in second dimension (`axis=1`) of a Tensor.

    Parameters
    ----------
    hidden_states : torch.Tensor, shape (b, m, n)
       Where b is the number of examples, m is the number of vectors
       for each example, and `n` is dimensionality of each vector.

    Raises
    ------
    ValueError
        If `hidden_states` does not have 3 dimensions.

    Returns
    -------
    torch.Tensor of dimension `(k, n)`.

    """
    _check_pooling_dimensionality(hidden_states)
    return hidden_states[:, -1]


def _check_pooling_dimensionality(hidden_states):
     if not len(hidden_states.shape) == 3:
        raise ValueError(
            "The input to the pooling function should have 3 dimensions: "
            "it's a batch of k examples, where each example has m vectors, "
            "each of dimensionality n. The function will pool the vectors "
            "for each example, returning a Tensor of shape (k, n).")


def create_subword_pooling_vsm(vocab, tokenizer, model, layer=1, pool_func=mean_pooling):
    vocab_ids = [hf_encode(w, tokenizer) for w in vocab]
    vocab_hiddens = [hf_represent(w, model, layer=layer) for w in vocab_ids]
    pooled = [pool_func(h) for h in vocab_hiddens]
    pooled = [p.squeeze().cpu().numpy() for p in pooled]
    return pd.DataFrame(pooled, index=vocab)


def word_relatedness_evaluation(dataset_df, vsm_df, distfunc=cosine):
    """
    Main function for word relatedness evaluations used in the assignment
    and bakeoff. The function makes predictions for word pairs in
    `dataset_df` using `vsm_df` and `distfunc`, and it returns a copy of
    `dataset_df` with a new column `'prediction'`, as well as the Spearman
    rank correlation between those preductions and the `'score'` column
    in `dataset_df`.

    The prediction for a word pair (w1, w1) is determined by applying
    `distfunc` to the representations of w1 and w2 in `vsm_df`. We return
    the negative of this value since it is assumed that `distfunc` is a
    distance function and the scores in `dataset_df` are for positive
    relatedness.

    Parameters
    ----------
    dataset_df : pd.DataFrame
        Required to have columns {'word1', 'word2', 'score'}.

    vsm_df : pd.DataFrame
        The vector space model used to get representations for the
        words in `dataset_df`. The index must contain every word
        represented in `dataset_df`.

    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`,
        `matching`, `jaccard`, as well as any other distance measure
        between 1d vectors.

    Raises
    ------
    ValueError
        If any words in `dataset_df` are not in the index of `vsm_df`.

    Returns
    -------
    tuple (dataset_df, rho)
        Where `dataset_df` is a `pd.DataFrame` -- a copy of the
        input with a new column `'prediction'` -- and `rho` is a float
        giving the Spearman rank correlation between the `'score'`
        and `prediction` values.

    """
    dataset_df = dataset_df.copy()

    dataset_vocab = set(dataset_df.word1.values) | set(dataset_df.word2.values)

    vsm_vocab = set(vsm_df.index)

    missing = dataset_vocab - vsm_vocab

    if missing:
        raise ValueError(
            "The following words are in the evaluation dataset but not in the "
            "VSM. Please switch to a VSM with an appropriate vocabulary:\n"
            "{}".format(sorted(missing)))

    def predict(row):
        x1 = vsm_df.loc[row.word1]
        x2 = vsm_df.loc[row.word2]
        return -distfunc(x1, x2)

    dataset_df['prediction'] = dataset_df.apply(predict, axis=1)

    rho = None

    if 'score' in dataset_df.columns:
        rho, pvalue = spearmanr(
            dataset_df.score.values,
            dataset_df.prediction.values)

    return dataset_df, rho
