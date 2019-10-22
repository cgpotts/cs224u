import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


class Retrofitter(object):
    """Implements the baseline retrofitting method of Faruqui et al.

    Parameters
    ----------
    max_iter : int indicating the maximum number of iterations to run.
    alpha : func from `edges.keys()` to floats or None
    beta : func from `edges.keys()` to floats or None
    tol : float
        If the average distance change between two rounds is at or
        below this value, we stop. Default to 10^-2 as suggested
        in the paper.
    verbose : bool
        Whether to print information about the optimization process.
    introspecting : bool
        Whether to accumulate a list of the retrofitting matrices
        at each step. This should be set to `True` only for small
        illustrative tasks. For large ones, it will impose huge
        memory demands.

    """
    def __init__(self, max_iter=100, alpha=None, beta=None, tol=1e-2,
            verbose=False, introspecting=False):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.verbose = verbose
        self.introspecting = introspecting

    def fit(self, X, edges):
        """The core internal retrofitting method.

        Parameters
        ----------
        X : np.array (distributional embeddings)
        edges : dict
            Mapping indices into `X` into sets of indices into `X`.

        Attributes
        ----------
        self.Y : np.array, same dimensions and arrangement as `X`.
           The retrofitting matrix.
        self.all_Y : list
           Set only if `self.introspecting=True`.

        Returns
        -------
        self

        """
        index = None
        columns = None
        if isinstance(X, pd.DataFrame):
            index = X.index
            columns = X.columns
            X = X.values

        if self.alpha is None:
            self.alpha = lambda x: 1.0
        if self.beta is None:
            self.beta = lambda x: 1.0 / len(edges[x])

        if self.introspecting:
            self.all_Y = []

        Y = X.copy()
        Y_prev = Y.copy()
        for iteration in range(1, self.max_iter+1):
            for i, vec in enumerate(X):
                neighbors = edges[i]
                n_neighbors = len(neighbors)
                if n_neighbors:
                    a = self.alpha(i)
                    b = self.beta(i)
                    retro = np.array([b * Y[j] for j in neighbors])
                    retro = retro.sum(axis=0) + (a * X[i])
                    norm = np.array([b for j in neighbors])
                    norm = norm.sum(axis=0) + a
                    Y[i] = retro / norm
            changes = self._measure_changes(Y, Y_prev)
            if changes <= self.tol:
                self._progress_bar(
                    "Converged at iteration {}; change was {:.4f} ".format(
                        iteration, changes))
                break
            else:
                if self.introspecting:
                    self.all_Y.append(Y.copy())
                Y_prev = Y.copy()
                self._progress_bar(
                    "Iteration {:d}; change was {:.4f}".format(
                        iteration, changes))
        if index is not None:
            Y = pd.DataFrame(Y, index=index, columns=columns)
        self.Y = Y
        return self.Y

    @staticmethod
    def _measure_changes(Y, Y_prev):
        return np.abs(
            np.mean(
                np.linalg.norm(
                    np.squeeze(Y_prev) - np.squeeze(Y),
                    ord=2)))

    def _progress_bar(self, msg):
        if self.verbose:
            utils.progress_bar(msg)



def plot_retro_vsm(Q, edges, ax=None, lims=None):
    ax = Q.plot.scatter(x=0, y=1, ax=ax)
    if lims is not None:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    _ = Q.apply(lambda x: ax.text(x[0], x[1], x.name, fontsize=18), axis=1)
    for i, vals in edges.items():
        for j in vals:
            x0, y0 = Q.iloc[i].values
            x1, y1 = (Q.iloc[j] - Q.iloc[i]) * 0.9
            ax.arrow(x0, y0, x1, y1, head_width=0.05, head_length=0.05)
    return ax


def plot_retro_path(Q_hat, edges, retrofitter=None):
    if retrofitter is None:
        retrofitter = Retrofitter(introspecting=True)
    retrofitter.introspecting = True
    retrofitter.fit(Q_hat, edges)
    all_Y = retrofitter.all_Y
    lims = [Q_hat.values.min()-0.1, Q_hat.values.max()+0.1]
    n_steps = len(all_Y)
    fig, axes = plt.subplots(nrows=1, ncols=n_steps+1, figsize=(12, 4), squeeze=False)
    plot_retro_vsm(Q_hat, edges, axes[0][0], lims=lims)
    for Q, ax in zip(all_Y, axes[0][1: ]):
        Q = pd.DataFrame(Q, index=Q_hat.index, columns=Q_hat.columns)
        ax = plot_retro_vsm(Q, edges, ax=ax, lims=lims)
    plt.tight_layout()
    return retrofitter
