import itertools
import numpy as np
import pandas as pd
import random
import sys
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


class GloVe:
    def __init__(self,
            n=100,
            xmax=100,
            alpha=0.75,
            max_iter=100,
            eta=0.05,
            tol=1e-5,
            display_progress=True):
        """
        Basic GloVe. This is mainly here as a reference implementation.
        We recommend using `torch_glove.py` instead.

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

        """
        self.n = n
        self.xmax = xmax
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta = eta
        self.tol = tol
        self.display_progress = display_progress

    def fit(self, df):
        """
        Learn the GloVe matrix.

        Parameters
        ----------
        df : pd.DataFrame or np.array, shape `(n_vocab, n_vocab)`
            This should be a matrix of (possibly scaled) co-occcurrence
            counts.

        Returns
        -------
        pd.DataFrame or np.array, shape `(n_vocab, self.n)`
           The type will be the same as the user's `df`. If it's a
           `pd.DataFrame`, the index will be the same as `df.index`.

        """
        X = self.convert_input_to_array(df)
        m = X.shape[0]
        # Parameters:
        W = utils.randmatrix(m, self.n)  # Word weights.
        C = utils.randmatrix(m, self.n)  # Context weights.
        B = utils.randmatrix(2, m)  # Word and context biases.
        # Precomputable GloVe values:
        X_log = utils.log_of_array_ignoring_zeros(X)
        X_weights = (np.minimum(X, self.xmax) / self.xmax)**self.alpha  # eq. (9)
        # Learning:
        indices = list(range(m))
        for iteration in range(self.max_iter):
            epoch_error = 0.0
            random.shuffle(indices)
            for i, j in itertools.product(indices, indices):
                if X[i, j] > 0.0:
                    weight = X_weights[i,j]
                    # Cost is J' based on eq. (8) in the paper:
                    diff = W[i].dot(C[j]) + B[0, i] + B[1, j] - X_log[i, j]
                    fdiff = diff * weight
                    # Gradients:
                    wgrad = fdiff * C[j]
                    cgrad = fdiff * W[i]
                    wbgrad = fdiff
                    wcgrad = fdiff
                    # Updates:
                    W[i] -= self.eta * wgrad
                    C[j] -= self.eta * cgrad
                    B[0, i] -= self.eta * wbgrad
                    B[1, j] -= self.eta * wcgrad
                    # One-half squared error term:
                    epoch_error += 0.5 * weight * (diff**2)

            epoch_error /= m

            if epoch_error <= self.tol:
                utils.progress_bar(
                    "Converged on iteration {} with error {}".format(
                        iteration, epoch_error, self.display_progress))
                break

            utils.progress_bar(
                "Finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, epoch_error, self.display_progress))

        # Return the sum of the word and context matrices, per the advice
        # in section 4.2:
        G = W + C
        self.embedding = self.convert_output(G, df)
        return self.embedding

    def score(self, X):
        """
        The goal of GloVe is to learn vectors whose dot products are
        proportional to the log co-occurrence probability. This score
        method assesses that directly using the current `self.embedding`.

        Parameters
        ----------
        X : pd.DataFrame or np.array, shape `(self.n_words, self.n_vocab)`
            The original count matrix.

        Returns
        -------
        float
            The Pearson correlation.

        """
        X = self.convert_input_to_array(X)
        G =  self.convert_input_to_array(self.embedding)
        mask = X > 0
        M = G.dot(G.T)
        X_log = utils.log_of_array_ignoring_zeros(X)
        row_log_prob = np.log(X.sum(axis=1))
        row_log_prob = np.outer(row_log_prob, np.ones(X.shape[1]))
        prob = X_log - row_log_prob
        return np.corrcoef(prob[mask].ravel(), M[mask].ravel())[0, 1]

    def convert_input_to_array(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X

    @staticmethod
    def convert_output(X_pred, X):
        if isinstance(X, pd.DataFrame):
            X_pred = pd.DataFrame(X_pred, index=X.index)
        return X_pred



def simple_example():
    utils.fix_random_seeds()

    X = np.array([
        [4.,  4.,  2.,  0.],
        [4., 61.,  8., 18.],
        [2.,  8., 10.,  0.],
        [0., 18.,  0.,  5.]])

    mod = GloVe(n=2, max_iter=1000)

    print(mod)

    G = mod.fit(X)

    print("\nLearned vectors:")
    print(G)

    print("We expect the dot product of learned vectors "
          "to be proportional to the log co-occurrence probs. "
          "Let's see how close we came:")

    corr = mod.score(X)

    print("Pearson's R: {} ".format(corr))

    return corr


if __name__ == '__main__':
    simple_example()
