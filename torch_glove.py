import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch_model_base import TorchModelBase
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


class TorchGloVeDataset(torch.utils.data.Dataset):
    def __init__(self, X_log, weights):
        """
        Dataset class for GloVe. A separate class is needed only because,
        for batching, we need the batch indices for the `forward` method in
        `TorchGloVeModel`.

        For details on the construction of the two tensor arguments,
        see `TorchGloVe.fit`.

        Parameters
        ----------
        X_log : torch.FloatTensor, shape `(n_vocab, n_vocab)`

        weights : torch.FloatTensor, shape `(n_vocab, n_vocab)`

        """
        self.X_log = X_log
        self.weights = weights
        assert len(self.X_log) == len(self.weights)

    def __len__(self):
        return len(self.X_log)

    def __getitem__(self, idx):
        return self.X_log[idx], idx, self.weights[idx]


class TorchGloVeModel(nn.Module):
    def __init__(self, n_words, embed_dim):
        super().__init__()
        self.n_words = n_words
        self.embed_dim = embed_dim
        self.W = self._init_weights(self.n_words, self.embed_dim)
        self.C = self._init_weights(self.n_words, self.embed_dim)
        self.bw = self._init_weights(self.n_words, 1)
        self.bc = self._init_weights(self.n_words, 1)

    def _init_weights(self, m, n):
        return nn.Parameter(
            xavier_uniform_(torch.empty(m, n)))

    def forward(self, X_log, idx):
        """
        Parameters
        ----------
        X_log : torch.FloatTensor, shape `(batch_size, n_vocab)`.

        idx : torch.LongTensor, shape `(batch_size, )`
            Indices of the vocab items in the current batch.

        Returns
        -------
        torch.FloatTensor, shape `(n_vocab, n_vocab)`.

        """
        preds = self.W[idx].matmul(self.C.T) + self.bw[idx] + self.bc.T
        diffs = preds - X_log
        return diffs


class TorchGloVeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduction = "sum"

    def forward(self, diffs, weights):
        return torch.sum(0.5 * torch.mul(weights, diffs**2))


class TorchGloVe(TorchModelBase):
    def __init__(self, embed_dim=100, alpha=0.75, xmax=100, **model_kwargs):
        """
        Defines the GloVe computation graph using the vectorized
        method given in

        Dingwall, Nicholas and Christopher Potts. 2018. Mittens: An
        extension of GloVe for learning domain-specialized representations.
        Proceedings of the 2018 Conference of the North American Chapter
        of the Association for Computational Linguistics: Human Language
        Technologies, 212-217.

        """
        super().__init__(**model_kwargs)
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.xmax = xmax
        # We can't actually split the data into train and test portions,
        # but we can evaluate how we're doing holistically to try to
        # find the best model:
        if model_kwargs.get("early_stopping"):
            self.validation_fraction = 1.0
        self.loss = TorchGloVeLoss()
        self.params += ['embed_dim', 'alpha', 'xmax']

    def build_dataset(self, X_log, weights):
        """
        Creates a simple `TorchGloVeDataset`, which is really just a
        TensorDataset that returns the examples indices as well as the
        core batch dataset structrues, so that we can do batch updates
        to the GloVe parameters.

        Parameters
        ----------
        X_log : torch.FloatTensor, shape `(n_vocab, n_vocab)`.
            The log of the count matrix, ignoring 0s. See `fit.

        weights : torch.FloatTensor, shape `(n_vocab, n_vocab)`

        Returns
        -------
        TorchGloVeDataset

        """
        X_log = torch.FloatTensor(X_log)
        weights = torch.FloatTensor(weights)
        dataset = TorchGloVeDataset(X_log, weights)
        return dataset

    def build_graph(self):
        """
        The core computation graph. Called by `fit` to set the
        attribute `model`.

        Returns
        -------
        TorchGloVeModel

        """
        return TorchGloVeModel(self.n_words, self.embed_dim)

    def fit(self, X):
        """
        Prepares `X` to permit learning against the GloVe objective,
        and then uses the superclass `fit` method to train the model
        parameters. Unlike the supervised models in this repository,
        this method returns the learned embedding (W + C) rather than
        `self`, so that it acts like a model that transforms a vector
        space (see also the autoencoder models).

        Parameters
        ----------
        X : np.array, shape `(n_words, n_words)`
            This should be a square matrix of possible scaled
            co-occurrence counts.

        Attributes
        ----------
        self.embedding: np.array, shape (n_words, embed_dim)
            The same matrix that is returned by the method.

        Returns
        -------
        embedding: np.array, shape (n_words, embed_dim)
            The same matrix that is stored as `self.embedding`.

        """
        X_vals = self.convert_input_to_array(X)
        self.n_words = len(X_vals)
        # This applies the function
        #
        #  f(x) = (x/self.xmax)**self.alpha if x < self.xmax, else 1.0
        #
        # to the full count matrix:
        bounded = np.minimum(X_vals, self.xmax)
        weights = (bounded / self.xmax)**self.alpha
        # Precompute log X[i, j] for all i, j:
        X_log = utils.log_of_array_ignoring_zeros(X_vals)
        super().fit(X_log, weights)
        # Per the advice in the paper, use the sum of the word and
        # context embeddings:
        embedding = self.model.W + self.model.C
        embedding = embedding.detach().cpu().numpy()
        # If the input was a `pd.DataFrame`, return one as well:
        self.embedding = self.convert_output(embedding, X)
        return self.embedding

    def score(self, X, y=None):
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
        G = (self.model.W + self.model.C).detach().cpu().numpy()
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
    import utils

    utils.fix_random_seeds()

    X = np.array([
        [4.,  4.,  2.,  0.],
        [4., 61.,  8., 18.],
        [2.,  8., 10.,  0.],
        [0., 18.,  0.,  5.]])

    mod = TorchGloVe(embed_dim=2, max_iter=1000)

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
