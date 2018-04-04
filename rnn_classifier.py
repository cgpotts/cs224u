import sys
import numpy as np
import random
from nn_model_base import NNModelBase
from utils import randvec, d_tanh, softmax

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2018"


class RNNClassifier(NNModelBase):
    """Simple Recurrent Neural Network for classification problems.
    The structure of the network is as follows:

                                                  y
                                                 /|
                                                b | W_hy
                                                  |
    h_0 -- W_hh --  h_1 -- W_hh -- h_2 -- W_hh -- h_3
                     |              |              |
                     | W_xh         | W_xh         | W_xh
                     |              |              |
                    x_1            x_2            x_3

    where x_i are the inputs, h_j are the hidden units, and y is a
    one-hot vector indicating the true label for this sequence. The
    parameters are W_xh, W_hh, W_hy, and the bias b. The inputs x_i
    come from a user-supplied embedding space for the vocabulary. These
    can either be random or pretrained. The network equations in brief:

        h[t] = tanh(x[t].dot(W_xh) + h[t-1].dot(W_hh))

        y = softmax(h[-1].dot(W_hy) + b)

    The network will work for any kind of classification task.

    """
    def __init__(self, vocab, hidden_dim=20, **kwargs):
        """
        Parameters
        ----------
        vocab : list of str
            This should be the vocabulary. It needs to be aligned with
            `embedding` in the sense that the ith element of vocab
            should be represented by the ith row of `embedding`.
        embedding : np.array or None
            Each row represents a word in `vocab`, as described above.
        embed_dim : int
            Dimensionality for the initial embeddings. This is ignored
            if `embedding` is not None, as a specified value there
            determines this value.
        hidden_dim : int
            Dimensionality for the hidden layer.
        eta : float
            Learning rate.
        max_iter : int
            Maximum number of training epochs for SGD.
        tol : float
            Training terminates if the error reaches this point (or
            `max_iter` is met).
        display_progress : bool
            Whether to print progress reports to stderr.

        All of the above are set as attributes. In addition, `self.embed_dim`
        is set to the dimensionality of the input representations.

        """
        super(RNNClassifier, self).__init__(
            vocab, hidden_dim=hidden_dim, **kwargs)

    def initialize_parameters(self):
        """
        Attributes
        ----------
        self.output_dim : int
            Set based on the length of the labels in `training_data`.
            This happens in `self.prepare_output_data`.
        self.W_xh : np.array
            Dense connections between the word representations
            and the hidden layers. Random initialization.
        self.W_hh : np.array
            Dense connections between the hidden representations.
            Random initialization.
        self.W_hy : np.array
            Dense connections from the final hidden layer to
            the output layer. Random initialization.
        self.b : np.array
            Output bias. Initialized to all 0.

        """
        self.W_xh = self.weight_init(self.embed_dim, self.hidden_dim)
        self.W_hh = self.weight_init(self.hidden_dim, self.hidden_dim)
        self.W_hy = self.weight_init(self.hidden_dim, self.output_dim)
        self.b = np.zeros(self.output_dim)

    def forward_propagation(self, seq):
        """
        Parameters
        ----------
        seq : list
            Variable length sequence of elements in the vocabulary.

        Returns
        ----------
        h : np.array
            Each row is for a hidden representation. The first row
            is an all-0 initial state. The others correspond to
            the inputs in seq.
        y : np.array
            The vector of predictions.
        """
        h = np.zeros((len(seq)+1, self.hidden_dim))
        for t in range(1, len(seq)+1):
            word_rep = self.get_word_rep(seq[t-1])
            h[t] = np.tanh(
                word_rep.dot(self.W_xh) + h[t-1].dot(self.W_hh))
        y = softmax(h[-1].dot(self.W_hy) + self.b)
        return h, y

    def backward_propagation(self, h, predictions, seq, labels):
        """
        Parameters
        ----------
        h : np.array, shape (m, self.hidden_dim)
            Matrix of hidden states. `m` is the shape of the current
            example (which is allowed to vary).
        predictions : np.array, dimension `len(self.classes)`
            Vector of predictions.
        seq : list  of lists
            The original example.
        labels : np.array, dimension `len(self.classes)`
            One-hot vector giving the true label.
        Returns
        -------
        tuple
            The matrices of derivatives (d_W_hy, d_b, d_W_hh, d_W_xh).

        """
        # Output errors:
        y_err = predictions
        y_err[np.argmax(labels)] -= 1
        h_err = y_err.dot(self.W_hy.T) * d_tanh(h[-1])
        d_W_hy = np.outer(h[-1], y_err)
        d_b = y_err
        # For accumulating the gradients through time:
        d_W_hh = np.zeros(self.W_hh.shape)
        d_W_xh = np.zeros(self.W_xh.shape)
        # Back-prop through time; the +1 is because the 0th
        # hidden state is the all-0s initial state.
        num_steps = len(seq)+1
        for t in reversed(range(1, num_steps)):
            d_W_hh += np.outer(h[t], h_err)
            word_rep = self.get_word_rep(seq[t-1])
            d_W_xh += np.outer(word_rep, h_err)
            h_err = h_err.dot(self.W_hh.T) * d_tanh(h[t])
        return (d_W_hy, d_b, d_W_hh, d_W_xh)

    def update_parameters(self, gradients):
        d_W_hy, d_b, d_W_hh, d_W_xh = gradients
        self.W_hy -= self.eta * d_W_hy
        self.b -= self.eta * d_b
        self.W_hh -= self.eta * d_W_hh
        self.W_xh -= self.eta * d_W_xh


def simple_example():
    vocab = ['a', 'b', '$UNK']

    train = [
        [list('ab'), 'good'],
        [list('aab'), 'good'],
        [list('abb'), 'good'],
        [list('aabb'), 'good'],
        [list('ba'), 'bad'],
        [list('baa'), 'bad'],
        [list('bba'), 'bad'],
        [list('bbaa'), 'bad']]

    test = [
        [list('aaab'), 'good'],
        [list('baaa'), 'bad']]

    mod = RNNClassifier(vocab=vocab, max_iter=100)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, _ = zip(*test)
    print('\nPredictions:', mod.predict(X_test))


if __name__ == '__main__':
    simple_example()
