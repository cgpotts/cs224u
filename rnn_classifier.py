import sys
import numpy as np
import random
from collections import defaultdict
import copy
from utils import randvec, randmatrix, d_tanh, softmax, progress_bar

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2018"


class RNNClassifier:
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
    def __init__(self,
            vocab,
            embedding,
            hidden_dim=20,
            eta=0.01,
            max_iter=100,
            tol=1.5e-8,
            display_progress=True):
        """
        Parameters
        ----------
        vocab : list of str
            This should be the vocabulary. It needs to be aligned with
            `embedding` in the sense that the ith element of vocab
            should be represented by the ith row of `embedding`.
        embedding : np.array
            Each row represents a word in `vocab`, as described above.
        hidden_dim : int (default: 10)
            Dimensionality for the hidden layer.
        eta : float (default: 0.05)
            Learning rate.
        max_iter : int (default: 100)
            Maximum number of training epochs for SGD.
        tol : float (default: 1.5e-8)
            Training terminates if the error reaches this point (or
            `max_iter` is met).
        display_progress : bool (default: True)
            Whether to print progress reports to stderr.

        All of the above are set as attributes. In addition, `self.word_dim`
        is set to the dimensionality of the input representations.

        """
        self.vocab = dict(zip(vocab, range(len(vocab))))
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.display_progress = display_progress
        self.word_dim = len(embedding[0])

    def get_word_rep(self, w):
        """For getting the input representation of word `w` from `self.embedding`."""
        word_index = self.vocab[w]
        return self.embedding[word_index]

    def fit(self, X, y):
        """Train the network.

        Parameters
        ----------
        X : list of lists
           Each element should be a list of elements in `self.vocab`.
        y : list
           The one-hot label vector.

        Attributes
        ----------
        self.output_dim : int
            Set based on the length of the labels in `training_data`.
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
        y = self.prepare_output_data(y)
        self.W_xh = self.weight_init(self.word_dim, self.hidden_dim)
        self.W_hh = self.weight_init(self.hidden_dim, self.hidden_dim)
        self.W_hy = self.weight_init(self.hidden_dim, self.output_dim)
        self.b = np.zeros(self.output_dim)
        # Unified view for training
        training_data = list(zip(X, y))
        # SGD:
        iteration = 0
        error = sys.float_info.max
        while error > self.tol and iteration < self.max_iter:
            error = 0.0
            random.shuffle(training_data)
            for seq, labels in training_data:
                self._forward_propagation(seq)
                # Cross-entropy error reduces to log(prediction-for-correct-label):
                error += -np.log(self.y[np.argmax(labels)])
                # Back-prop:
                d_W_hy, d_b, d_W_hh, d_W_xh = self._backward_propagation(seq, labels)
                # Updates:
                self.W_hy -= self.eta * d_W_hy
                self.b -= self.eta * d_b
                self.W_hh -= self.eta * d_W_hh
                self.W_xh -= self.eta * d_W_xh
            iteration += 1
            if self.display_progress:
                # Report the average error:
                error /= len(training_data)
                progress_bar("Finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, error))
        if self.display_progress:
            sys.stderr.write('\n')

    def _forward_propagation(self, seq):
        """
        Parameters
        ----------
        seq : list
            Variable length sequence of elements in the vocabulary.

        Attributes
        ----------
        self.h : np.array
            Each row is for a hidden representation. The first row
            is an all-0 initial state. The others correspond to
            the inputs in seq.

        self.y : np.array
            The vector of predictions.
        """
        self.h = np.zeros((len(seq)+1, self.hidden_dim))
        for t in range(1, len(seq)+1):
            word_rep = self.get_word_rep(seq[t-1])
            self.h[t] = np.tanh(
                word_rep.dot(self.W_xh) + self.h[t-1].dot(self.W_hh))
        self.y = softmax(self.h[-1].dot(self.W_hy) + self.b)

    def _backward_propagation(self, seq, y_):
        """
        Parameters
        ----------
        seq : list
            Variable length sequence of elements in the vocabulary. This
            is needed both for its lengths and for its input representations.

        y_ : list
            The label vector.

        Returns
        -------
        tuple
            The matrices of derivatives (d_W_hy, d_b, d_W_hh, d_W_xh).

        """
        # Output errors:
        y_err = self.y
        y_err[np.argmax(y_)] -= 1
        h_err = y_err.dot(self.W_hy.T) * d_tanh(self.h[-1])
        d_W_hy = np.outer(self.h[-1], y_err)
        d_b = y_err
        # For accumulating the gradients through time:
        d_W_hh = np.zeros(self.W_hh.shape)
        d_W_xh = np.zeros(self.W_xh.shape)
        # Back-prop through time; the +1 is because the 0th
        # hidden state is the all-0s initial state.
        num_steps = len(seq)+1
        for t in reversed(range(1, num_steps)):
            d_W_hh += np.outer(self.h[t], h_err)
            word_rep = self.get_word_rep(seq[t-1])
            d_W_xh += np.outer(word_rep, h_err)
            h_err = h_err.dot(self.W_hh.T) * d_tanh(self.h[t])
        return (d_W_hy, d_b, d_W_hh, d_W_xh)

    def predict_one_proba(self, seq):
        """Softmax predictions for a single example.

        Parameters
        ----------
        seq : list
            Variable length sequence of elements in the vocabulary.

        Returns
        -------
        np.array

        """
        self._forward_propagation(seq)
        return self.y

    def predict_proba(self, X):
        """Softmax predictions for a list of examples.

        Parameters
        ----------
        X : list of lists
            List of examples, each of which should be a list of items
            from `self.vocab`.

        Returns
        -------
        list of np.array

        """
        return [self.predict_one_proba(seq) for seq in X]

    def predict(self, X):
        """Predictions for a list of examples.

        Parameters
        ----------
        X : list of lists
            List of examples, each of which should be a list of items
            from `self.vocab`.

        Returns
        -------
        list

        """
        return [self.predict_one(seq) for seq in X]

    def predict_one(self, seq):
        """Predictions for a single example.

        Parameters
        ----------
        seq : list
            Variable length sequence of elements in the vocabulary.

        Returns
        -------
        int
            The index of the highest probability class according to
            the model.

        """
        probs = self.predict_one_proba(seq)
        return self.classes[np.argmax(self.y)]

    @staticmethod
    def weight_init(m, n):
        """Uses the Xavier Glorot method for initializing the weights
        of an `m` by `n` matrix.
        """
        x = np.sqrt(6.0/(m+n))
        return randmatrix(m, n, lower=-x, upper=x)

    def prepare_output_data(self, y):
        """Format `y` into a vector of one-hot encoded vectors.

        Parameters
        ----------
        y : list

        Returns
        -------
        np.array with length the same as y and each row the
        length of the number of classes

        """
        self.classes = sorted(set(y))
        self.output_dim = len(self.classes)
        y = self._onehot_encode(y)
        return y

    def _onehot_encode(self, y):
        classmap = dict(zip(self.classes, range(self.output_dim)))
        y_ = np.zeros((len(y), self.output_dim))
        for i, cls in enumerate(y):
            y_[i][classmap[cls]] = 1.0
        return y_


def simple_example():
    vocab = ['a', 'b']

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

    embedding = np.array([randvec(10) for _ in vocab])

    mod = RNNClassifier(
        vocab=vocab,
        embedding=embedding,
        max_iter=100)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, _ = zip(*test)
    print('\nPredictions:', mod.predict(X_test))


if __name__ == '__main__':
    simple_example()
