from collections import OrderedDict
import numpy as np
from np_model_base import NNModelBase
from utils import softmax

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


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

    Parameters
    ----------
    vocab : list of str
        This should be the vocabulary. It needs to be aligned with
        `embedding` in the sense that the ith element of vocab
        should be represented by the ith row of `embedding`. Ignored
        if `use_embedding=False`.
    embedding : np.array or None
        Each row represents a word in `vocab`, as described above.
    use_embedding : bool
        If True, then incoming examples are presumed to be lists of
        elements of the vocabulary. If False, then they are presumed
        to be lists of vectors. In this case, the `embedding` and
       `embed_dim` arguments are ignored, since no embedding is needed
       and `embed_dim` is set by the nature of the incoming vectors.
    embed_dim : int
        Dimensionality for the initial embeddings. This is ignored
        if `embedding` is not None, as a specified value there
        determines this value. Also ignored if `use_embedding=False`.

    All of the above are set as attributes. In addition, `self.embed_dim`
    is set to the dimensionality of the input representations.

    """
    def __init__(self,
            vocab,
            embedding=None,
            use_embedding=True,
            embed_dim=50,
            **kwargs):
        self.vocab = vocab
        self.vocab_lookup = dict(zip(self.vocab, range(len(self.vocab))))
        self.use_embedding = use_embedding
        if self.use_embedding:
            if embedding is None:
                embedding = self._define_embedding_matrix(
                    len(self.vocab), embed_dim)
            self.embedding = embedding
            self.embed_dim = self.embedding.shape[1]
        super(RNNClassifier, self).__init__(**kwargs)
        self.params += ['embedding', 'embed_dim']

    def fit(self, X, y):
        if not self.use_embedding:
            self.embed_dim = len(X[0][0])
        return super(RNNClassifier, self).fit(X, y)

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
            if self.use_embedding:
                word_rep = self.get_word_rep(seq[t-1])
            else:
                word_rep = seq[t-1]
            h[t] = self.hidden_activation(
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
        h_err = y_err.dot(self.W_hy.T) * self.d_hidden_activation(h[-1])
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
            if self.use_embedding:
                word_rep = self.get_word_rep(seq[t-1])
            else:
                 word_rep = seq[t-1]
            d_W_xh += np.outer(word_rep, h_err)
            h_err = h_err.dot(self.W_hh.T) * self.d_hidden_activation(h[t])
        return (d_W_hy, d_b, d_W_hh, d_W_xh)

    def update_parameters(self, gradients):
        d_W_hy, d_b, d_W_hh, d_W_xh = gradients
        self.W_hy -= self.eta * d_W_hy
        self.b -= self.eta * d_b
        self.W_hh -= self.eta * d_W_hh
        self.W_xh -= self.eta * d_W_xh


def simple_example(initial_embedding=False, use_embedding=True):
    vocab = ['a', 'b', '$UNK']

    # No b before an a
    train = [
        [list('ab'), 'good'],
        [list('aab'), 'good'],
        [list('abb'), 'good'],
        [list('aabb'), 'good'],
        [list('ba'), 'bad'],
        [list('baa'), 'bad'],
        [list('bba'), 'bad'],
        [list('bbaa'), 'bad'],
        [list('aba'), 'bad']
    ]

    test = [
        [list('baaa'), 'bad'],
        [list('abaa'), 'bad'],
        [list('bbaa'), 'bad'],
        [list('aaab'), 'good'],
        [list('aaabb'), 'good']
    ]

    if initial_embedding:
        import numpy as np
        # `embed_dim=60` to make sure that it gets changed internally:
        embedding = np.random.uniform(
            low=-1.0, high=1.0, size=(len(vocab), 60))
    else:
        embedding = None

    mod = RNNClassifier(
        vocab=vocab,
        max_iter=100,
        embedding=embedding,
        use_embedding=use_embedding,
        embed_dim=50,
        hidden_dim=50)

    X, y = zip(*train)
    X_test, y_test = zip(*test)

    # Just to illustrate how we can process incoming sequences of
    # vectors, we create an embedding and use it to preprocess the
    # train and test sets:
    if not use_embedding:
        import numpy as np
        from copy import copy
        # `embed_dim=60` to make sure that it gets changed internally:
        embedding = np.random.uniform(
            low=-1.0, high=1.0, size=(len(vocab), 60))
        X = [[embedding[vocab.index(w)] for w in ex] for ex in X]
        # So we can display the examples sensibly:
        X_test_orig = copy(X_test)
        X_test = [[embedding[vocab.index(w)] for w in ex] for ex in X_test]
    else:
        X_test_orig = X_test

    mod.fit(X, y)

    preds = mod.predict(X_test)

    print("\nPredictions:")

    for ex, pred, gold in zip(X_test_orig, preds, y_test):
        score = "correct" if pred == gold else "incorrect"
        print("{0:>6} - predicted: {1:>4}; actual: {2:>4} - {3}".format(
            "".join(ex), pred, gold, score))


if __name__ == '__main__':
    simple_example()
