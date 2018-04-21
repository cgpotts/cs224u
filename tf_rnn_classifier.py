import numpy as np
import tensorflow as tf
from tf_model_base import TfModelBase
import warnings

__author__ = 'Chris Potts'

# Ignore the TensorFlow warning
#   Converting sparse IndexedSlices to a dense Tensor of unknown shape.
#   This may consume a large amount of memory.
warnings.filterwarnings("ignore", category=UserWarning)


class TfRNNClassifier(TfModelBase):
    """Defines an RNN in which the final hidden state is used as
    the basis for a softmax classifier predicting a label:

    h_t = tanh(x_tW_xh + h_{t-1}W_hh)
    y   = softmax(h_nW_hy + b)

    t <= 1 <= n and the initial state h_0 is set to all 0s.

    Parameters
    ----------
    vocab : list
        The full vocabulary. `_convert_X` will convert the data provided
        to `fit` and `predict` methods into a list of indices into this
        list of items.
    embedding : 2d np.array or None
        If `None`, then a random embedding matrix is constructed.
        Otherwise, this should be a 2d array aligned row-wise with
        `vocab`, with each row giving the input representation for the
        corresponding word. For instance, to roughly duplicate what
        is done by default, one could do
            `np.array([np.random.randn(h) for _ in vocab])`
        where n is the embedding dimensionality (`embed_dim`).
    embed_dim : int
        Dimensionality of the inputs/embeddings. If `embedding`
        is supplied, then this value is set to be the same as its
        column dimensionality. Otherwise, this value is used to create
        the embedding Tensor (see `_define_embedding`).
    max_length : int
        Maximum sequence length.
    train_embedding : bool
        Whether to update the embedding matrix when training.
    cell_class : tf.nn.rnn_cell class
       The default is `tf.nn.rnn_cell.LSTMCell`. Other prominent options:
       `tf.nn.rnn_cell.BasicRNNCell`, and `tf.nn.rnn_cell.GRUCell`.
    hidden_activation : tf.nn activation
       E.g., tf.nn.relu, tf.nn.relu, tf.nn.selu.
    hidden_dim : int
        Dimensionality of the hidden layer.
    max_iter : int
        Maximum number of iterations allowed in training.
    eta : float
        Learning rate.
    tol : float
        Stopping criterion for the loss.
    """
    def __init__(self,
            vocab,
            embedding=None,
            embed_dim=50,
            max_length=20,
            train_embedding=True,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            **kwargs):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.train_embedding = train_embedding
        self.cell_class = cell_class
        super(TfRNNClassifier, self).__init__(**kwargs)
        self.params += [
            'embedding', 'embed_dim', 'max_length', 'train_embedding']

    def build_graph(self):
        self._define_embedding()

        self.inputs = tf.placeholder(
            tf.int32, [None, self.max_length])

        self.ex_lengths = tf.placeholder(tf.int32, [None])

        # Outputs as usual:
        self.outputs = tf.placeholder(
            tf.float32, shape=[None, self.output_dim])

        # This converts the inputs to a list of lists of dense vector
        # representations:
        self.feats = tf.nn.embedding_lookup(
            self.embedding, self.inputs)

        # Defines the RNN structure:
        self.cell = self.cell_class(
            self.hidden_dim, activation=self.hidden_activation)

        # Run the RNN:
        outputs, state = tf.nn.dynamic_rnn(
            self.cell,
            self.feats,
            dtype=tf.float32,
            sequence_length=self.ex_lengths)

        # How can I be sure that I have found the last true state? This
        # first option seems to work for all cell types but sometimes
        # leads to indexing errors and is in general pretty complex:
        #
        # self.last = self._get_last_non_masked(outputs, self.ex_lengths)
        #
        # This option is more reliable, but is it definitely getting
        # the final true state?
        self.last = self._get_final_state(self.cell, state)

        # Softmax classifier on the final hidden state:
        self.W_hy = self.weight_init(
            self.hidden_dim, self.output_dim, 'W_hy')
        self.b_y = self.bias_init(self.output_dim, 'b_y')
        self.model = tf.matmul(self.last, self.W_hy) + self.b_y

    def train_dict(self, X, y):
        """Converts `X` to an np.array` using _convert_X` and feeds
        this to `inputs`, , and gets the true length of each example
        and passes it to `fit` as well. `y` is fed to `outputs`.

        Parameters
        ----------
        X : list of lists
        y : list

        Returns
        -------
        dict, list of int

        """
        X, ex_lengths = self._convert_X(X)
        return {self.inputs: X, self.ex_lengths: ex_lengths, self.outputs: y}

    def test_dict(self, X):
        """Converts `X` to an np.array` using _convert_X` and feeds
        this to `inputs`, and gets the true length of each example and
        passes it to `fit` as well.

        Parameters
        ----------
        X : list of lists
        y : list

        Returns
        -------
        dict, list of int

        """
        X, ex_lengths = self._convert_X(X)
        return {self.inputs: X, self.ex_lengths: ex_lengths}

    @staticmethod
    def _get_final_state(cell, state):
        """Get the final state from an RNN, managing differences in
        the TensorFlow API for cells.

        Parameters
        ----------
        cell : tf.nn.rnn_cell instance
        state : second argument returned by `tf.nn.dynamic_rnn`

        Returns
        -------
        Tensor

        """
        # If the cell is LSTMCell, then `state` is an `LSTMStateTuple`
        # and we want the second (output) Tensor -- see
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
        #
        if isinstance(cell, tf.nn.rnn_cell.LSTMCell):
            return state[1]
        else:
            # For other cell types, it seems we can just do this. I assume
            # that `state` is the last *true* state, not one of the
            # zero-padded ones (?).
            return state

    @staticmethod
    def _get_last_non_masked(outputs, lengths):
        """This method finds the last hidden state that is based on a
        non-null sequence element. It is adapted from

        https://danijar.com/variable-sequence-lengths-in-tensorflow/

        It's not currently being used, but it *might* be a more surefire
        way of ensuring that one retrieves the last true state. Compare
        with `_get_final_state.

        Parameters
        ----------
        outputs : a 3d Tensor of hidden states
        lengths : a 1d Tensor of ints

        Returns
        -------
        A 1d tensor, the last element of outputs that is based on a
        non-null input.

        """
        batch_size = tf.shape(outputs)[0]
        max_length = int(outputs.get_shape()[1])
        output_size = int(outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (lengths - 1)
        flat = tf.reshape(outputs, [-1, output_size])
        last = tf.gather(flat, index)
        return last

    def _define_embedding(self):
        """Build the embedding matrix. If the user supplied a matrix, it
        is converted into a Tensor, else a random Tensor is built. This
        method sets `self.embedding` for use and returns None.
        """
        if type(self.embedding) == type(None):
            self.embedding = tf.Variable(
                tf.random_uniform(
                    [self.vocab_size, self.embed_dim], -1.0, 1.0),
                trainable=self.train_embedding)
        else:
            self.embedding = tf.Variable(
                initial_value=self.embedding,
                dtype=tf.float32,
                trainable=self.train_embedding)
            self.embed_dim = self.embedding.shape[1]

    def _convert_X(self, X):
        """Convert `X` to a list of list of indices into `self.vocab`,
        where all the lists have length `self.max_length`, which
        truncates the beginning of longer sequences and zero-pads the
        end of shorter sequences.

        Parameters
        ----------
        X : array-like
            The rows must be lists of objects in `self.vocab`.

        Returns
        -------
        np.array of int-type objects
            List of list of indices into `self.vocab`
        """
        new_X = np.zeros((len(X), self.max_length), dtype='int')
        ex_lengths = []
        index = dict(zip(self.vocab, range(len(self.vocab))))
        unk_index = index['$UNK']
        for i in range(new_X.shape[0]):
            ex_len = min([len(X[i]), self.max_length])
            ex_lengths.append(ex_len)
            vals = X[i][-self.max_length: ]
            vals = [index.get(w, unk_index) for w in vals]
            temp = np.zeros((self.max_length,), dtype='int')
            temp[0: len(vals)] = vals
            new_X[i] = temp
        return new_X, ex_lengths


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

    mod = TfRNNClassifier(
        vocab=vocab, max_iter=100, max_length=4)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, _ = zip(*test)
    print('\nPredictions:', mod.predict(X_test))


if __name__ == '__main__':
    simple_example()
