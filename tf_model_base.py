import numpy as np
import pandas as pd
import random
import sys
import tensorflow as tf

__author__ = 'Chris Potts'


class TfModelBase(object):
    """
    Subclasses need only define `build_graph`, `train_dict`, and
    `test_dict`. (They can redefine other methods as well.)

    Parameters
    ----------
    hidden_dim : int
    hidden_activation : tf.nn function
    max_iter : int
    eta : float
        Learning rate
    tol : float
        Stopping criterion for the loss.
    display_progress : int
        For value i, progress is printed every ith iteration.

    Attributes
    ----------
    errors : list
        Tracks loss from each iteration during training.
    """
    def __init__(self, hidden_dim=50, hidden_activation=tf.nn.tanh,
            batch_size=1028, max_iter=100, eta=0.01, tol=1e-4, display_progress=1):
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.eta = eta
        self.tol = tol
        self.display_progress = display_progress
        self.errors = []
        self.dev_predictions = []
        self.params = [
            'hidden_dim', 'hidden_activation', 'max_iter', 'eta']

    def build_graph(self):
        """Define the computation graph. This needs to define
        variables on which the cost function depends, so see
        `cost_function` below unless you are defining your own.

        """
        raise NotImplementedError

    def train_dict(self, X, y):
        """This method should feed `X` to the placeholder that defines
        the inputs and `y` to the placeholder that defines the output.
        For example:

        {self.inputs: X, self.outputs: y}

        This is used during training.

        """
        raise NotImplementedError

    def test_dict(self, X, y):
        """This method should feed `X` to the placeholder that defines
        the inputs. For example:

        {self.inputs: X}

        This is used during training.

        """
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        """Standard `fit` method.

        Parameters
        ----------
        X : np.array
        y : array-like
        kwargs : dict
            For passing other parameters, e.g., a test set that we
            want to monitor performance on.

        Returns
        -------
        self

        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Incremental performance:
        X_dev = kwargs.get('X_dev')
        if X_dev is not None:
            dev_iter = kwargs.get('test_iter', 10)

        # One-hot encoding of target `y`, and creation
        # of a class attribute.
        y = self.prepare_output_data(y)

        self.input_dim = len(X[0])

        # Start the session:
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

        # Build the computation graph. This method is instantiated by
        # individual subclasses. It defines the model.
        self.build_graph()

        # Optimizer set-up:
        self.cost = self.get_cost_function()
        self.optimizer = self.get_optimizer()

        # Initialize the session variables:
        self.sess.run(tf.global_variables_initializer())

        # Training, full dataset for each iteration:
        for i in range(1, self.max_iter+1):
            loss = 0
            for X_batch, y_batch in self.batch_iterator(X, y):
                _, batch_loss = self.sess.run(
                    [self.optimizer, self.cost],
                    feed_dict=self.train_dict(X_batch, y_batch))
                loss += batch_loss
            self.errors.append(loss)
            if X_dev is not None and i > 0 and i % dev_iter == 0:
                self.dev_predictions.append(self.predict(X_dev))
            if loss < self.tol:
                self._progressbar("stopping with loss < self.tol", i)
                break
            else:
                self._progressbar("loss: {}".format(loss), i)
        return self

    def batch_iterator(self, X, y):
        dataset = list(zip(X, y))
        random.shuffle(dataset)
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i: i+self.batch_size]
            X_batch, y_batch = zip(*batch)
            yield X_batch, y_batch

    def get_cost_function(self, **kwargs):
        """Uses `softmax_cross_entropy_with_logits` so the
        input should *not* have a softmax activation
        applied to it.
        """
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.model, labels=self.outputs))

    def get_optimizer(self):
        return tf.train.GradientDescentOptimizer(
            self.eta).minimize(self.cost)

    def predict_proba(self, X):
        """Return probabilistic predictions.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        np.array of predictions, dimension m x n, where m is the length
        of X and n is the number of classes

        """
        self.probs = tf.nn.softmax(self.model)
        return self.sess.run(
            self.probs, feed_dict=self.test_dict(X))

    def predict(self, X):
        """Return classifier predictions, as the class with the
        highest probability for each example.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        list

        """
        probs = self.predict_proba(X)
        return [self.classes[np.argmax(row)] for row in probs]

    def _onehot_encode(self, y):
        classmap = dict(zip(self.classes, range(self.output_dim)))
        y_ = np.zeros((len(y), self.output_dim))
        for i, cls in enumerate(y):
            y_[i][classmap[cls]] = 1.0
        return y_

    def _progressbar(self, msg, index):
        if self.display_progress and index % self.display_progress == 0:
            sys.stderr.write('\r')
            sys.stderr.write("Iteration {}: {}".format(index, msg))
            sys.stderr.flush()

    def weight_init(self, m, n, name):
        """
        Uses the Xavier Glorot method for initializing
        weights. This is built in to TensorFlow as
        `tf.contrib.layers.xavier_initializer`, but it's
        nice to see all the details.
        """
        x = np.sqrt(6.0/(m+n))
        with tf.name_scope(name) as scope:
            return tf.Variable(
                tf.random_uniform(
                    [m, n], minval=-x, maxval=x), name=name)

    def bias_init(self, dim, name, constant=0.0):
        """Default all 0s bias, but `constant` can be
        used to specify other values."""
        with tf.name_scope(name) as scope:
            return tf.Variable(
                tf.constant(constant, shape=[dim]), name=name)

    def prepare_output_data(self, y):
        """Format `y` so that Tensorflow can deal with it, by turning
        it into a vector of one-hot encoded vectors.

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

    def get_params(self, deep=True):
        """Gets the hyperparameters for the model, as given by the
        `self.params` attribute. This is called `get_params` for
        compatibility with sklearn.

        Returns
        -------
        dict
            Map from attribute names to their values.

        """
        return {p: getattr(self, p) for p in self.params}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self
