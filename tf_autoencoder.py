import numpy as np
import pandas as pd
import tensorflow as tf
from tf_model_base import TfModelBase

__author__ = 'Chris Potts'


class TfAutoencoder(TfModelBase):
    """A simple autoencoder. The graph and parameters are identical
    to those of the `TfShallowNeuralClassifier`. The changes are that
    the outputs are identical to the inputs, and we use a squared-error
    loss function. Use `fit_transform` to return the matrix of
    hidden representations; `predict` will return the reconstruction
    of the input matrix.

    Parameters
    ----------
    hidden_dim : int
    max_iter : int
    eta : float
    tol : float
    """
    def __init__(self, eta=0.0001, **kwargs):
        super(TfAutoencoder, self).__init__(eta=eta, **kwargs)

    def fit(self, X):
        """Returns the matrix of hidden representations."""
        super(TfAutoencoder, self).fit(X, X)
        H = self.sess.run(self.hidden, feed_dict=self.test_dict(X))
        if isinstance(X, pd.DataFrame):
            H = pd.DataFrame(H, index=X.index)
        return H

    def prepare_output_data(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.output_dim = X.shape[1]
        return X

    def build_graph(self):
        """Same as `TfShallowNeuralClassifier` but with a single
        weight matrix:

        hidden = tanh(xW_xh + b_h)
        model = hW_xh^T + b_y

        where softmax is applied to model by the optimizer.
        """
        # Input and output placeholders
        self.inputs = tf.placeholder(
            tf.float32, shape=[None, self.input_dim])
        self.outputs = tf.placeholder(
            tf.float32, shape=[None, self.output_dim])

        # Parameters:
        self.W_xh = self.weight_init(
            self.input_dim, self.hidden_dim, name='W_xh')
        self.b_h = self.bias_init(
            self.hidden_dim, name='b_h')
        # This is a version without tied weights:
        # self.W_hy = self.weight_init(
        #    self.hidden_dim, self.output_dim, name='W_hy')
        self.W_hy = tf.transpose(self.W_xh)
        self.b_y = self.bias_init(
            self.output_dim, name='b_y')

        # The graph:
        self.hidden = self.hidden_activation(
            tf.matmul(self.inputs, self.W_xh) + self.b_h)
        self.model = tf.matmul(self.hidden, self.W_hy) + self.b_y

    def predict(self, X):
        return self.predict_proba(X)

    def get_cost_function(self, **kwargs):
        """Also expressible as

        tf.reduce_mean(
            tf.constant(0.5) * tf.square(self.outputs - self.model))
        """
        return tf.losses.mean_squared_error(
            labels=self.outputs,
            predictions=self.model,
            weights=0.5)

    def train_dict(self, X, y):
        return {self.inputs: X, self.outputs: y}

    def test_dict(self, X):
        return {self.inputs: X}


def simple_example():
    def randmatrix(m, n, sigma=0.1, mu=0):
        return sigma * np.random.randn(m, n) + mu

    rank = 20
    nrow = 1000
    ncol = 100

    X = randmatrix(nrow, rank).dot(randmatrix(rank, ncol))
    ae = TfAutoencoder(hidden_dim=rank, max_iter=100)
    ae.fit(X)
    X_pred = ae.predict(X)
    mse = (0.5*(X_pred - X)**2).mean()
    print("\nMSE between actual and reconstructed: {}".format(mse))


if __name__ == '__main__':
    simple_example()
