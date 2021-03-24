import numpy as np
from np_model_base import NNModelBase
import pandas as pd
from sklearn.metrics import r2_score

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


class Autoencoder(NNModelBase):
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)

    def prepare_output_data(self, y):
        return y

    def fit(self, X):
        self.input_dim = X.shape[1]
        self.output_dim = self.input_dim
        X_array = self.convert_input_to_array(X)
        super().fit(X_array, X_array)
        H = self.hidden_activation(X.dot(self.W_xh))
        H = self.convert_output(H, X)
        return H

    @staticmethod
    def get_error(predictions, labels):
        return (0.5 * (predictions - labels)**2).sum()

    def initialize_parameters(self):
        self.W_xh = self.weight_init(self.input_dim, self.hidden_dim)
        self.b_xh = self.bias_init(self.hidden_dim)
        self.W_hy = self.weight_init(self.hidden_dim, self.output_dim)
        self.b_hy = self.bias_init(self.output_dim)

    def update_parameters(self, gradients):
        d_W_hy, d_b_hy, d_W_xh, d_b_xh = gradients
        self.W_hy -= self.eta * d_W_hy
        self.b_hy -= self.eta * d_b_hy
        self.W_xh -= self.eta * d_W_xh
        self.b_xh -= self.eta * d_b_xh

    def forward_propagation(self, x):
        h = self.hidden_activation(x.dot(self.W_xh) + self.b_xh)
        y = h.dot(self.W_hy) + self.b_hy
        return h, y

    def backward_propagation(self, h, predictions, x, labels):
        y_err = predictions - labels
        d_b_hy = y_err
        h_err = y_err.dot(self.W_hy.T) * self.d_hidden_activation(h)
        d_W_hy = np.outer(h, y_err)
        d_W_xh = np.outer(x, h_err)
        d_b_xh = h_err
        return d_W_hy, d_b_hy, d_W_xh, d_b_xh

    def predict(self, X):
        h, y = self.forward_propagation(X)
        return y

    def score(self, X, y=None):
        y = X if y is None else y
        preds = self.predict(X)
        return r2_score(y, preds)

    @staticmethod
    def convert_input_to_array(X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X

    @staticmethod
    def convert_output(X_pred, X):
        if isinstance(X, pd.DataFrame):
            X_pred = pd.DataFrame(X_pred, index=X.index)
        return X_pred


def simple_example():
    import numpy as np
    import utils

    utils.fix_random_seeds()

    def randmatrix(m, n, sigma=0.1, mu=0):
        return sigma * np.random.randn(m, n) + mu

    rank = 20
    nrow = 1000
    ncol = 100

    X = randmatrix(nrow, rank).dot(randmatrix(rank, ncol))

    mod = Autoencoder(hidden_dim=rank, max_iter=200)

    print(mod)

    H = mod.fit(X)

    X_pred = mod.predict(X)

    mse = ((X_pred - X)**2).mean()

    print("\nMSE between actual and reconstructed: {0:0.09f}".format(mse))

    r2 = mod.score(X)

    print("R^2 score: {}".format(r2))

    print("Hidden representations")
    print(H)

    return r2


if __name__ == '__main__':
    simple_example()
