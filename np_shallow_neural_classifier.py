import numpy as np
from np_model_base import NNModelBase
from utils import randvec, randmatrix, softmax, progress_bar

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


class ShallowNeuralClassifier(NNModelBase):
    """Fit a model

    h = f(xW1 + b1)
    y = softmax(hW2 + b2)

    with a cross entropy loss.
    """
    def __init__(self, **kwargs):
        """All the parameters are set as attributes.

        Parameters
        ----------
        hidden_dim : int (default: 40)
            Dimensionality of the hidden layer.
        hidden_activation : vectorized activation function
            The non-linear activation function used by the
            network for the hidden and output layers.
        d_hidden_activation : vectorized activation function derivative.
            The derivative of `afunc`. It does not ensure that this
            matches `afunc`, and craziness will result from mismatches!
        max_iter : int default: 100)
            Maximum number of training epochs.
        eta : float (default: 0.05)
            Learning rate.
        tol : float (default: 1.5e-8)
            Training terminates if the error reaches this point (or
            `maxiter` is met).
        display_progress : bool (default: True)
           Whether to use the simple over-writing `progress_bar`
           to show progress.

        """
        super(ShallowNeuralClassifier, self).__init__(**kwargs)
        self.params += ['hidden_activation', 'd_hidden_activation']

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        super().fit(X, y)

    def initialize_parameters(self):
        self.W_xh = self.weight_init(self.input_dim, self.hidden_dim)
        self.b_xh = np.zeros(self.hidden_dim)
        self.W_hy = self.weight_init(self.hidden_dim, self.output_dim)
        self.b_hy = np.zeros(self.output_dim)

    def update_parameters(self, gradients):
        d_W_hy, d_b_hy, d_W_xh, d_b_xh = gradients
        self.W_hy -= self.eta * d_W_hy
        self.b_hy -= self.eta * d_b_hy
        self.W_xh -= self.eta * d_W_xh
        self.b_xh -= self.eta * d_b_xh

    def forward_propagation(self, x):
        h = self.hidden_activation(x.dot(self.W_xh) + self.b_xh)
        y = softmax(h.dot(self.W_hy) + self.b_hy)
        return h, y

    def backward_propagation(self, h, predictions, x, labels):
        y_err = predictions.copy()
        y_err[np.argmax(labels)] -= 1
        d_b_hy = y_err
        h_err = y_err.dot(self.W_hy.T) * self.d_hidden_activation(h)
        d_W_hy = np.outer(h, y_err)
        d_W_xh = np.outer(x, h_err)
        d_b_xh = h_err
        return d_W_hy, d_b_hy, d_W_xh, d_b_xh


def simple_example():
    """Assess on the digits dataset."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    mod = ShallowNeuralClassifier(max_iter=100)

    print(mod)

    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)

    print("\nClassification report:")

    print(classification_report(y_test, predictions))

    return accuracy_score(y_test, predictions)


if __name__ == '__main__':
    simple_example()
