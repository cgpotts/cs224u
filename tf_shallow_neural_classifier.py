import tensorflow as tf
from tf_model_base import TfModelBase

__author__ = 'Chris Potts'


class TfShallowNeuralClassifier(TfModelBase):
    """Defines a one-layer network on top of `TfNeuralBase`.
    This just means defining the graph and associated
    `train_dict` and `test_dict` methods.

    Parameters
    ----------
    hidden_dim : int
    max_iter : int
    eta : float
    tol : float

    """
    def __init__(self, **kwargs):
        super(TfShallowNeuralClassifier, self).__init__(**kwargs)

    def build_graph(self):
        """Builds a graph

        hidden = relu(xW_xh + b_h)
        model = softmax(hW_hy + b_y)
        """
        self.define_parameters()

        # The graph:
        self.hidden = self.hidden_activation(
            tf.matmul(self.inputs, self.W_xh) + self.b_h)
        self.model = tf.matmul(self.hidden, self.W_hy) + self.b_y

    def define_parameters(self):
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
        self.W_hy = self.weight_init(
            self.hidden_dim, self.output_dim, name='W_hy')
        self.b_y = self.bias_init(
            self.output_dim, name='b_y')

    def train_dict(self, X, y):
        return {self.inputs: X, self.outputs: y}

    def test_dict(self, X):
        return {self.inputs: X}


def simple_example():
    """Assess on the digits dataset and informally compare
    against LogisticRegression.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.linear_model import LogisticRegression

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    models = [
        TfShallowNeuralClassifier(max_iter=2000),
        LogisticRegression()
    ]

    for mod in models:
        print(mod)
        clf = mod.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print(classification_report(y_test, predictions))


if __name__ == '__main__':
   simple_example()
