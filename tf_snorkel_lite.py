import numpy as np
import sys
import tensorflow as tf
from tf_model_base import TfModelBase


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2018"


class TfSnorkelGenerative(TfModelBase):
    """This is a TensorFlow implementation of the basic Snorkel model,
    in which we implicitly (and probably falsely) assume that the
    labeling functions are independent given the class label. The
    implementation is based in part on the blog post here:

    https://hazyresearch.github.io/snorkel/blog/dp_with_tf_blog_post.html

    which very helpfully describes how to compute the objective from
    the matrix of labeling function outputs for each example.

    Parameters
    ----------
    l2_penalty : float
        0.0 means no L2 regularization, and larger is stronger.
    l1_penalty : float
        0.0 means no L1 regularization, and larger is stronger.
    classes : tuple of str
        Names for the 0 and 1 classes, in that order.
    """
    def __init__(self, l2_penalty=0.0, l1_penalty=0.0,
            classes=('negative', 'positive'), **kwargs):
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        super().__init__(self, **kwargs)
        self.classes = classes

    def build_graph(self):
        self.inputs = tf.placeholder(
            tf.float32, [self.input_dim, self.input_dim])
        # TODO: I am unsure of this use of `tf.nn.sigmoid`. It is not
        # done in the blog post, but the blog post code does not
        # guarantee that these weights will stay in (0, 1), and hence
        # the log-odds calculation in `self.get_weights` can be undefined.
        self.W = tf.nn.sigmoid(tf.Variable(tf.ones([self.input_dim, 1])))
        y = self.W * tf.transpose(self.W)
        # Per the blog post, this zeros out the diagonal, which is
        # always 1:
        diag = tf.zeros(self.input_dim)
        mask = tf.ones((self.input_dim, self.input_dim))
        mask = tf.matrix_set_diag(mask, diag)
        self.y_aug = tf.multiply(y, mask)
        self.inputs_aug = tf.multiply(self.inputs, mask)

    def fit(self, X):
        # This is a nonce `y` vector to work around a restrictive design
        # in the superclass's fit method. Its values are not used.
        y = [None for _ in range(len(X))]
        # We need to see the full label matrix to get accurate estimates
        # of their relative alignment across the entire dataset:
        self.batch_size = X.shape[0]
        super().fit(X, y)

    def prepare_output_data(self, y):
        return y

    def get_cost_function(self):
        """Defines

        min_{W} ||Z - WW^{T}||^{2}_{F}

        where Z is computed by `_compute_overlap_matrix`. The variables
        `inputs_aug` and `y_aug` correspond to Z and WW^{T} with
        their diagonals zeroed out.

        """
        l2 = tf.nn.l2_loss(self.W) * self.l2_penalty
        l1 = tf.reduce_sum(tf.abs(self.W)) * self.l1_penalty
        loss = tf.reduce_sum(tf.square(self.inputs_aug - self.y_aug))
        return loss + l1 + l2

    def train_dict(self, X, y):
        Z = self._compute_overlap_matrix(X)
        return {self.inputs: Z}

    def test_dict(self, X):
        Z = self._compute_overlap_matrix(X)
        return {self.inputs: Z}

    def predict_proba(self, X):
        """Probabilistic predictions: the noisy labels that we assign
        to the examples in `X`. These are defined by Xw, where w
        is rescaled to be in the label space [0,1] and then converted
        to log-odds.

        Parameters
        ----------
        X : np.array
            Examples to make predictions about. These need to be
            created by the same labeling functions used for the
            training data.

        Returns
        -------
        np.array
            Vector of predictions, same length as `X.shape[0]`.
        """

        # This rescales the weights from [-1, 1] as the prediction
        # space into [0, 1]:
        w = self.get_weights()
        return self.sigmoid(X.dot(w)).ravel()

    def get_weights(self):
        w = (self.W.eval() + 1.0) / 2.0
        w = w.ravel()
        w = self.logit(w)
        return w

    @staticmethod
    def sigmoid(x):
        return np.exp(x) / (1.0 + np.exp(x))

    @staticmethod
    def logit(x):
        return np.log(x / (1.0 - x))

    def predict(self, X, threshold=0.5):
        """Returns the list of labels as determined by classifying
        the predicted probabilities using `threshold`. The labels are
        indices into `self.classes`.
        """
        proba = self.predict_proba(X)
        preds = [1 if p > threshold else 0 for p in proba]
        preds = [self.classes[x] for x in preds]
        return preds

    def _compute_overlap_matrix(self, X):
        """Computes a matrix giving, for each pair of labeling functions,
        the probability that they agree on the label they assign.
        This calculation includes a smoothing factor to handle cases
        where neither labeling function makes a prediction (and thus
        the normalizing value would be 0).

        Parameters
        ----------
        X : np.array, shape (m x n)
            The matrix of labeling function outputs per example.

        Returns
        -------
        np.array, shape (n x n)

        """
        X = np.array(X)
        epsilon = sys.float_info.epsilon
        X_abs = np.abs(X)
        Z = (X.T.dot(X) + epsilon) / (X_abs.T.dot(X_abs) + epsilon)
        return Z


class TfLogisticRegression(TfModelBase):
    """Basic Logistic Regression that can learn from probability
    values, rather than just labels.

    Parameters
    ----------
    l2_penalty : float
        0.0 means no L2 regularization, and larger is stronger.

    """
    def __init__(self, l2_penalty=0.0, **kwargs):
        self.l2_penalty = l2_penalty
        super().__init__(self, **kwargs)

    def build_graph(self):
        self.inputs = tf.placeholder(
            tf.float32, [None, self.input_dim])
        self.outputs = tf.placeholder(
            tf.float32, [None, 1])
        self.W = tf.Variable(tf.ones([self.input_dim, 1]))
        self.b = tf.Variable(tf.ones([1, 1]))
        self.model = tf.matmul(self.inputs, self.W) + self.b
        self.probs = tf.nn.sigmoid(self.model)

    def get_cost_function(self):
        l2 = tf.nn.l2_loss(self.W) * self.l2_penalty
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.outputs,
                logits=self.model))
        return loss + l2

    def predict_proba(self, X):
        proba = self.sess.run(self.probs, feed_dict=self.test_dict(X))
        return proba.ravel()

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return [1 if p > threshold else 0 for p in proba]

    def prepare_output_data(self, y):
        return np.array(y).reshape(-1, 1)

    def train_dict(self, X, y):
        return {self.inputs: X, self.outputs: y}

    def test_dict(self, X):
        return {self.inputs: X}


def simple_example_generative():
    T = ["gastroenteritis", "gaucher disease", "blue sclera",
        "cure nantais", "charolais", "devon blue"]

    def contains_biological_word(text):
        disease_words = {'disease', 'syndrome', 'cure'}
        return 1 if {w for w in disease_words if w in text} else 0

    def ends_in_itis(text):
        return 1 if text.endswith('itis') else 0

    def sounds_french(text):
        return -1 if text.endswith('ais') else 0

    def contains_color_word(text):
        colors = {'red', 'blue', 'purple'}
        return -1 if {w for w in colors if w in text} else 0

    def apply_labelers(T, labelers):
        return np.array([[l(t) for l in labelers] for t in T])

    labelers = [contains_biological_word, ends_in_itis,
                sounds_french, contains_color_word]

    L = apply_labelers(T, labelers)

    snorkel = TfSnorkelGenerative(
        max_iter=1000, eta=0.1, classes=('cheese', 'disease'))

    snorkel.fit(L)

    proba = snorkel.predict_proba(L)

    pred = snorkel.predict(L)

    print("\nTfSnorkelGenerative coefs: {}".format(snorkel.get_weights()))

    print("\nTfSnorkelGenerative predicted probs: {}".format(proba))

    print("\nTfSnorkelGenerative predicted labels: {}".format(pred))

    return pred


def simple_example_logistic_regression():
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score

    data = load_breast_cancer()

    X = data.data
    y = data.target

    def bounded_jitter(x):
        x = x + np.random.randn(len(x)) + 0.0001
        return np.maximum(np.minimum(x, 1), 0)

    # Show that we can learn from probalities, with 0/1 labels as a
    # special case:
    y_jittered = bounded_jitter(y)

    mod = TfLogisticRegression(l2_penalty=0.0, max_iter=2000)

    mod.fit(X, y_jittered)

    predictions = mod.predict(X)

    acc = accuracy_score(y, predictions)

    print("\nTfLogisticRegression dataset accuracy: {}".format(acc))

    return acc


if __name__ == '__main__':
    simple_example_generative()

    #simple_example_logistic_regression()
