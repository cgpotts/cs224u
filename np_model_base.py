import numpy as np
import random
from utils import randvec, randmatrix, progress_bar, d_tanh

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


class NNModelBase(object):
    def __init__(self,
            hidden_dim=50,
            hidden_activation=np.tanh,
            d_hidden_activation=d_tanh,
            eta=0.01,
            max_iter=100,
            tol=1e-6,
            display_progress=True):
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.d_hidden_activation = d_hidden_activation
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.display_progress = display_progress
        self.params = ['hidden_dim', 'eta', 'max_iter']

    def initialize_parameters(self):
        raise NotImplementedError

    def update_parameters(self, gradients):
        raise NotImplementedError

    def forward_propagation(self):
        raise NotImplementedError

    def backward_propagation(self):
        raise NotImplementedError

    def fit(self, X, y):
        """Train the network.

        Parameters
        ----------
        X : list of lists
           Each element should be a list of elements in `self.vocab`.
        y : list
           The one-hot label vector.

        Returns
        ----------
        self

        """
        y = self.prepare_output_data(y)
        self.initialize_parameters()
        # Unified view for training
        training_data = list(zip(X, y))
        # SGD:
        iteration = 0
        for iteration in range(1, self.max_iter+1):
            error = 0.0
            random.shuffle(training_data)
            for ex, labels in training_data:
                hidden_states, predictions = self.forward_propagation(ex)
                error += self.get_error(predictions, labels)
                # Back-prop:
                gradients = self.backward_propagation(
                    hidden_states, predictions, ex, labels)
                self.update_parameters(gradients)
            error /= len(training_data)
            if error <= self.tol:
                if self.display_progress:
                    progress_bar(
                        "Converged on iteration {} with error {}".format(
                            iteration, error))
                break
            else:
                if self.display_progress:
                    progress_bar(
                        "Finished epoch {} of {}; error is {}".format
                        (iteration, self.max_iter, error))
        return self

    @staticmethod
    def get_error(predictions, labels):
        """Cross-entropy error: -log(prediction-for-correct-label).

        Parameters
        ----------
        predictions : np.array
            Predicted probabilities for each class
        labels : np.array
            One-hot encoded vector.

        Returns
        -------
        float

        """
        return -np.log(predictions[np.argmax(labels)])

    @staticmethod
    def _define_embedding_matrix(vocab_size, embed_dim):
        return np.random.uniform(
            low=-1.0, high=1.0, size=(vocab_size, embed_dim))

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
        hidden_states, predictions = self.forward_propagation(seq)
        return predictions

    def predict_proba(self, X):
        """Softmax predictions for a list of examples.

        Parameters
        ----------
        X : list of lists
            List of examples.

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
            List of examples.

        Returns
        -------
        list

        """
        return [self.predict_one(ex) for ex in X]

    def predict_one(self, x):
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
        probs = self.predict_one_proba(x)
        return self.classes[np.argmax(probs)]

    def get_word_rep(self, w):
        """For getting the input representation of word `w` from
        `self.embedding`.

        Parameters
        ----------
        w : str

        Returns
        -------
        np.array, dimension `self.embed_dim`

        """
        if w in self.vocab_lookup:
            word_index = self.vocab_lookup[w]
        else:
            word_index = self.vocab_lookup['$UNK']
        return self.embedding[word_index]

    @staticmethod
    def weight_init(m, n):
        """Uses the Xavier Glorot method for initializing the weights
        of an `m` by `n` matrix.

        Parameters
        ----------
        m : int
            Row dimension
        n : int
            Column dimension

        Returns
        -------
        np.array, shape `(m, n)`

        """
        #x = np.sqrt(6.0/(m+n))
        x = np.sqrt(1.0 / n)
        return randmatrix(m, n, lower=-x, upper=x)

    @staticmethod
    def bias_init(n):
        """Uses the current PyTorch default `nn.Linear`."""
        x = np.sqrt(1.0 / n)
        return randvec(n, lower=-x, upper=x)

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
        """Maps a single label `y` to a one-hot encoding with 1.0 in
        the position of y and 0.0 for all other classes.

        Parameters
        ----------
        y : object
            Typically a str, int, or bool

        Returns
        -------
        np.array, dimension `len(self.classes)`

        """
        classmap = dict(zip(self.classes, range(self.output_dim)))
        y_ = np.zeros((len(y), self.output_dim))
        for i, cls in enumerate(y):
            y_[i][classmap[cls]] = 1.0
        return y_

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
        params = self.params.copy()
        # Obligatorily add `vocab` so that sklearn passes it in when
        # creating new model instances during cross-validation:
        if hasattr(self, 'vocab'):
            params += ['vocab']
        return {p: getattr(self, p) for p in params}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def __repr__(self):
        param_str = ["{}={}".format(a, getattr(self, a)) for a in self.params]
        param_str = ",\n\t".join(param_str)
        return "{}(\n\t{})".format(self.__class__.__name__, param_str)
