import numpy as np
import random

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


class BasicSGDClassifier(object):
    """Basic implementation hinge-loss stochastic sub-gradient descent
    optimization, intended to illustrate the basic concepts of classifier
    optimization in code."""
    def __init__(self, max_iter=10, eta=0.1):
        """
        Parameters
        ----------
        max_iter : int (default: 10)
            Number of training epochs (full runs through shuffled data).
        eta : float (default: 0.1)
            Learning rate parameter.

        """
        self.max_iter = max_iter
        self.eta = eta
        self.params = ['max_iter', 'eta']

    def fit(self, feat_matrix, labels):
        """Core optimization function.

        Parameters
        ----------
        feat_matrix : 2d matrix (np.array or any scipy.sparse type)
            The design matrix, one row per example. Hence, the row
            dimensionality is the example count and the column
            dimensionality is number of features.

        labels : list
            The labels for each example, hence assumed to have the
            same length as, and be aligned with, `feat_matrix`.

        For attributes, we follow the `sklearn` style of using a
        final `_` for attributes that are created by `fit` methods:

        Attributes
        ----------
        self.classes_ : list
            The set of class labels in sorted order.

        self.n_classes_ : int
            Length of `self.classes_`

        self.coef_ : np.array of dimension (class count, feature count)
            These are the weights, named as in `sklearn`. They are
            organized so that each row represents the feature weights
            for a given class, as is typical in `sklearn`.

        """
        # We'll deal with the labels via their indices into self.classes_:
        self.classes_ = sorted(set(labels))
        self.n_classes_ = len(self.classes_)
        # Useful dimensions to store:
        examplecount, featcount = feat_matrix.shape
        # The weight matrix -- classes by row:
        self.coef_ = np.zeros((self.n_classes_, featcount))
        # Indices for shuffling the data at the start of each epoch:
        indices = list(range(examplecount))
        for _ in range(self.max_iter):
            random.shuffle(indices)
            for i in indices:
                # Training instance as a feature rep and a label index:
                rep = feat_matrix[i]
                label_index = self.classes_.index(labels[i])
                # Costs are 1.0 except for the true label:
                costs = np.ones(self.n_classes_)
                costs[label_index] = 0.0
                # Make a prediction:
                predicted_index = self.predict_one(rep, costs=costs)
                # Weight update if it's an incorrect prediction:
                if predicted_index != label_index:
                    self.coef_[label_index] += self.eta * rep

    def predict_one(self, rep, costs=0.0):
        """The core classification function. The code just needs to
        figure out which class is highest scoring and make a random
        choice from that set (in case of ties).

        Parameters
        ----------
        rep : np.array of dimension featcount or
              `scipy.sparse` matrix of dimension (1 x `featcount`)

        costs : float or np.array of dimension self.classcount
            Where this is 0.0, we're doing prediction. Where it
            is an array, we expect a 0.0 at the coordinate
            corresponding to the true label and a 1.0 in all
            other positions.

        Returns
        -------
        int
            The index of the correct class. This is for the
            sake of the `fit` method. `predict` returns the class
            names themselves.

        """
        scores = rep.dot(self.coef_.T) + costs
        # Manage the difference between scipy and numpy 1d matrices:
        scores = scores.reshape(self.n_classes_)
        # Set of highest scoring label indices (in case of ties):
        candidates = np.argwhere(scores==np.max(scores)).flatten()
        return random.choice(candidates)

    def predict(self, reps):
        """Batch prediction function for experiments.

        Parameters
        ----------
        reps : list or feature matrix
           A featurized set of examples to make predictions about.

        Returns
        -------
        list of str
            A list of class names -- the predictions. Unlike `predict_one`,
            it returns the class name rather than its index.

        """
        return [self.classes_[self.predict_one(rep)] for rep in reps]

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


def simple_example():
    """Assess on the digits dataset and informally compare
    against LogisticRegression.
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score


    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    mod = BasicSGDClassifier(max_iter=500)

    print(mod)

    mod.fit(X_train, y_train)

    predictions = mod.predict(X_test)

    print(classification_report(y_test, predictions))

    return accuracy_score(y_test, predictions)


if __name__ == '__main__':
    simple_example()
