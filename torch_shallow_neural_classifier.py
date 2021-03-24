import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


class TorchShallowNeuralClassifier(TorchModelBase):
    def __init__(self,
            hidden_dim=50,
            hidden_activation=nn.Tanh(),
            **base_kwargs):
        """
        A model

        h = f(xW_xh + b_h)
        y = softmax(hW_hy + b_y)

        with a cross-entropy loss and f determined by `hidden_activation`.

        Parameters
        ----------
        hidden_dim : int
            Dimensionality of the hidden layer.

        hidden_activation : nn.Module
            The non-activation function used by the network for the
            hidden layer.

        **base_kwargs
            For details, see `torch_model_base.py`.

        Attributes
        ----------
        loss: nn.CrossEntropyLoss(reduction="mean")

        self.params: list
            Extends TorchModelBase.params with names for all of the
            arguments for this class to support tuning of these values
            using `sklearn.model_selection` tools.

        """
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        super().__init__(**base_kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.params += ['hidden_dim', 'hidden_activation']

    def build_graph(self):
        """
        Define the model's computation graph.

        Returns
        -------
        nn.Module

        """
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.n_classes_))

    def build_dataset(self, X, y=None):
        """
        Define datasets for the model.

        Parameters
        ----------
        X : iterable of length `n_examples`
           Each element must have the same length.

        y: None or iterable of length `n_examples`

        Attributes
        ----------
        input_dim : int
            Set based on `X.shape[1]` after `X` has been converted to
            `np.array`.

        Returns
        -------
        torch.utils.data.TensorDataset` Where `y=None`, the dataset will
        yield single tensors `X`. Where `y` is specified, it will yield
        `(X, y)` pairs.

        """
        X = np.array(X)
        self.input_dim = X.shape[1]
        X = torch.FloatTensor(X)
        if y is None:
            dataset = torch.utils.data.TensorDataset(X)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)
        return dataset

    def score(self, X, y, device=None):
        """
        Uses macro-F1 as the score function. Note: this departs from
        `sklearn`, where classifiers use accuracy as their scoring
        function. Using macro-F1 is more consistent with our course.

        This function can be used to evaluate models, but its primary
        use is in cross-validation and hyperparameter tuning.

        Parameters
        ----------
        X: np.array, shape `(n_examples, n_features)`

        y: iterable, shape `len(n_examples)`
            These can be the raw labels. They will converted internally
            as needed. See `build_dataset`.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        float

        """
        preds = self.predict(X, device=device)
        return utils.safe_macro_f1(y, preds)

    def predict_proba(self, X, device=None):
        """
        Predicted probabilities for the examples in `X`.

        Parameters
        ----------
        X : np.array, shape `(n_examples, n_features)`

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        np.array, shape `(len(X), self.n_classes_)`
            Each row of this matrix will sum to 1.0.

        """
        preds = self._predict(X, device=device)
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        return probs

    def predict(self, X, device=None):
        """
        Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Parameters
        ----------
        X : np.array, shape `(n_examples, n_features)`

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        list, length len(X)

        """
        probs = self.predict_proba(X, device=device)
        return [self.classes_[i] for i in probs.argmax(axis=1)]


def simple_example():
    """Assess on the digits dataset."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    utils.fix_random_seeds()

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    mod = TorchShallowNeuralClassifier()

    print(mod)

    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)

    print("\nClassification report:")

    print(classification_report(y_test, preds))

    return accuracy_score(y_test, preds)


if __name__ == '__main__':
    simple_example()
