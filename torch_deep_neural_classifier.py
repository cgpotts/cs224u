import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import utils

__author__ = "Atticus Geiger"
__version__ = "CS224u, Stanford, Spring 2022"


class ActivationLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_activation):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, device=device)
        self.activation = hidden_activation

    def forward(self, x):
        return self.activation(self.linear(x))


class TorchDeepNeuralClassifier(TorchShallowNeuralClassifier):
    def __init__(self,
            num_layers=1,
            **base_kwargs):
        """
        A dense, feed-forward network with the number of hidden layers
        set by `num_layers`.

        Parameters
        ----------
        num_layers : int
            Number of hidden layers in the network.

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
        self.num_layers = num_layers
        super().__init__(**base_kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.params += ['num_layers']

    def build_graph(self):
        """
        Define the model's computation graph.

        Returns
        -------
        nn.Module

        """
        # Input to hidden:
        self.layers = [
            ActivationLayer(
                self.input_dim, self.hidden_dim, self.device, self.hidden_activation)]
        # Hidden to hidden:
        for _ in range(self.num_layers-1):
            self.layers += [
                ActivationLayer(
                    self.hidden_dim, self.hidden_dim, self.device, self.hidden_activation)]
        # Hidden to output:
        self.layers.append(
            nn.Linear(self.hidden_dim, self.n_classes_, device=self.device))
        return nn.Sequential(*self.layers)



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

    mod = TorchDeepNeuralClassifier(num_layers=2)

    print(mod)

    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)

    print("\nClassification report:")

    print(classification_report(y_test, preds))

    return accuracy_score(y_test, preds)


if __name__ == '__main__':
    simple_example()
