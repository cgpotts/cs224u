import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


class TorchAutoencoder(TorchModelBase):
    def __init__(self,
            hidden_dim=50,
            hidden_activation=nn.Tanh(),
            **base_kwargs):
        """
        A simple autoencoder:

        h = f(xW_xh + b_h)
        y = f(hW_hy + b_y)

        where W_xh.shape == W_hy.T.shape and thus `y` has the same
        dimensionality as `x`.

        The graph and parameters are identical to those of the
        `TorchShallowNeuralClassifier`. The changes are that the
        outputs are identical to the inputs, and we use a squared-error
        loss function.

        Parameters
        ----------
        hidden_dim : int
            Dimensionality of the hidden layer.

        hidden_activation : nn.Module
            The non-activation function used by the network for the
            hidden layer. Default `nn.Tanh()`.

        Attributes
        ----------
        loss: nn.MSELoss(reduction="mean")

        self.params: list
            Extends TorchModelBase.params with ['hidden_dim',
            'hidden_activation'] to support tuning of these
            values using `sklearn.model_selection` tools.

        """
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        super().__init__(**base_kwargs)
        self.loss = nn.MSELoss(reduction="mean")
        self.params += ['hidden_dim', 'hidden_activation']

    def build_dataset(self, X, y=None):
        """
        Datasets for the model. For internal reasons, when training,
        we create datasets that yield identical copies of `X` as the
        inputs and outputs.

        Parameters
        ----------
        X : iterable of length `n_examples`
           Each element must have the same length.

        y: None or iterable of length `n_examples`
           Each element must have the same length.

        Attributes
        ----------
        input_dim: int
            Set by `X.shape[1]` once it is converted to `np.array`.

        output_dim: int
            Identical to `self.input_dim`.

        Returns
        -------
        torch.utils.data.TensorDataset` Where `y=None`, the dataset will
        yield single tensors `X`. Where `y` is specified, it will yield
        `(X, y)` pairs. for this model, `y==X`.

        """
        # Data prep:
        X = np.array(X)
        self.input_dim = X.shape[1]
        self.output_dim = X.shape[1]
        # Dataset:
        X = self.convert_input_to_tensor(X)
        if y is None:
            dataset = torch.utils.data.TensorDataset(X)
        else:
            y = self.convert_input_to_tensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)
        return dataset

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
            nn.Linear(self.hidden_dim, self.output_dim))

    def fit(self, X):
        """Returns the matrix of hidden representations.

        Parameters
        ----------
        X : np.array or pd.DataFrame

        Returns
        -------
        np.array or pd.DataFrame (depending on the nature of the input)
        This will have shape `(len(X), self.hidden_dim)`.

        """
        super().fit(X, X)
        # Hidden representations:
        with torch.no_grad():
            dataset = self.build_dataset(X)
            hidden_dataloader = self._build_dataloader(dataset, shuffle=False)
            H = []
            for X_input in hidden_dataloader:
                X_input = [x.to(self.device) for x in X_input]
                batch = self.model[1](self.model[0](*X_input))
                batch = batch.to("cpu")
                H.append(batch)
            H = torch.cat(H)
            return self.convert_output(H, X)

    def score(self, X, y=None, device=None):
        """
        Score the model based on the R^2 score between the input and the
        model's reconstruction of those inputs. It might seem more natural
        to use the mean-squared error for scoring, but it's helpful for
        cross-validators if the score function returns values for which
        positive is better. I assume this is why models like
        `sklearn.linear_model.LinearRegression` use `r2_score` as well.

        Parameters
        ----------
        X : np.array, shape `(n_examples, n_features)`

        y : None or iterable, shape `(n_examples, n_features)`
            If `None`, then `X` plays this role. We expect that, wherever
            `y` is used, it is a copy of `X`. Both interfaces are supported
            to facilitate interactions with external tools.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        float

        """
        y = X if y is None else y
        preds = self.predict(X, device=device)
        return r2_score(y, preds)

    def predict(self, X, device=None):
        """
        Returns the reconstructed matrix.

        Parameters
        ----------
        X : np.array or pd.DataFrame

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        np.array or pd.DataFrame (depending on the nature of the input)
        This will have the same shape as `X`.

        """
        X_pred = self._predict(X, device=device)
        return self.convert_output(X_pred, X)

    def convert_input_to_tensor(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return torch.FloatTensor(X)

    @staticmethod
    def convert_output(X_pred, X):
        X_pred = X_pred.cpu().numpy()
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
    nrow = 2000
    ncol = 100

    X = randmatrix(nrow, rank).dot(randmatrix(rank, ncol))

    mod = TorchAutoencoder()

    print(mod)

    H = mod.fit(X)

    X_pred = mod.predict(X)

    mse = ((X_pred - X)**2).mean()

    print("\nMSE between actual and reconstructed: {}".format(mse))

    r2 = mod.score(X)

    print("R^2 score: {}".format(r2))

    print("Hidden representations")
    print(H)

    return r2


if __name__ == '__main__':
    simple_example()
