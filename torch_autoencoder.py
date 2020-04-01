import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
from utils import progress_bar

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


class TorchAutoencoder(TorchModelBase):
    """A simple autoencoder. The graph and parameters are identical
    to those of the `TorchShallowNeuralClassifier`. The changes are that
    the outputs are identical to the inputs, and we use a squared-error
    loss function.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the hidden layer.
    hidden_activation : vectorized activation function
        The non-linear activation function used by the network for the
        hidden layer. Default `nn.Tanh()`.
    max_iter : int
        Maximum number of training epochs.
    eta : float
        Learning rate.
    optimizer : PyTorch optimizer
        Default is `torch.optim.Adam`.
    l2_strength : float
        L2 regularization strength. Default 0 is no regularization.
    device : 'cpu' or 'cuda'
        The default is to use 'cuda' iff available
    warm_start : bool
        If True, calling `fit` will resume training with previously
        defined trainable parameters. If False, calling `fit` will
        reinitialize all trainable parameters. Default: False.

    """
    def __init__(self, **kwargs):
        super(TorchAutoencoder, self).__init__(**kwargs)

    def define_graph(self):
        return nn.Sequential(
            nn.Linear(self.input_dim_, self.hidden_dim),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.output_dim_))

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
        # Data prep:
        self.input_dim_ = X.shape[1]
        self.output_dim_ = X.shape[1]
        # Dataset:
        X_tensor = self.convert_input_to_tensor(X)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=True)
        # Graph
        if not self.warm_start or not hasattr(self, "model"):
            self.model = self.define_graph()
            self.opt = self.optimizer(
                self.model.parameters(),
                lr=self.eta,
                weight_decay=self.l2_strength)
        self.model.to(self.device)
        self.model.train()
        # Optimization:
        loss = nn.MSELoss()
        # Train:
        for iteration in range(1, self.max_iter+1):
            epoch_error = 0.0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                batch_preds = self.model(X_batch)
                err = loss(batch_preds, y_batch)
                epoch_error += err.item()
                self.opt.zero_grad()
                err.backward()
                self.opt.step()
            self.errors.append(epoch_error)
            progress_bar(
                "Finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, err))
        # Hidden representations:
        with torch.no_grad():
            self.model.to('cpu')
            H = self.model[1](self.model[0](X_tensor))
            return self.convert_output(H, X)

    def predict(self, X):
        """Returns the reconstructed matrix.

        Parameters
        ----------
        X : np.array or pd.DataFrame

        Returns
        -------
        np.array or pd.DataFrame (depending on the nature of the input)
        This will have the same shape as `X`.

        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = self.convert_input_to_tensor(X)
            self.model.to('cpu')
            X_pred = self.model(X_tensor)
            return self.convert_output(X_pred, X)

    def convert_input_to_tensor(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = torch.tensor(X, dtype=torch.float)
        return X

    @staticmethod
    def convert_output(X_pred, X):
        X_pred = X_pred.cpu().numpy()
        if isinstance(X, pd.DataFrame):
            X_pred = pd.DataFrame(X_pred, index=X.index)
        return X_pred


def simple_example():
    import numpy as np

    np.random.seed(seed=42)

    def randmatrix(m, n, sigma=0.1, mu=0):
        return sigma * np.random.randn(m, n) + mu

    rank = 20
    nrow = 1000
    ncol = 100

    X = randmatrix(nrow, rank).dot(randmatrix(rank, ncol))
    ae = TorchAutoencoder(hidden_dim=rank, max_iter=200)
    H = ae.fit(X)
    X_pred = ae.predict(X)
    mse = (0.5*(X_pred - X)**2).mean()
    print("\nMSE between actual and reconstructed: {0:0.06f}".format(mse))
    print("Hidden representations")
    print(H)
    return mse


if __name__ == '__main__':
   simple_example()
