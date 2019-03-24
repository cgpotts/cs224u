import torch
import torch.nn as nn

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


class TorchModelBase(object):
    def __init__(self,
            hidden_dim=50,
            hidden_activation=nn.Tanh(),
            batch_size=1028,
            max_iter=100,
            eta=0.01,
            optimizer=torch.optim.Adam,
            l2_strength=0,
            device=None):
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.eta = eta
        self.optimizer = optimizer
        self.l2_strength = l2_strength
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.params = [
            'hidden_dim',
            'hidden_activation',
            'batch_size',
            'max_iter',
            'eta',
            'optimizer']
        self.errors = []
        self.dev_predictions = {}

    def get_params(self, deep=True):
        return {p: getattr(self, p) for p in self.params}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def __repr__(self):
        param_str = ["{}={}".format(a, getattr(self, a)) for a in self.params]
        param_str = ",\n\t".join(param_str)
        return "{}(\n\t{})".format(self.__class__.__name__, param_str)
