import pickle
import torch
import torch.nn as nn

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


class TorchModelBase(object):
    def __init__(self,
            hidden_dim=50,
            hidden_activation=nn.Tanh(),
            batch_size=1028,
            max_iter=100,
            eta=0.01,
            optimizer=torch.optim.Adam,
            l2_strength=0,
            warm_start=False,
            device=None):
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.eta = eta
        self.optimizer = optimizer
        self.l2_strength = l2_strength
        self.warm_start = warm_start
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.params = [
            'hidden_dim',
            'hidden_activation',
            'batch_size',
            'max_iter',
            'eta',
            'optimizer',
            'l2_strength']
        self.errors = []
        self.dev_predictions = {}

    def get_params(self, deep=True):
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

    def to_pickle(self, output_filename):
        """Serialize the entire class instance. Importantly, this
        is different from using the standard `torch.save` method:

        torch.save(self.model.state_dict(), output_filename)

        The above stores only the underlying model parameters. In
        contrast, the current method ensures that all of the model
        parameters are on the CPU and then stores the full instance.
        This is necessary to ensure that we retain all the information
        needed to read new examples and make predictions.

        Parameters
        ----------
        output_filename : str
            Full path for the output file.

        """
        self.model = self.model.cpu()
        with open(output_filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(src_filename):
        """Load an entire class instance onto the CPU. This also sets
        `self.warm_start = True` so that the loaded parameters are used
        if `fit` is called.

        Importantly, this is different from recommended PyTorch method:

        self.model.load_state_dict(torch.load(src_filename))

        We cannot reliably do this with new instances, because we need
        to see new examples in order to set some of the model
        dimensionalities and obtain information about what the class
        labels are. Thus, the current method loads an entire serialized
        class as created by `to_pickle`.

        The training and prediction code move the model parameters to
        `self.device`.

        Parameters
        ----------
        src_filename : str
            Full path to the serialized model file.

        """
        with open(src_filename, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        param_str = ["{}={}".format(a, getattr(self, a)) for a in self.params]
        param_str = ",\n\t".join(param_str)
        return "{}(\n\t{})".format(self.__class__.__name__, param_str)
