import tensorflow as tf

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


class TfModelBase(object):
    def __init__(self,
            hidden_dim=50,
            hidden_activation=tf.nn.tanh,
            batch_size=1028,
            max_iter=100,
            eta=0.01,
            model_dir=None):
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.eta = eta
        self.model_dir = model_dir
        self.params = [
            'hidden_dim',
            'hidden_activation',
            'batch_size',
            'max_iter',
            'eta']

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

    def __repr__(self):
        param_str = ["{}={}".format(a, getattr(self, a)) for a in self.params]
        param_str = ",\n\t".join(param_str)
        return "{}(\n\t{})".format(self.__class__.__name__, param_str)

    def __str__(self):
        return repr(self)
