import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"

class TorchShallowNeuralClassifierIIT(TorchShallowNeuralClassifier):
    def __init__(self,id_to_coords, **base_kwargs):
        super().__init__(**base_kwargs)
        loss_function= nn.CrossEntropyLoss(reduction="mean")
        self.loss = lambda preds, labels: loss_function(preds[0],labels[:,0]) + loss_function(preds[1],labels[:,1])
        self.id_to_coords = id_to_coords
        self.shuffle_train = False

    def build_graph(self):
        model = super().build_graph()
        model.no_IIT_forward = model.forward
        model.forward = self.IIT_forward
        return model

    def batched_indices(self, max_len):
        batch_indices = [ x for x in range((max_len // self.batch_size))]
        output = []
        while len(batch_indices) != 0:
            batch_index = random.sample(batch_indices, 1)[0]
            batch_indices.remove(batch_index)
            output.append([batch_index*self.batch_size + x for x in range(self.batch_size)])
        return output

    def build_dataset(self, base, source, base_y, IIT_y, coord_ids):
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
        X = np.array(base)
        X2 = np.array(source)
        self.input_dim = X.shape[1]
        X = torch.FloatTensor(X)
        X2 = torch.FloatTensor(X2)
        coord_ids = torch.FloatTensor(np.array(coord_ids))

        IIT_y = np.array(IIT_y)
        self.classes_ = sorted(set(IIT_y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        IIT_y = [class2index[int(label)] for label in IIT_y]
        IIT_y = torch.tensor(IIT_y)

        base_y = np.array(base_y)
        self.classes_ = sorted(set(base_y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        base_y = [class2index[label] for label in base_y]
        base_y = torch.tensor(base_y)

        bigX = torch.stack((X,X2, coord_ids.unsqueeze(1).expand(-1, X.shape[1])), dim=1)
        bigy = torch.stack((IIT_y, base_y), dim=1)
        dataset = torch.utils.data.TensorDataset(bigX,bigy)
        return dataset

    def prep_input(self, base, source, coord_ids):
        bigX = torch.stack((base,source, coord_ids.unsqueeze(1).expand(-1, base.shape[1])), dim=1)
        return bigX

    def IIT_forward(self, X):
        base, source, coord_ids = X[:,0,:].squeeze(1), X[:,1,:].squeeze(1),X[:,2,:].squeeze(1)
        get = self.id_to_coords[int(coord_ids.flatten()[0])]
        base = base.type(torch.FloatTensor).to(self.device)
        source = source.type(torch.FloatTensor).to(self.device)
        self.activation = dict()
        handlers = self._get_set(get,None)
        source_logits = self.model.no_IIT_forward(source)
        for handler in handlers:
            handler.remove()

        base_logits = self.model.no_IIT_forward(base)
        set = {k:get[k] for k in get}
        set["intervention"] = self.activation[f'{get["layer"]}-{get["start"]}-{get["end"]}']
        handlers = self._get_set(get, set)
        counterfactual_logits = self.model.no_IIT_forward(base)
        for handler in handlers:
            handler.remove()

        return counterfactual_logits, base_logits

if __name__ == '__main__':
    simple_example()
