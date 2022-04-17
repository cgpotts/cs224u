import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch_deep_neural_classifier import TorchDeepNeuralClassifier
import utils
from iit import IITModel

__author__ = "Atticus Geiger"
__version__ = "CS224u, Stanford, Spring 2022"


class CrossEntropyLossIIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, preds, labels):
        return self.loss(preds[0], labels[: , 0]) + self.loss(preds[1], labels[:,1])


class TorchDeepNeuralClassifierIIT(TorchDeepNeuralClassifier):
    def __init__(self, id_to_coords=None, **base_kwargs):
        super().__init__(**base_kwargs)
        self.loss = CrossEntropyLossIIT()
        self.id_to_coords = id_to_coords
        self.shuffle_train = False

    def build_graph(self):
        model = super().build_graph()
        IITmodel = IITModel(model, self.layers, self.id_to_coords, self.device)
        return IITmodel

    def batched_indices(self, max_len):
        batch_indices = [ x for x in range((max_len // self.batch_size))]
        output = []
        while len(batch_indices) != 0:
            batch_index = random.sample(batch_indices, 1)[0]
            batch_indices.remove(batch_index)
            output.append([batch_index*self.batch_size + x for x in range(self.batch_size)])
        return output

    def build_dataset(self, base, sources, base_y, IIT_y, coord_ids):
        base = torch.FloatTensor(np.array(base))
        sources = [torch.FloatTensor(np.array(source)) for source in sources]
        self.input_dim = base.shape[1]
        coord_ids = torch.FloatTensor(np.array(coord_ids))

        base_y = np.array(base_y)
        self.classes_ = sorted(set(base_y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        base_y = [class2index[label] for label in base_y]
        base_y = torch.tensor(base_y)

        IIT_y = np.array(IIT_y)
        IIT_y = [class2index[int(label)] for label in IIT_y]
        IIT_y = torch.tensor(IIT_y)

        bigX = torch.stack([base, coord_ids.unsqueeze(1).expand(-1, base.shape[1])] + sources, dim=1)
        bigy = torch.stack((IIT_y, base_y), dim=1)
        dataset = torch.utils.data.TensorDataset(bigX,bigy)
        return dataset

    def prep_input(self, base, sources, coord_ids):
        bigX = torch.stack([base, coord_ids.unsqueeze(1).expand(-1, base.shape[1])] + sources, dim=1)
        return bigX
