import random
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
from utils import progress_bar

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


class TorchTreeDataset(torch.utils.data.Dataset):
    def __init__(self, trees, y):
        assert len(trees) == len(y)
        self.trees = trees
        self.y = y

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        return (self.trees[idx], self.y[idx])

    @staticmethod
    def collate_fn(batch):
        X, y = zip(*batch)
        y = torch.tensor(y, dtype=torch.long)
        return X, y


class TorchTreeNNModel(nn.Module):
    def __init__(self, vocab, embed_dim, embedding, output_dim, hidden_activation):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.vocab_lookup = dict(zip(self.vocab, range(self.vocab_size)))
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim * 2
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim
        self.tree_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.embedding = self._define_embedding(embedding)
        self.classifier_layer = nn.Linear(self.embed_dim, self.output_dim)

    def _define_embedding(self, embedding):
        if embedding is None:
           return nn.Embedding(self.vocab_size, self.embed_dim)
        else:
            embedding = torch.tensor(embedding, dtype=torch.float)
            return nn.Embedding.from_pretrained(embedding)

    def forward(self, trees):
        roots = torch.zeros([len(trees), self.embed_dim])
        for i, tree in enumerate(trees):
            roots[i] = self.interpret(tree)
        return self.classifier_layer(roots)

    def interpret(self, subtree):
        if isinstance(subtree, str):
            i = self.vocab_lookup.get(subtree, self.vocab_lookup['$UNK'])
            ind = torch.tensor([i], dtype=torch.long)
            return self.embedding(ind)
        elif len(subtree) == 1:
            return self.interpret(subtree[0])
        else:
            left_subtree, right_subtree = subtree[0], subtree[1]
            left_subtree = self.interpret(left_subtree)
            right_subtree = self.interpret(right_subtree)
            combined = torch.cat((left_subtree, right_subtree), dim=1)
            root_rep = self.hidden_activation(self.tree_layer(combined))
            return root_rep


class TorchTreeNN(TorchModelBase):
    def __init__(self, vocab, embedding=None, embed_dim=50, **kwargs):
        self.vocab = vocab
        self.embedding = embedding
        self.embed_dim = embed_dim
        if self.embedding is not None:
            self.embed_dim = embedding.shape[1]
        super(TorchTreeNN, self).__init__(**kwargs)
        self.params += ['embed_dim', 'embedding']
        self.device = 'cpu'

    def fit(self, X, y, **kwargs):
        """Standard `fit` method.

        Parameters
        ----------
        X : np.array
        y : array-like
        kwargs : dict
            For passing other parameters. If 'X_dev' is included,
            then performance is monitored every 10 epochs; use
            `dev_iter` to control this number.

        Returns
        -------
        self

        """
        # Incremental performance:
        X_dev = kwargs.get('X_dev')
        if X_dev is not None:
            dev_iter = kwargs.get('dev_iter', 10)
        # Data prep:
        self.classes_ = sorted(set(y))
        self.n_classes_ = len(self.classes_)
        class2index = dict(zip(self.classes_, range(self.n_classes_)))
        y = [class2index[label] for label in y]
        # Dataset:
        dataset = TorchTreeDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            collate_fn=dataset.collate_fn)
        # Model:
        self.model = TorchTreeNNModel(
            vocab=self.vocab,
            embedding=self.embedding,
            embed_dim=self.embed_dim,
            output_dim=self.n_classes_,
            hidden_activation=self.hidden_activation)
        self.model.to(self.device)
        # Optimization:
        loss = nn.CrossEntropyLoss()
        optimizer = self.optimizer(self.model.parameters(), lr=self.eta)
        # Train:
        for iteration in range(1, self.max_iter+1):
            epoch_error = 0.0
            for X_batch, y_batch in dataloader:
                batch_preds = self.model(X_batch)
                err = loss(batch_preds, y_batch)
                epoch_error += err.item()
                optimizer.zero_grad()
                err.backward()
                optimizer.step()
            # Incremental predictions where possible:
            if X_dev is not None and iteration > 0 and iteration % dev_iter == 0:
                self.dev_predictions[iteration] = self.predict(X_dev)
            self.errors.append(epoch_error)
            progress_bar(
                "Finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, epoch_error))
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            preds = self.model(X)
            return torch.softmax(preds, dim=1).numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return [self.classes_[i] for i in probs.argmax(axis=1)]


def simple_example(initial_embedding=False):
    from nltk.tree import Tree

    train = [
        ["(N 1)", "odd"],
        ["(N 2)", "even"],
        ["(N (N 1))", "odd"],
        ["(N (N 2))", "even"],
        ["(N (N 1) (B (F +) (N 1)))", "even"],
        ["(N (N 1) (B (F +) (N 2)))", "odd"],
        ["(N (N 2) (B (F +) (N 1)))", "odd"],
        ["(N (N 2) (B (F +) (N 2)))", "even"],
        ["(N (N 1) (B (F +) (N (N 1) (B (F +) (N 2)))))", "even"]
    ]

    test = [
        ["(N (N 1) (B (F +) (N (N 1) (B (F +) (N 1)))))", "odd"],
        ["(N (N 2) (B (F +) (N (N 2) (B (F +) (N 2)))))", "even"],
        ["(N (N 2) (B (F +) (N (N 2) (B (F +) (N 1)))))", "odd"],
        ["(N (N 1) (B (F +) (N (N 2) (B (F +) (N 1)))))", "odd"],
        ["(N (N 2) (B (F +) (N (N 1) (B (F +) (N 2)))))", "odd"]
    ]

    vocab = ["1", "+", "2", "$UNK"]

    X_train, y_train = zip(*train)
    X_train = [Tree.fromstring(x) for x in X_train]

    X_test, y_test = zip(*test)
    X_test = [Tree.fromstring(x) for x in X_test]

    if initial_embedding:
        import numpy as np
        embedding = np.random.uniform(
            low=-1.0, high=1.0, size=(len(vocab), 50))
    else:
        embedding = None

    mod = TorchTreeNN(
        vocab,
        embed_dim=50,
        hidden_dim=50,
        max_iter=1000,
        embedding=embedding)

    mod.fit(X_train, y_train)

    print("\nTest predictions:")

    preds = mod.predict(X_test)

    for tree, label, pred in zip(X_test, y_test, preds):
        print("{}\n\tPredicted: {}\n\tActual: {}".format(tree, pred, label))


if __name__ == '__main__':
    simple_example()
