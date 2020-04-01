import random
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
from utils import progress_bar

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


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

    def forward(self, tree):
        """Recursively interprets `tree`, applying a classifier layer
        to the final representation.

        Parameters
        ----------
        tree : nltk.tree.Tree

        Returns
        -------
        torch.LongTensor, label (str)

        """
        root = self.interpret(tree)
        return self.classifier_layer(root)

    def interpret(self, subtree):
        # Terminal nodes are str:
        if isinstance(subtree, str):
            i = self.vocab_lookup.get(subtree, self.vocab_lookup['$UNK'])
            ind = torch.tensor([i], dtype=torch.long)
            return self.embedding(ind)
        # Non-branching nodes:
        elif len(subtree) == 1:
            return self.interpret(subtree[0])
        # Branching nodes:
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

    def build_graph(self):
        return TorchTreeNNModel(
            vocab=self.vocab,
            embedding=self.embedding,
            embed_dim=self.embed_dim,
            output_dim=self.n_classes_,
            hidden_activation=self.hidden_activation)

    def fit(self, X, y=None, **kwargs):
        """Fairly standard `fit` method except that, if `y=None`,
        then the labels `y` are presumed to come from the root nodes
        of the trees in `X`. We retain the option of giving them
        as a separate argument for consistency with the other model
        interfaces, and so that we can use sklearn cross-validation
        methods with this class.

        Parameters
        ----------
        X : list of `nltk.Tree` instances
        y : iterable of labels, or None
        kwargs : dict
            For passing other parameters. If 'X_dev' is included,
            then performance is monitored every 10 epochs; use
            `dev_iter` to control this number.

        Returns
        -------
        self

        """
        # Labels:
        if y is None:
            y = [t.label() for t in X]
        self.classes_ = sorted(set(y))
        self.n_classes_ = len(self.classes_)
        self.class2index = dict(zip(self.classes_, range(self.n_classes_)))

        # Incremental performance:
        X_dev = kwargs.get('X_dev')
        if X_dev is not None:
            dev_iter = kwargs.get('dev_iter', 10)

        # Model:
        if not self.warm_start or not hasattr(self, "model"):
            self.model = self.build_graph()
            self.opt = self.optimizer(
                self.model.parameters(),
                lr=self.eta,
                weight_decay=self.l2_strength)
        self.model.to(self.device)
        self.model.train()

        # Optimization:
        loss = nn.CrossEntropyLoss()

        # Train:
        dataset = list(zip(X, y))
        for iteration in range(1, self.max_iter+1):
            epoch_error = 0.0
            random.shuffle(dataset)
            for tree, label in dataset:
                pred = self.model.forward(tree)
                label = self.convert_label(label)
                err = loss(pred, label)
                epoch_error += err.item()
                self.opt.zero_grad()
                err.backward()
                self.opt.step()
            # Incremental predictions where possible:
            if X_dev is not None and iteration > 0 and iteration % dev_iter == 0:
                self.dev_predictions[iteration] = self.predict(X_dev)
                self.model.train()
            self.errors.append(epoch_error)
            progress_bar(
                "Finished epoch {} of {}; error is {}".format(
                    iteration, self.max_iter, epoch_error/len(X)))
        return self

    def convert_label(self, label):
        """Convert a class label to a format that PyTorch can handle.

        Parameters
        ----------
        label : member of `self.classes_`

        Returns
        -------
        torch.LongTensor of length 1

        """
        i = self.class2index[label]
        return torch.LongTensor([i])

    def predict_proba(self, X):
        """Predicted probabilities for the examples in `X`.

        Parameters
        ----------
        X : list of nltk.tree.Tree

        Returns
        -------
        np.array with shape (len(X), self.n_classes_)

        """
        self.model.eval()
        with torch.no_grad():
            preds = []
            for tree in X:
                pred = self.model.forward(tree)
                preds.append(pred.squeeze())
            preds = torch.stack(preds)
            return torch.softmax(preds, dim=1).numpy()

    def predict(self, X):
        """Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Parameters
        ----------
        X : list of nltk.tree.Tree

        Returns
        -------
        list of length len(X)

        """
        probs = self.predict_proba(X)
        return [self.classes_[i] for i in probs.argmax(axis=1)]


def simple_example(initial_embedding=False, separate_y=False):
    from nltk.tree import Tree

    train = [
        "(odd 1)",
        "(even 2)",
        "(even (odd 1) (neutral (neutral +) (odd 1)))",
        "(odd (odd 1) (neutral (neutral +) (even 2)))",
        "(odd (even 2) (neutral (neutral +) (odd 1)))",
        "(even (even 2) (neutral (neutral +) (even 2)))",
        "(even (odd 1) (neutral (neutral +) (odd (odd 1) (neutral (neutral +) (even 2)))))"]

    test = [
        "(odd (odd 1))",
        "(even (even 2))",
        "(odd (odd 1) (neutral (neutral +) (even (odd 1) (neutral (neutral +) (odd 1)))))",
        "(even (even 2) (neutral (neutral +) (even (even 2) (neutral (neutral +) (even 2)))))",
        "(odd (even 2) (neutral (neutral +) (odd (even 2) (neutral (neutral +) (odd 1)))))",
        "(even (odd 1) (neutral (neutral +) (odd (even 2) (neutral (neutral +) (odd 1)))))",
        "(odd (even 2) (neutral (neutral +) (odd (odd 1) (neutral (neutral +) (even 2)))))"]

    vocab = ["1", "+", "2", "$UNK"]

    X_train = [Tree.fromstring(x) for x in train]
    X_test = [Tree.fromstring(x) for x in test]

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
        max_iter=50,
        embedding=embedding)

    if separate_y:
        y = [t.label() for t in X_train]
    else:
        y = None

    mod.fit(X_train, y=y)

    print("\nTest predictions:")

    preds = mod.predict(X_test)

    y_test = [t.label() for t in X_test]

    correct = 0
    for tree, label, pred in zip(X_test, y_test, preds):
        if pred == label:
            correct += 1
        print("{}\n\tPredicted: {}\n\tActual: {}".format(tree, pred, label))
    print("{}/{} correct".format(correct, len(X_test)))


if __name__ == '__main__':
    simple_example(separate_y=True)
