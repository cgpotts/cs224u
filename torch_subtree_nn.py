from nltk.tree import Tree
import torch
from torch_tree_nn import TorchTreeNNModel, TorchTreeNN

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


class TorchSubtreeNNModel(TorchTreeNNModel):

    def forward(self, tree):
        """The key changes: (i) we apply the classifier layer to all
        of the hidden representations, and (ii) we return the labels
        on all of the nodes, rather than just the label on the root.
        """
        reps, labels = self.interpret(tree, reps=[], labels=[])
        reps = torch.stack(reps)
        return self.classifier_layer(reps), labels

    def interpret(self, subtree, reps, labels):
        """The key changes: `reps` and `labels` store all of the
        hidden representations and labels in aligned flattened
        lists that can be passed to the PyTorch loss function.
        """
        # This is the preterminal case, like (positive happy)
        if len(list(subtree.subtrees())) == 1:
            i = self.vocab_lookup.get(subtree[0], self.vocab_lookup['$UNK'])
            ind = torch.tensor([i], dtype=torch.long)
            root = self.embedding(ind)
            return [root.squeeze()], [subtree.label()]
        # This is the non-branching case, like the root of
        # (positive (positive happy))
        elif len(subtree) == 1:
            reps, labels = self.interpret(subtree[0], reps, labels)
            new_reps = reps + [reps[-1]]
            new_labels = labels + [subtree.label()]
            return new_reps, new_labels
        # The branching case:
        else:
            left_subtree, right_subtree = subtree[0], subtree[1]
            left_reps, left_labels = self.interpret(left_subtree, reps, labels)
            left_root = left_reps[-1]
            right_reps, right_labels = self.interpret(right_subtree, reps, labels)
            right_root = right_reps[-1]
            combined = torch.cat((left_root, right_root), dim=0)
            root = self.hidden_activation(self.tree_layer(combined))
            new_reps = left_reps + right_reps + [root.squeeze()]
            new_labels = left_labels + right_labels + [subtree.label()]
            return new_reps, new_labels


class TorchSubtreeNN(TorchTreeNN):

    def build_graph(self):
        """This is the same as the `build_graph method for `TorchTreeNN`
        except that it uses  `TorchTreeNNModelWithSubtreeSupervision`.
        """
        self.model = TorchSubtreeNNModel(
            vocab=self.vocab,
            embedding=self.embedding,
            embed_dim=self.embed_dim,
            output_dim=self.n_classes_,
            hidden_activation=self.hidden_activation)

    @staticmethod
    def get_classes(X):
        """We have to recurse through all the trees to ensure that we
        see all the labels. (`TorchTreeNN` need only look at the root
        nodes.)
        """
        labels = set()
        for tree in X:
            for subtree in tree.subtrees():
                labels.add(subtree.label())
        return sorted(labels)

    def convert_label(self, labels):
        """Whereas `TorchTreeNN` has to convert just a single label
        when it processes an example, now we have to convert the full
        list of labels into indices.
        """
        indices = [self.class2index[label] for label in labels]
        return torch.LongTensor(indices)

    def predict_proba(self, X):
        """Returns a list of lists of prediction vectors, one list of
        vectors per tree in `X`.
        """
        self.model.eval()
        with torch.no_grad():
            preds = []
            for tree in X:
                pred, _ = self.model.forward(tree)
                pred = torch.softmax(pred, dim=1).numpy()
                preds.append(pred)
            return preds

    def predict(self, X):
        """Returns a list of lists of predictions, one list per tree
        in `X`.
        """
        preds = self.predict_proba(X)
        return [[self.classes_[p.argmax()] for p in pred] for pred in preds]

    def predict_proba_root(self, X):
        """Returns just the vector of predicted probabilities for the
        root nodes of the trees in `X`.
        """
        preds = self.predict_proba(X)
        return [pred[-1] for pred in preds]

    def predict_root(self, X):
        """Returns just the predicted classes for the root nodes of
        the trees in `X`.
        """
        preds = self.predict_proba_root(X)
        return [self.classes_[pred.argmax()] for pred in preds]


def simple_example():
    train = [
        "(odd 1)",
        "(even 2)",
        "(odd (odd 1))",
        "(even (even 2))",
        "(even (odd 1) (neutral (neutral +) (odd 1)))",
        "(odd (odd 1) (neutral (neutral +) (even 2)))",
        "(odd (even 2) (neutral (neutral +) (odd 1)))",
        "(even (even 2) (neutral (neutral +) (even 2)))",
        "(even (odd 1) (neutral (neutral +) (odd (odd 1) (neutral (neutral +) (even 2)))))"
    ]

    test = [
        "(odd (odd 1) (neutral (neutral +) (even (odd 1) (neutral (neutral +) (odd 1)))))",
        "(even (even 2) (neutral (neutral +) (even (even 2) (neutral (neutral +) (even 2)))))",
        "(odd (even 2) (neutral (neutral +) (odd (even 2) (neutral (neutral +) (odd 1)))))",
        "(odd (odd 1) (neutral (neutral +) (odd (even 2) (neutral (neutral +) (odd 1)))))",
        "(odd (even 2) (neutral (neutral +) (odd (odd 1) (neutral (neutral +) (even 2)))))"
    ]

    vocab = ["1", "+", "2", "$UNK"]

    X_train = [Tree.fromstring(x) for x in train]
    X_test = [Tree.fromstring(x) for x in test]

    mod = TorchSubtreeNN(
        vocab,
        embed_dim=50,
        hidden_dim=50,
        max_iter=500,
        embedding=None)

    mod.fit(X_train)

    print("\nTest predictions:")

    preds = mod.predict_root(X_test)

    y_test = [t.label() for t in X_test]

    for tree, label, pred in zip(X_test, y_test, preds):
        print("{}\n\tPredicted: {}\n\tActual: {}".format(tree, pred, label))


if __name__ == '__main__':
    simple_example()
