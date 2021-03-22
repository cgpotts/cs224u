import random
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


class TorchTreeNNModel(nn.Module):
    def __init__(self,
            vocab,
            embed_dim,
            embedding,
            output_dim,
            hidden_activation,
            freeze_embedding=False):
        """
        Defines the core computation graph for TorchTreeNN. At its heart,
        this is a standard tree-structured neural network with a simple
        combination function

        p = f([left;right] + b)

        where left and right are the representations of the child nodes, f
        is an activation function, and p is the representation of the parent.
        See `forward` for a decription of how it is computed with data
        structures that can be batched efficiently.

        """
        super().__init__()
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim * 2
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim
        self.tree_layer = nn.Linear(self.hidden_dim, self.embed_dim)
        self.freeze_embedding = freeze_embedding
        self.embedding = self._define_embedding(
            embedding, self.vocab_size, self.embed_dim, self.freeze_embedding)
        self.classifier_layer = nn.Linear(self.embed_dim, self.output_dim)

    def _define_embedding(self, embedding, vocab_size, embed_dim, freeze_embedding):
        if embedding is None:
            emb = nn.Embedding(vocab_size, embed_dim)
            emb.weight.requires_grad = not freeze_embedding
            return emb
        else:
            embedding = torch.FloatTensor(embedding)
            return nn.Embedding.from_pretrained(
                embedding, freeze=freeze_embedding)

    def forward(self, subtree_batch, subtree_lens_batch, emb_ind_batch):
        """
        Recursively interpret a batch of examples as formatted by
        `TorchTreeNN._build_tree_rep`. Each member of `emb_ind_batch`
        is a list of indices into our embedding space. We look them
        all up. A subset are actually lexical representations. The rest
        are modified by the intrpretatation loop. For example, the tree

             A
             |
          ------
          |    |
          B    E
          |
        -----
        |   |
        C   D

        is represented as

        emb_ind=[0, 0, i, j, k]

        and

        subtrees=[[2,2,2], [3,3,3], [4,4,4], [2, 3, 4] [0,1,2]].

        We create the (5, embed_dim) matrix reps. The first three subtrees
        are skipped, and the fourth modifies reps[2] by running
        f(reps[3];reps[4]), where f is the combination function. Finally,
        reps[0] is modified by processing f(reps[1];reps[2]). This mirrors
        the process of bottom-up, right-to-left interpretation.

        Parameters
        ----------
        subtree_batch : torch.LongTensor
            Shape (batch_size, max_batch_len, 3)
        subtree_lens_batch : torch.LongTensor
            Shape (batch_size, ). These are used to avoid processing
            padded elements of members of `subtree_batch`.
        emb_ind_batch : torch.LongTensor
            Shape (batch_size, max_batch_len)

        Returns
        -------
        torch.FloatTensor
            Shape (batch_size, embed_dim).

        """
        logits = []

        iterator = zip(subtree_batch, subtree_lens_batch, emb_ind_batch)
        for subtrees, subtree_len, emb_inds in iterator:
            reps = self.embedding(emb_inds)
            for i in range(subtree_len):
                parent, left, right = subtrees[i]
                # Skip the lexical subtrees; we don't actually want to
                # change them as though they were local trees.
                if left != right:
                    combined = torch.cat((reps[left], reps[right]), dim=0)
                    root_rep = self.hidden_activation(
                        self.tree_layer(combined))
                    reps[parent] = root_rep
            root = reps[0]
            logits.append(self.classifier_layer(root))
        logits = torch.stack(logits)
        return logits


class TorchTreeNN(TorchModelBase):
    def __init__(self,
            vocab,
            embedding=None,
            embed_dim=50,
            hidden_activation=nn.Tanh(),
            freeze_embedding=False,
            **base_kwargs):
        """
        Tree-structured Neural Network for classification problems.
        The network will work for any kind of classification task.

        Parameters
        ----------
        vocab : list of str
            This should be the vocabulary. It needs to be aligned with
            `embedding` in the sense that the ith element of vocab
            should be represented by the ith row of `embedding`. Ignored
            if `use_embedding=False`.

        embedding : np.array or None
            Each row represents a word in `vocab`, as described above.

        embed_dim : int
            Dimensionality for the initial embeddings. This is ignored
            if `embedding` is not None, as a specified value there
            determines this value. Also ignored if `use_embedding=False`.

        hidden_activation : nn.Module
            The non-activation function used by the network for the
            hidden layer. Default `nn.Tanh()`.

        freeze_embedding : bool
            If True, the embedding will be updated during training. If
            False, the embedding will be frozen. This parameter applies
            to both randomly initialized and pretrained embeddings.

        **base_kwargs
            For details, see `torch_model_base.py`.

        Attributes
        ----------
        vocab_size : int

        vocab_lookup : dict
            Look-up from vocab items to indices.

        loss: nn.CrossEntropyLoss(reduction="mean")

        self.params: list
            Extends TorchModelBase.params with names for all of the
            arguments for this class to support tuning of these values
            using `sklearn.model_selection` tools.

        """
        self.vocab = vocab
        self.embedding = embedding
        self.embed_dim = embed_dim
        if self.embedding is not None:
            self.embed_dim = embedding.shape[1]
        self.hidden_activation = hidden_activation
        self.freeze_embedding = freeze_embedding
        super().__init__(**base_kwargs)
        self.params += [
            'embed_dim',
            'embedding',
            'hidden_activation',
            'freeze_embedding']
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.vocab_lookup = dict(zip(self.vocab, range(self.vocab_size)))
        self.loss = nn.CrossEntropyLoss()

    def build_graph(self):
        """
        The core computation graph. This is called by `fit`, which sets
        the `self.model` attribute.

        Returns
        -------
        TorchTreeNNModel

        """
        model = TorchTreeNNModel(
            vocab=self.vocab,
            embedding=self.embedding,
            embed_dim=self.embed_dim,
            output_dim=self.n_classes_,
            hidden_activation=self.hidden_activation,
            freeze_embedding=self.freeze_embedding)

        self.embed_dim = model.embed_dim

        return model

    def build_dataset(self, trees, y=None):
        """
        Format data for training and prediction. This is somewhat
        involved. See `self._build_tree_rep` for a description of the
        core logic.

        Parameters
        ----------
        trees : list of nltk.Tree instances

        Returns
        -------
        torch.utils.data.TensorDataset

        """
        all_subtree_indices = []
        all_emb_indices = []
        all_subtree_lens = []
        for tree in trees:
            subtree, emb = self._tree2tensors(tree)
            all_subtree_indices.append(subtree)
            all_subtree_lens.append(len(subtree))
            all_emb_indices.append(emb)
        all_subtree_indices = torch.nn.utils.rnn.pad_sequence(
            all_subtree_indices, batch_first=True)
        all_emb_indices = torch.nn.utils.rnn.pad_sequence(
            all_emb_indices, batch_first=True)
        all_subtree_lens = torch.tensor(all_subtree_lens)
        if y is None:
            return torch.utils.data.TensorDataset(
                all_subtree_indices, all_subtree_lens, all_emb_indices)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            self.class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [self.class2index[x] for x in y]
            y = torch.tensor(y)
            return torch.utils.data.TensorDataset(
                all_subtree_indices, all_subtree_lens, all_emb_indices, y)

    def _tree2tensors(self, tree):
        subtree_indices, emb_indices, _ = self._build_tree_rep(tree)
        # Reverse the order so that the tree is interpreted bottom up
        # and right to left:
        subtree_indices = torch.tensor(subtree_indices[::-1])
        emb_indices = torch.tensor(emb_indices)
        return subtree_indices, emb_indices

    def _build_tree_rep(self, tree, n=0):
        """Turns an nltk.Tree `tree` into a list of subtree indices
        and a list of embedding  indices for terminal nodes (and False
        for non-terminal nodes). For example, the tree

          A
        -----
        |   |
        B   C

        becomes the list of subtrees [[0, 1, 2], [1, 1, 1], [2, 2, 2]]
        and the list of lexical signals [False, N, M], where N and M
        are the embedding indices for B and C according to
        `vocab_lookup`.

        Lexical items are signaled with triples [N, N, N]. The intention
        is that these will be skipped by the model that interprets
        these trees. They are included only so that even simple trees
        like

        A
        |
        B

        will have non-empty lists of subtrees.

        The algorithm does a left-to-right, depth-first traversal. Here
        is what that looks like in terms of indices:

                0
                |
            ----------
            1        4
            |        |
          -----    -----
          2   3    5   6
                       |
                     -----
                     |   |
                     7   8

        and the above tree then creates the list of subtrees

        [[0, 1, 4],
        [1, 2, 3],
        [2, 2, 2],
        [3, 3, 3],
        [4, 5, 6],
        [5, 5, 5],
        [6, 7, 8],
        [7, 7, 7],
        [8, 8, 8]]

        Parameters
        ----------
        tree : nltk.Tree
        vocab_lookup : dict
            Should map terminal nodes to embedding indices, and
            needs to include a key `$UNK` to handle unseen words.
        n : int
            Used when the function is called recursively.

        Returns
        -------
        subtree_indices: list of length-3 lists of node indices
        emb_index: list of int and False
        n: current node index

        """
        if isinstance(tree, str):
            # For lexical items, we create dummy local trees and skip
            # them during interpretation. This ensures that even
            # single-node trees have non-empty subtree sequences which
            # is important for padding and batching.
            subtree_indices = [[n, n, n]]
            emb_index = self.vocab_lookup.get(tree[0], self.vocab_lookup['$UNK'])
            emb_index = [emb_index]
            return subtree_indices, emb_index, n
        elif len(tree) == 1:
            return self._build_tree_rep(tree[0], n=n)
        else:
            subtree_indices = [n]
            emb_indices = [False]  # Used for non-lexical nodes.
            # Add the left child index:
            subtree_indices.append(n+1)
            # Now go recursively into the left daughter.
            l_ind, l_emb, n = self._build_tree_rep(tree[0], n=n+1)
            # Add the right child index:
            subtree_indices.append(n+1)
            # Now go recursively into the right daughter:
            r_ind, r_emb, n = self._build_tree_rep(tree[1], n=n+1)
        # Combine all of the info:
        subtree_indices = [subtree_indices] + l_ind + r_ind
        emb_indices += l_emb + r_emb
        return subtree_indices, emb_indices, n

    def predict_proba(self, X, device=None):
        """Predicted probabilities for the examples in `X`.

        Parameters
        ----------
        X : list of nltk.tree.Tree

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        np.array with shape (len(X), self.n_classes_)

        """
        preds = self._predict(X, device=device)
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        return probs

    def predict(self, X, device=None):
        """Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Parameters
        ----------
        X : list of nltk.tree.Tree

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        list of length len(X)

        """
        probs = self.predict_proba(X, device=device)
        return [self.classes_[i] for i in probs.argmax(axis=1)]

    def score(self, X, y, device=None):
        """
        Uses macro-F1 as the score function. Note: this departs from
        `sklearn`, where classifiers use accuracy as their scoring
        function. Using macro-F1 is more consistent with our course.

        This function can be used to evaluate models, but its primary
        use is in cross-validation and hyperparameter tuning.

        Parameters
        ----------
        X : list of nltk.Tree instances

        y : iterable, shape `len(n_examples)`
            These can be the raw labels. They will converted internally
            as needed. See `build_dataset`.

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        float

        """
        preds = self.predict(X, device=device)
        return utils.safe_macro_f1(y, preds)


def simple_example():
    from nltk.tree import Tree

    utils.fix_random_seeds()

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
    y_train = [t.label() for t in X_train]

    X_test = [Tree.fromstring(x) for x in test]
    y_test = [t.label() for t in X_test]

    mod = TorchTreeNN(vocab)

    print(mod)

    mod.fit(X_train, y_train)

    print("\nTest predictions:")

    preds = mod.predict(X_test)

    correct = 0
    for tree, label, pred in zip(X_test, y_test, preds):
        correct += int(correct == label)
        print("{}\n\tPredicted: {}\n\tActual: {}".format(tree, pred, label))
    print("{}/{} correct".format(correct, len(X_test)))

    return mod.score(X_test, y_test)


if __name__ == '__main__':
    simple_example()
