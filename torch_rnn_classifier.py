import numpy as np
from operator import itemgetter
import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2021"


class TorchRNNDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, seq_lengths, y=None):
        """
        Dataset class for RNN classifiers. The heavy-lifting is done by
        `collate_fn`, which handles the padding and packing necessary to
        efficiently process variable length sequences.

        Parameters
        ----------
        sequences : list of `torch.LongTensor`, `len(n_examples)`

        seq_lengths : torch.LongTensor, shape `(n_examples, )`

        y : None or torch.LongTensor, shape `(n_examples, )`
            If None, then we are in prediction mode. Otherwise, these are
            indices into the list of classes.

        """
        assert len(sequences) == len(seq_lengths)
        self.sequences = sequences
        self.seq_lengths = seq_lengths
        if y is not None:
            assert len(sequences) == len(y)
        self.y = y

    @staticmethod
    def collate_fn(batch):
        """
        Format a batch of examples for use in both training and prediction.

        Parameters
        ----------
        batch : tuple of length 2 (prediction) or 3 (training)
            The first element is the list of input sequences. The
            second is the list of lengths for those sequences. The third,
            where present, is the list of labels.

        Returns
        -------
        X : torch.Tensor, shape `(batch_size, max_batch_length)`
            As padded by `torch.nn.utils.rnn.pad_sequence.

        seq_lengths : torch.LongTensor, shape `(batch_size, )`

        y : torch.LongTensor, shape `(batch_size, )`
            Only for training. In the case where `y` cannot be turned into
            a Tensor, we assume it is because it is a list of variable
            length sequences and to use `torch.nn.utils.rnn.pad_sequence`.
            The hope is that this will accomodate sequence prediction.

        """
        batch_elements = list(zip(*batch))
        X = batch_elements[0]
        seq_lengths = batch_elements[1]
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
        seq_lengths = torch.tensor(seq_lengths)
        if len(batch_elements) == 3:
            y = batch_elements[2]
            # We can try to accommodate the case where `y` is a sequence
            # loss with potentially different lengths by resorting to
            # padding if creating a tensor is not possible:
            try:
                y = torch.tensor(y)
            except ValueError:
                y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
            return X, seq_lengths, y
        else:
            return X, seq_lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.sequences[idx], self.seq_lengths[idx], self.y[idx]
        else:
            return self.sequences[idx], self.seq_lengths[idx]


class TorchRNNModel(nn.Module):
    def __init__(self,
            vocab_size,
            embed_dim=50,
            embedding=None,
            use_embedding=True,
            rnn_cell_class=nn.LSTM,
            hidden_dim=50,
            bidirectional=False,
            freeze_embedding=False):
        """
        Defines the core RNN computation graph. For an explanation of the
        parameters, see `TorchRNNClassifierModel`. This class handles just
        the RNN components of the overall classifier model.
        `TorchRNNClassifierModel` uses the output states to create a
        classifier.

        """
        super().__init__()
        self.vocab_size = vocab_size
        self.use_embedding = use_embedding
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.freeze_embedding = freeze_embedding
        # Graph
        if self.use_embedding:
            self.embedding = self._define_embedding(
                embedding, vocab_size, self.embed_dim, self.freeze_embedding)
            self.embed_dim = self.embedding.embedding_dim
        self.rnn = rnn_cell_class(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional)

    def forward(self, X, seq_lengths):
        if self.use_embedding:
            X = self.embedding(X)
        embs = torch.nn.utils.rnn.pack_padded_sequence(
            X,
            batch_first=True,
            lengths=seq_lengths.cpu(),
            enforce_sorted=False)
        outputs, state = self.rnn(embs)
        return outputs, state

    @staticmethod
    def _define_embedding(embedding, vocab_size, embed_dim, freeze_embedding):
        if embedding is None:
            emb = nn.Embedding(vocab_size, embed_dim)
            emb.weight.requires_grad = not freeze_embedding
            return emb
        elif isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
            return nn.Embedding.from_pretrained(
                embedding, freeze=freeze_embedding)
        else:
            return embedding


class TorchRNNClassifierModel(nn.Module):
    def __init__(self, rnn, output_dim, classifier_activation):
        """
        Defines the core computation graph for `TorchRNNClassifier`. This
        involves using the outputs of a `TorchRNNModel` instance to
        build a softmax classifier:

        h[t] = rnn(x[t], h[t-1])
        h = f(h[-1].dot(W_hy) + b_h)
        y = softmax(hW + b_y)

        This class uses its `rnn` parameter to compute each `h[1]`, and
        then it adds the classifier parameters that use `h[-1]` as inputs.
        Where `bidirectional=True`, `h[-1]` is `torch.cat([h[0], h[-1])`.

        """
        super().__init__()
        self.rnn = rnn
        self.output_dim = output_dim
        self.hidden_dim = self.rnn.hidden_dim
        if self.rnn.bidirectional:
            self.classifier_dim = self.hidden_dim * 2
        else:
            self.classifier_dim = self.hidden_dim
        self.hidden_layer = nn.Linear(
            self.classifier_dim, self.hidden_dim)
        self.classifier_activation = classifier_activation
        self.classifier_layer = nn.Linear(
            self.hidden_dim, self.output_dim)

    def forward(self, X, seq_lengths):
        outputs, state = self.rnn(X, seq_lengths)
        state = self.get_batch_final_states(state)
        if self.rnn.bidirectional:
            state = torch.cat((state[0], state[1]), dim=1)
        h = self.classifier_activation(self.hidden_layer(state))
        logits = self.classifier_layer(h)
        return logits

    def get_batch_final_states(self, state):
        if self.rnn.rnn.__class__.__name__ == 'LSTM':
            return state[0].squeeze(0)
        else:
            return state.squeeze(0)


class TorchRNNClassifier(TorchModelBase):
    def __init__(self,
            vocab,
            hidden_dim=50,
            embedding=None,
            use_embedding=True,
            embed_dim=50,
            rnn_cell_class=nn.LSTM,
            bidirectional=False,
            freeze_embedding=False,
            classifier_activation=nn.ReLU(),
            **base_kwargs):
        """
        RNN-based Recurrent Neural Network for classification problems.
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

        use_embedding : bool
            If True, then incoming examples are presumed to be lists of
            elements of the vocabulary. If False, then they are presumed
            to be lists of vectors. In this case, the `embedding` and
            `embed_dim` arguments are ignored, since no embedding is needed
            and `embed_dim` is set by the nature of the incoming vectors.

        embed_dim : int
            Dimensionality for the initial embeddings. This is ignored
            if `embedding` is not None, as a specified value there
            determines this value. Also ignored if `use_embedding=False`.

        rnn_cell_class : class for PyTorch recurrent layer
            Should be just the class name, not an instance of the class.

        hidden_dim : int
            Dimensionality of the hidden layer in the RNN.

        bidirectional : bool
            If True, then the final hidden states from passes in both
            directions are used.

        freeze_embedding : bool
            If True, the embedding will be updated during training. If
            False, the embedding will be frozen. This parameter applies
            to both randomly initialized and pretrained embeddings.

        classifier_activation : nn.Module
            The non-activation function used by the network for the
            hidden layer of the classifier.

        **base_kwargs
            For details, see `torch_model_base.py`.

        Attributes
        ----------
        loss: nn.CrossEntropyLoss(reduction="mean")

        self.params: list
            Extends TorchModelBase.params with names for all of the
            arguments for this class to support tuning of these values
            using `sklearn.model_selection` tools.

        """
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.use_embedding = use_embedding
        self.embed_dim = embed_dim
        self.rnn_cell_class = rnn_cell_class
        self.bidirectional = bidirectional
        self.freeze_embedding = freeze_embedding
        self.classifier_activation = classifier_activation
        super().__init__(**base_kwargs)
        self.params += [
            'hidden_dim',
            'embed_dim',
            'embedding',
            'use_embedding',
            'rnn_cell_class',
            'bidirectional',
            'freeze_embedding',
            'classifier_activation']
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def build_graph(self):
        """
        The core computation graph. This is called by `fit`, which sets
        the `self.model` attribute.

        Returns
        -------
        TorchRNNModel

        """
        rnn = TorchRNNModel(
            vocab_size=len(self.vocab),
            embedding=self.embedding,
            use_embedding=self.use_embedding,
            embed_dim=self.embed_dim,
            rnn_cell_class=self.rnn_cell_class,
            hidden_dim=self.hidden_dim,
            bidirectional=self.bidirectional,
            freeze_embedding=self.freeze_embedding)

        model = TorchRNNClassifierModel(
            rnn=rnn,
            output_dim=self.n_classes_,
            classifier_activation=self.classifier_activation)

        self.embed_dim = rnn.embed_dim

        return model

    def build_dataset(self, X, y=None):
        """
        Format data for training and prediction.

        Parameters
        ----------
        X : list of lists
            The raw sequences. The lists are expected to contain
            elements of `self.vocab`. This method converts them to
            indices for PyTorch.

        y : list or None
            The raw labels. This method turns them into indices for
            PyTorch processing. If None, then we are in prediction
            mode.

        Returns
        -------
        TorchRNNDataset

        """
        X, seq_lengths = self._prepare_sequences(X)
        if y is None:
            return TorchRNNDataset(X, seq_lengths)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            return TorchRNNDataset(X, seq_lengths, y)

    def _prepare_sequences(self, X):
        """
        Internal method for turning X into a list of indices into
        `self.vocab` and calculating the true lengths of the elements
        in `X`.

        Parameters
        ----------
        X : list of lists, `len(n_examples)`

        Returns
        -------
        new_X : list of lists, `len(n_examples)`

        seq_lengths : torch.LongTensor, shape `(n_examples, )`

        """
        if self.use_embedding:
            new_X = []
            seq_lengths = []
            index = dict(zip(self.vocab, range(len(self.vocab))))
            unk_index = index['$UNK']
            for ex in X:
                seq = [index.get(w, unk_index) for w in ex]
                seq = torch.tensor(seq)
                new_X.append(seq)
                seq_lengths.append(len(seq))
        else:
            new_X = [torch.FloatTensor(ex) for ex in X]
            seq_lengths = [len(ex) for ex in X]
            self.embed_dim = X[0][0].shape[0]
        seq_lengths = torch.tensor(seq_lengths)
        return new_X, seq_lengths

    def score(self, X, y, device=None):
        """
        Uses macro-F1 as the score function. Note: this departs from
        `sklearn`, where classifiers use accuracy as their scoring
        function. Using macro-F1 is more consistent with our course.

        This function can be used to evaluate models, but its primary
        use is in cross-validation and hyperparameter tuning.

        Parameters
        ----------
        X: np.array, shape `(n_examples, n_features)`

        y: iterable, shape `len(n_examples)`
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

    def predict_proba(self, X, device=None):
        """
        Predicted probabilities for the examples in `X`.

        Parameters
        ----------
        X : np.array, shape `(n_examples, n_features)`

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        np.array, shape `(len(X), self.n_classes_)`
            Each row of this matrix will sum to 1.0.

        """
        preds = self._predict(X, device=device)
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        return probs

    def predict(self, X, device=None):
        """
        Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Parameters
        ----------
        X : np.array, shape `(n_examples, n_features)`

        device: str or None
            Allows the user to temporarily change the device used
            during prediction. This is useful if predictions require a
            lot of memory and so are better done on the CPU. After
            prediction is done, the model is returned to `self.device`.

        Returns
        -------
        list, length len(X)

        """
        probs = self.predict_proba(X, device=device)
        return [self.classes_[i] for i in probs.argmax(axis=1)]


def simple_example():
    utils.fix_random_seeds()

    vocab = ['a', 'b', '$UNK']

    # No b before an a
    train = [
        [list('ab'), 'good'],
        [list('aab'), 'good'],
        [list('abb'), 'good'],
        [list('aabb'), 'good'],
        [list('ba'), 'bad'],
        [list('baa'), 'bad'],
        [list('bba'), 'bad'],
        [list('bbaa'), 'bad'],
        [list('aba'), 'bad']]

    test = [
        [list('baaa'), 'bad'],
        [list('abaa'), 'bad'],
        [list('bbaa'), 'bad'],
        [list('aaab'), 'good'],
        [list('aaabb'), 'good']]

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    mod = TorchRNNClassifier(vocab)

    print(mod)

    mod.fit(X_train, y_train)

    preds = mod.predict(X_test)

    print("\nPredictions:")

    for ex, pred, gold in zip(X_test, preds, y_test):
        score = "correct" if pred == gold else "incorrect"
        print("{0:>6} - predicted: {1:>4}; actual: {2:>4} - {3}".format(
            "".join(ex), pred, gold, score))

    return mod.score(X_test, y_test)


if __name__ == '__main__':
    simple_example()
