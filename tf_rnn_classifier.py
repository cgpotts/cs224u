import tensorflow as tf
from tf_model_base import TfModelBase

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


class TfRNNClassifier(TfModelBase):
    def __init__(self,
            vocab,
            embedding=None,
            embed_dim=50,
            train_embedding=True,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            **kwargs):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.train_embedding = train_embedding
        self.cell_class = cell_class
        super(TfRNNClassifier, self).__init__(**kwargs)
        self.params += [
            'embedding', 'embed_dim', 'train_embedding']

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        self.n_classes_ = len(self.classes_)
        self.estimator = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.model_dir)
        input_fn = lambda: self._train_input_fn(X, y)
        self.estimator.train(input_fn)
        return self

    def _train_input_fn(self, X, y):
        shapes = ({'indices': [None], 'length': ()}, {'y': ()})
        defaults = ({'indices': 0, 'length': 0}, {'y': 0})
        output_types = (
            {'indices': tf.int32, 'length': tf.int32},
            {'y': tf.int32})
        dataset = tf.data.Dataset.from_generator(
            lambda: self._dataset_generator(X, y),
            output_types=output_types,
            output_shapes=shapes)
        dataset = (dataset
                    .shuffle(len(X))
                    .repeat(self.max_iter)
                    .padded_batch(
                        batch_size=self.batch_size,
                        padded_shapes=shapes,
                        padding_values=defaults))
        return dataset

    def _test_input_fn(self, X):
        shapes = {'indices': [None], 'length': ()}
        defaults = {'indices': 0, 'length': 0}
        output_types = {'indices': tf.int32, 'length': tf.int32}
        dataset = tf.data.Dataset.from_generator(
            lambda: self._dataset_generator(X),
            output_types=output_types,
            output_shapes=shapes)
        dataset = dataset.padded_batch(
            batch_size=self.batch_size,
            padded_shapes=shapes,
            padding_values=defaults)
        return dataset

    def _model_fn(self, features, labels, mode):
        indices = features['indices']
        sequence_length = features['length']
        # Graph:
        self.embedding_ = self.define_or_get_embedding()
        feats = tf.nn.embedding_lookup(
            self.embedding_, indices)
        self.cell = self.cell_class(
            self.hidden_dim, activation=self.hidden_activation)
        outputs, state = tf.nn.dynamic_rnn(
            self.cell,
            feats,
            dtype=tf.float32,
            sequence_length=sequence_length)
        last = self._get_final_state(self.cell, state)
        logits = tf.layers.dense(last, self.n_classes_)
        # Predictions:
        preds = tf.argmax(logits, axis=-1)
        # Predicting:
        if mode == tf.estimator.ModeKeys.PREDICT:
            proba = tf.nn.softmax(logits)
            results = {'proba': proba, 'pred': preds}
            return tf.estimator.EstimatorSpec(mode, predictions=results)
        else:
            labels = labels['y']
            loss = tf.losses.sparse_softmax_cross_entropy(
                logits=logits, labels=labels)
            metrics = {
                'accuracy': tf.metrics.accuracy(labels, preds)
            }
            # Evaluation mode to enable early stopping based on metrics:
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)
            # Training:
            elif mode == tf.estimator.ModeKeys.TRAIN:
                global_step = tf.train.get_or_create_global_step()
                train_op = tf.train.AdamOptimizer(self.eta).minimize(
                    loss, global_step=global_step)
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

    def define_or_get_embedding(self):
        if self.embedding is None:
            shape = [self.vocab_size, self.embed_dim]
            embedding_intializer = None
        else:
            embedding_intializer = tf.Variable(
                self.embedding, dtype=tf.float32)
            shape = None
            self.embed_dim = embedding_intializer.shape[1]
        return tf.get_variable(
            'embedding',
            shape=shape,
            initializer=embedding_intializer,
            trainable=self.train_embedding)

    def predict_proba(self, X):
        input_fn = lambda: self._test_input_fn(X)
        return [x['proba'] for x in self.estimator.predict(input_fn)]

    def predict(self, X):
        input_fn = lambda: self._test_input_fn(X)
        return [self.classes_[x['pred']] for x in self.estimator.predict(input_fn)]

    def _dataset_generator(self, X, y=None):
        labels2index = dict(zip(self.classes_, range(len(self.classes_))))
        index = dict(zip(self.vocab, range(len(self.vocab))))
        unk_index = index['$UNK']
        for i, ex in enumerate(X):
            ex_len = len(ex)
            indices = [index.get(w, unk_index) for w in ex]
            if y is None:
                yield {'indices': indices, 'length': ex_len}
            else:
                yield ({'indices': indices, 'length': ex_len},
                       {'y': labels2index[y[i]]})

    def _define_embedding(self, embedding):
        """Build the embedding matrix. If the user supplied a matrix, it
        is converted into a Tensor, else a random Tensor is built. This
        method sets `self.embedding` for use and returns None.
        """
        if embedding is None:
            return tf.get_variable(
                'embedding',
                shape=[self.vocab_size, self.embed_dim],
                #initializer= tf.random_uniform(
                #    [self.vocab_size, self.embed_dim], -1.0, 1.0),
                trainable=self.train_embedding)
        else:
            embed = tf.Variable(
                initial_value=self.embedding,
                dtype=tf.float32,
                trainable=self.train_embedding)
            embed = embedding.shape[1]
            return embed

    @staticmethod
    def _get_final_state(cell, state):
        """Get the final state from an RNN, managing differences in
        the TensorFlow API for cells.

        Parameters
        ----------
        cell : tf.nn.rnn_cell instance
        state : second argument returned by `tf.nn.dynamic_rnn`

        Returns
        -------
        Tensor

        """
        # If the cell is LSTMCell, then `state` is an `LSTMStateTuple`
        # and we want the second (output) Tensor -- see
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
        #
        if isinstance(cell, tf.nn.rnn_cell.LSTMCell):
            return state[1]
        else:
            return state


def simple_example(initial_embedding=False):
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
        [list('aba'), 'bad']
    ]

    test = [
        [list('baaa'), 'bad'],
        [list('abaa'), 'bad'],
        [list('bbaa'), 'bad'],
        [list('aaab'), 'good'],
        [list('aaabb'), 'good']
    ]

    if initial_embedding:
        import numpy as np
        embedding = np.random.uniform(
            low=-1.0, high=1.0, size=(len(vocab), 50))
    else:
        embedding = None

    mod = TfRNNClassifier(
        vocab=vocab,
        max_iter=1000,
        embed_dim=50,
        embedding=embedding,
        hidden_dim=50)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, y_test = zip(*test)

    preds = mod.predict(X_test)

    print("\nPredictions:")

    for ex, pred, gold in zip(X_test, preds, y_test):
        score = "correct" if pred == gold else "incorrect"
        print("{0:>6} - predicted: {1:>4}; actual: {2:>4} - {3}".format(
            "".join(ex), pred, gold, score))


if __name__ == '__main__':
    simple_example()
