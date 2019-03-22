import tensorflow as tf
from tf_model_base import TfModelBase

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


class TfShallowNeuralClassifier(TfModelBase):
    def __init__(self, **kwargs):
        super(TfShallowNeuralClassifier, self).__init__(**kwargs)

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
        label2index = dict(zip(self.classes_, range(len(self.classes_))))
        y = [label2index[label] for label in y]
        dataset = tf.data.Dataset.from_tensor_slices(({'X': X}, {'y': y}))
        dataset = (dataset
                    .shuffle(len(X))
                    .repeat(self.max_iter)
                    .batch(self.batch_size))
        return dataset

    def _test_input_fn(self, X):
        dataset = tf.data.Dataset.from_tensor_slices({'X': X})
        dataset = dataset.batch(len(X))
        return dataset

    def _model_fn(self, features, labels, mode):
        features = features['X']
        # Graph:
        hidden = tf.layers.dense(
            features,
            self.batch_size,
            activation=self.hidden_activation)
        logits = tf.layers.dense(
            hidden,
            self.n_classes_)
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

    def predict_proba(self, X):
        input_fn = lambda: self._test_input_fn(X)
        return [x['proba'] for x in self.estimator.predict(input_fn)]

    def predict(self, X):
        input_fn = lambda: self._test_input_fn(X)
        return [self.classes_[x['pred']] for x in self.estimator.predict(input_fn)]


def simple_example():
    """Assess on the digits dataset."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    mod = TfShallowNeuralClassifier()

    print(mod)

    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)

    print(classification_report(y_test, predictions))

    return accuracy_score(y_test, predictions)


if __name__ == '__main__':
   simple_example()
