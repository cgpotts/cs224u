import numpy as np
import pandas as pd
import tensorflow as tf
from tf_model_base import TfModelBase

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2019"


class TfAutoencoder(TfModelBase):
    def __init__(self, **kwargs):
        super(TfAutoencoder, self).__init__(**kwargs)

    def fit(self, X):
        self.output_dim_ = X.shape[1]
        self.input_dim_ = X.shape[1]
        self.estimator = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.model_dir)
        input_fn = lambda: self._train_input_fn(X)
        self.estimator.train(input_fn)
        # Hidden representations:
        hidden_input_fn = lambda: self._test_input_fn(X)
        H = np.array([x['hidden'] for x in self.estimator.predict(hidden_input_fn)])
        if isinstance(X, pd.DataFrame):
            H = pd.DataFrame(H, index=X.index)
        return H

    def _train_input_fn(self, X):
        dataset = tf.data.Dataset.from_tensor_slices(({'X': X}, {'y':X}))
        dataset = (dataset
                    .shuffle(X.shape[0])
                    .repeat(self.max_iter)
                    .batch(self.batch_size))
        return dataset

    def _test_input_fn(self, X):
        dataset = tf.data.Dataset.from_tensor_slices({'X': X})
        dataset = dataset.batch(X.shape[0])
        return dataset

    def _model_fn(self, features, labels, mode):
        features = features['X']
        # Graph:
        hidden = tf.layers.dense(
            features,
            self.hidden_dim,
            activation=self.hidden_activation)
        preds = tf.layers.dense(
            hidden,
            self.output_dim_)
        # Predicting:
        if mode == tf.estimator.ModeKeys.PREDICT:
            results = {'hidden': hidden, 'pred': preds}
            return tf.estimator.EstimatorSpec(mode, predictions=results)
        else:
            labels = labels['y']
            loss = tf.losses.mean_squared_error(
                labels=labels,
                predictions=preds,
                weights=0.5)
            metrics = {
                'mse': tf.metrics.mean_squared_error(labels, preds)
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

    def predict(self, X):
        input_fn = lambda: self._test_input_fn(X)
        X_pred = [x['pred'] for x in self.estimator.predict(input_fn)]
        if isinstance(X, pd.DataFrame):
            X_pred = pd.DataFrame(X_pred, index=X.index)
        return X_pred


def simple_example():

    def randmatrix(m, n, sigma=0.1, mu=0):
        return sigma * np.random.randn(m, n) + mu

    rank = 20
    nrow = 1000
    ncol = 100

    X = randmatrix(nrow, rank).dot(randmatrix(rank, ncol))
    ae = TfAutoencoder(hidden_dim=rank, max_iter=200)
    H = ae.fit(X)
    X_pred = ae.predict(X)
    mse = (0.5*(X_pred - X)**2).mean()
    print("MSE between actual and reconstructed: {}".format(mse))
    print("Hidden representations")
    print(H)
    return mse

if __name__ == '__main__':
   simple_example()
