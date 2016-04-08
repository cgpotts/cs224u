import sys
import numpy as np
import random
from collections import defaultdict
import copy
from utils import randvec, randmatrix, d_tanh, softmax, progress_bar


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2016"


class ClassifierRNN:
    """Very simple Recurrent Neural Network for classification
    problems. The structure of the network is as follows:

                                                  y 
                                                 /|
                                                b | W_hy
                                                  |
    h_0 -- W_hh --  h_1 -- W_hh -- h_2 -- W_hh -- h_3
                     |              |              |
                     | W_xh         | W_xh         | W_xh
                     |              |              |
                    x_1            x_2            x_3
    
    where x_i are the inputs, h_j are the hidden units, and y is a
    one-hot vector indicating the true label for this sequence. The
    parameters are W_xh, W_hh, W_hy, and the bias b. The inputs x_i
    come from a user-supplied embedding space for the vocabulary. These
    can either be random or pretrained. The network equations in brief:

        h[t] = tanh(x[t].dot(W_xh) + h[t-1].dot(W_hh))

        y = softmax(h[-1].dot(W_hy) + b)

    The network will work for any kind of classification task. For
    NLI, we process the premise and hypothesis in order and then
    use the final hidden state as the basis for the predictions:

                                 [1, 0, 0] (entailment, contradiction, neutral)
                                  / |
                                 b  |
                                    |
    h0 - h1 - h2 - h3 -  h4 -  h5 - h6   
         |    |    |      |    |    |    
         x3  x2   x1     x3   x2    x4
         |    |    |      |    |    | look-up in embedding space
       every dog danced every dog  moved
    
    """    
    def __init__(self,
            vocab,
            embedding,
            hidden_dim=20,
            eta=0.01,
            maxiter=100,
            epsilon=1.5e-8,
            display_progress=True):
        """
        Parameters
        ----------
        vocab : list of str
            This should be the vocabulary. It needs to be aligned with
            `embedding` in the sense that the ith element of vocab
            should be represented by the ith row of `embedding`.

        embedding : np.array
            Each row represents a word in `vocab`, as described above.

        hidden_dim : int (default: 10)
            Dimensionality for the hidden layer.

        eta : float (default: 0.05)
            Learning rate.

        maxiter : int (default: 100)
            Maximum number of training epochs for SGD.

        epsilon : float (default: 1.5e-8)
            Training terminates if the error reaches this point (or 
            `maxiter` is met).

        display_progress : bool (default: True)
            Whether to print progress reports to stderr.

        All of the above are set as attributes. In addition, `self.word_dim`
        is set to the dimensionality of the input representations.
        
        """
        self.vocab = dict(zip(vocab, range(len(vocab))))
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.eta = eta
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.display_progress = display_progress
        self.word_dim = len(embedding[0])

    def get_word_rep(self, w):
        """For getting the input representation of word `w` from `self.embedding`."""
        word_index = self.vocab[w]
        return self.embedding[word_index]
                        
    def fit(self, training_data):
        """Train the network.

        Parameters
        ----------
        training_data : list of pairs
            In each pair, the first element should be a list of items
            from the vocabulary (for the NLI task, this is the
            concatenation of the premise and hypothesis), and the
            second element should be the one-hot label vector.

        Attributes
        ----------
        self.output_dim : int
            Set based on the length of the labels in `training_data`.
        
        self.W_xh : np.array
            Dense connections between the word representations
            and the hidden layers. Random initialization.

        self.W_hh : np.array
            Dense connections between the hidden representations.
            Random initialization.

        self.W_hy : np.array
            Dense connections from the final hidden layer to
            the output layer. Random initialization.

        self.b : np.array
            Output bias. Initialized to all 0.
    
        """              
        self.output_dim = len(training_data[0][1])
        self.W_xh = randmatrix(self.word_dim, self.hidden_dim)
        self.W_hh = randmatrix(self.hidden_dim, self.hidden_dim)
        self.W_hy = randmatrix(self.hidden_dim, self.output_dim)
        self.b = np.zeros(self.output_dim)
        # SGD:
        iteration = 0
        error = sys.float_info.max
        while error > self.epsilon and iteration < self.maxiter:
            error = 0.0
            random.shuffle(training_data)
            for seq, labels in training_data:
                self._forward_propagation(seq)
                # Cross-entropy error reduces to log(prediction-for-correct-label):
                error += -np.log(self.y[np.argmax(labels)])
                # Back-prop:
                d_W_hy, d_b, d_W_hh, d_W_xh = self._backward_propagation(seq, labels)
                # Updates:
                self.W_hy -= self.eta * d_W_hy
                self.b -= self.eta * d_b
                self.W_hh -= self.eta * d_W_hh
                self.W_xh -= self.eta * d_W_xh
            iteration += 1
            if self.display_progress:
                # Report the average error:
                error /= len(training_data)
                progress_bar("Finished epoch %s of %s; error is %s" % (iteration, self.maxiter, error))
        if self.display_progress:
            sys.stderr.write('\n')
            
    def _forward_propagation(self, seq):
        """
        Parameters
        ----------
        seq : list
            Variable length sequence of elements in the vocabulary.

        Attributes
        ----------
        self.h : np.array
            Each row is for a hidden representation. The first row
            is an all-0 initial state. The others correspond to
            the inputs in seq.

        self.y : np.array
            The vector of predictions.
        """
        self.h = np.zeros((len(seq)+1, self.hidden_dim))
        for t in range(1, len(seq)+1):
            word_rep = self.get_word_rep(seq[t-1])
            self.h[t] = np.tanh(word_rep.dot(self.W_xh) + self.h[t-1].dot(self.W_hh))
        self.y = softmax(self.h[-1].dot(self.W_hy) + self.b)

    def _backward_propagation(self, seq, y_):
        """
        Parameters
        ----------
        seq : list
            Variable length sequence of elements in the vocabulary. This
            is needed both for its lengths and for its input representations.

        y_ : list
            The label vector.

        Returns
        -------
        tuple
            The matrices of derivatives (d_W_hy, d_b, d_W_hh, d_W_xh).
        
        """            
        # Output errors:
        y_err = self.y
        y_err[np.argmax(y_)] -= 1
        h_err = y_err.dot(self.W_hy.T) * d_tanh(self.h[-1])
        d_W_hy = np.outer(self.h[-1], y_err)
        d_b = y_err
        # For accumulating the gradients through time:
        d_W_hh = np.zeros(self.W_hh.shape)
        d_W_xh = np.zeros(self.W_xh.shape)
        # Back-prop through time; the +1 is because the 0th
        # hidden state is the all-0s initial state.
        num_steps = len(seq)+1
        for t in reversed(range(1, num_steps)):
            d_W_hh += np.outer(self.h[t], h_err)
            word_rep = self.get_word_rep(seq[t-1])
            d_W_xh += np.outer(word_rep, h_err)
            h_err = h_err.dot(self.W_hh.T) * d_tanh(self.h[t])
        return (d_W_hy, d_b, d_W_hh, d_W_xh)
    
    def predict(self, seq):
        """
        Parameters
        ----------
        seq : list
            Variable length sequence of elements in the vocabulary.

        Returns
        -------
        int
            The index of the highest probability class according to
            the model.
        
        """
        self._forward_propagation(seq)
        return np.argmax(self.y)
        
######################################################################    

if __name__ == '__main__':

    T = 'T'
    F = 'F'
    
    train = [
        # p  q      XOR
        ([T ,T], [1.,   0.]),
        ([T, F], [0.,   1.]),
        ([F, T], [0.,   1.]),
        ([F, F], [1.,   0.])]
    
    vocab = [T, F]
    embedding = np.array([randvec(10) for _ in vocab])
    
    mod = ClassifierRNN(vocab=vocab, embedding=embedding, maxiter=1000)
    mod.fit(copy.copy(train))
    
    for x, y in train:
        p = mod.predict(x)
        print(p == np.argmax(y), mod.y, y)

