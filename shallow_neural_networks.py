import sys
import copy
import random
import numpy as np
from numpy import dot, outer
import utils

try:
    import tensorflow as tf    
except ImportError:
    print("Warning: TensorFlow is not installed, so you won't be able to use `TfShallowNeuralNetwork`.")


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2016"


def d_tanh(z):
    """The derivative of the hyperbolic tangent function. 
    z should be a float or np-array."""
    return 1.0 - z**2


class ShallowNeuralNetwork:
    """Fit a model f(f(xW1 + b1)W2 + b2)"""    
    def __init__(self, 
            hidden_dim=40,
            afunc=np.tanh, 
            d_afunc=d_tanh,
            maxiter=100,
            eta=0.05,
            epsilon=1.5e-8,
            display_progress=True):
        """All the parameters are set as attributes.
        
        Parameters
        ----------
        hidden_dim : int (default: 40)
            Dimensionality of the hidden layer.
            
        afunc : vectorized activation function (default: np.tanh)
            The non-linear activation function used by the 
            network for the hidden and output layers.
            
        d_afunc : vectorized activation function derivative (default: `d_tanh`)
            The derivative of `afunc`. It does not ensure that this 
            matches `afunc`, and craziness will result from mismatches!

        maxiter : int default: 100)
            Maximum number of training epochs.
            
        eta : float (default: 0.05)
            Learning rate.
            
        epsilon : float (default: 1.5e-8)
            Training terminates if the error reaches this point (or 
            `maxiter` is met).
                    
        display_progress : bool (default: True)
           Whether to use the simple over-writing `progress_bar`
           to show progress.                    
        
        """
        self.input_dim = None  # Set by the training data.
        self.output_dim = None # Set by the training data.
        self.hidden_dim = hidden_dim        
        self.afunc = afunc 
        self.d_afunc = d_afunc 
        self.maxiter = maxiter
        self.eta = eta        
        self.epsilon = epsilon
        self.display_progress = display_progress
                
    def forward_propagation(self, ex): 
        """Computes the forward pass. ex shoud be a vector 
        of the same dimensionality as self.input_dim.
        No value is returned, but the output layer self.y
        is updated, as are self.x and self.h"""        
        self.x[ : -1] = ex # ignore the bias
        self.h[ : -1] = self.afunc(dot(self.x, self.W1)) # ignore the bias
        self.y = self.afunc(dot(self.h, self.W2))        
        
    def backward_propagation(self, y_):
        """Send the error signal back through the network.
        y_ is the ground-truth label we compare against."""
        y_ = np.array(y_)       
        self.y_err = (y_ - self.y) * self.d_afunc(self.y)
        h_err = dot(self.y_err, self.W2.T) * self.d_afunc(self.h)
        self.W2 += self.eta * outer(self.h, self.y_err)
        self.W1 += self.eta * outer(self.x, h_err[:-1]) # ignore the bias
        return np.sum(0.5 * (y_ - self.y)**2)

    def fit(self, training_data): 
        """The training algorithm. 
        
        Parameters
        ----------
        training_data : list
            A list of (example, label) pairs, where `example`
            and `label` are both np.array instances.
        
        Attributes
        ----------
        self.x : the input layer 
        self.h : the hidden layer
        self.y : the output layer
        self.W1 : dense weight connection from self.x to self.h
        self.W2 : dense weight connection from self.h to self.y
        
        Both self.W1 and self.W2 have the bias as their final column.
        
        The following attributes are created here for efficiency but 
        used only in `backward_propagation`:
        
        self.y_err : vector of output errors
        self.x_err : vector of input errors
        
        """
        # Dimensions determined by the data:
        self.input_dim = len(training_data[0][0])
        self.output_dim = len(training_data[0][1])
        # Parameter initialization:
        self.x = np.ones(self.input_dim+1)  # +1 for the bias                                         
        self.h = np.ones(self.hidden_dim+1) # +1 for the bias        
        self.y = np.ones(self.output_dim)        
        self.W1 = utils.randmatrix(self.input_dim+1, self.hidden_dim)
        self.W2 = utils.randmatrix(self.hidden_dim+1, self.output_dim)        
        self.y_err = np.zeros(self.output_dim)
        self.x_err = np.zeros(self.input_dim+1)
        # SGD:
        iteration = 0
        error = sys.float_info.max
        while error > self.epsilon and iteration < self.maxiter:            
            error = 0.0
            random.shuffle(training_data)
            for ex, labels in training_data:
                self.forward_propagation(ex)
                error += self.backward_propagation(labels)
            iteration += 1
            if self.display_progress:
                utils.progress_bar('completed iteration %s; error is %s' % (iteration, error))
        if self.display_progress:
            sys.stderr.write('\n')
                    
    def predict(self, ex):
        """Prediction for `ex`, which must be featurized as the
        training data were. Simply runs `foward_propagation` and
        returns a copy of self.y."""
        self.forward_propagation(ex)
        return copy.deepcopy(self.y)    
    
######################################################################

class TfShallowNeuralNetwork:
    """Fairly exact reproduction of `ShallowNeuralNetwork` in
    TensorFlow, differing only in some details of optimization."""
    def __init__(self, hidden_dim=40, maxiter=100, eta=0.05):
        """All the parameters are set as attributes.
        
        Parameters
        ----------
        hidden_dim : int (default: 40)
            Dimensionality of the hidden layer.                   

        maxiter : int default: 100)
            Maximum number of training epochs.
            
        eta : float (default: 0.05)
            Learning rate.                 
        
        """
        self.input_dim = None
        self.hidden_dim = hidden_dim
        self.output_dim = None
        self.maxiter = maxiter
        self.eta = eta            
                
    def fit(self, training_data):
        """The training algorithm. 
        
        Parameters
        ----------
        training_data : list
            A list of (example, label) pairs, where `example`
            and `label` are both np.array instances.
        
        Attributes
        ----------
        self.sess : the TensorFlow session
        self.x : place holder for input data
        self.h : the hidden layer
        self.y : the output layer -- more like the full model here.
        self.W1 : dense weight connection from self.x to self.h
        self.b1 : bias
        self.W2 : dense weight connection from self.h to self.y
        self.b2 : bias
        self.y_ : placeholder for training data
                
        """
        self.sess = tf.InteractiveSession()
        # Dimensions determined by the data:
        self.input_dim = len(training_data[0][0])
        self.output_dim = len(training_data[0][1])        
        # Network initialization. For the inputs x, None in the first
        # dimension allows us to train and evaluate on datasets
        # of different size.
        self.x = tf.placeholder(tf.float32, [None, self.input_dim])
        self.W1 = tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim]))
        self.b1 = tf.Variable(tf.random_normal([self.hidden_dim]))
        self.W2 = tf.Variable(tf.random_normal([self.hidden_dim, self.output_dim]))
        self.b2 = tf.Variable(tf.random_normal([self.output_dim]))
        # Network structure. As before, we use tanh for both 
        # layers. This is not strictly necessary, and TensorFlow
        # makes it easier to try different combinations.
        self.h = tf.nn.tanh(tf.matmul(self.x, self.W1) + self.b1)    
        self.y = tf.nn.tanh(tf.matmul(self.h, self.W2) + self.b2)        
        # A place holder for the true labels. None in the first
        # dimension allows us to train and evaluate on datasets
        # of different size.
        self.y_ = tf.placeholder(tf.float32, [None, self.output_dim])
        # This defines the objective as one of reducing the 
        # one-half squared total error. This could easily 
        # be made into a user-supplied parameter to facilitate
        # exploration of other costs. See
        # https://www.tensorflow.org/versions/r0.7/api_docs/python/math_ops.html#reduction
        cost = tf.reduce_sum(0.5 * (self.y_ - self.y)**tf.constant(2.0))
        # Simple GradientDescent (as opposed to the stochastic version
        # used by `ShallowNeuralNetwork`). For more options, see
        # https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#optimizers
        self.optimizer = tf.train.GradientDescentOptimizer(self.eta).minimize(cost)
        # TF session initialization:   
        init = tf.initialize_all_variables()
        self.sess.run(init)
        # Train (for larger datasets, the epochs should be batched):
        x, y_ = zip(*training_data)
        for iteration in range(self.maxiter):
            self.sess.run(self.optimizer, feed_dict={self.x: x, self.y_: y_})

    def predict(self, ex):
        """
        Prediction for `ex`. This runs the model (forward propagation with
        self.x replaced by the single example `ex`).

        Parameters
        ----------
        ex : np.array
          Must be featurized as the training data were.

        Returns
        -------
        np.array
            The predicted outputs, dimension self.output_dim. TensorFlow
            assumes self.x is a list of examples and so returns a list of
            predictions. Since we're classifying just one, we return the
            list's only member.
            
        """
        return self.sess.run(self.y, feed_dict={self.x: [ex]})[0]



if __name__ == '__main__':

    def logical_operator_example(net):
        train = [
            # p  q    (p=q) (p v q)
            ([1.,1.], [1.,   1.]), # T T ==> T, T
            ([1.,0.], [0.,   1.]), # T F ==> F, T
            ([0.,1.], [0.,   1.]), # F T ==> F, T
            ([0.,0.], [1.,   0.])] # F F ==> T, F
        net.fit(copy.deepcopy(train))        
        for ex, labels in train:
            prediction = net.predict(ex)
            print(ex, labels, np.round(prediction, 2))

    print('From scratch')
    logical_operator_example(ShallowNeuralNetwork(hidden_dim=4, maxiter=1000))
    print('TensorFlow')
    logical_operator_example(TfShallowNeuralNetwork(hidden_dim=4, maxiter=1000))
