#!/usr/bin/env python

# In-class CS224u bake-off, April 8, 2015
# Chris Potts

from distributedwordreps import *
import random
import pickle
import numpy as np
from collections import defaultdict
import sklearn.metrics

# Class labels:
SUBSET = 1.0    # Left word entails right, as in (hippo, mammal)
SUPERSET = -1.0 # Right word entails left, as in (mammal, hippo)

# In case you want to make use of GloVe vectors somehow ...
# It's worth checking on higher dimensionality versions too:
# http://nlp.stanford.edu/projects/glove/
#
# GLOVE_MAT, GLOVE_VOCAB, _ = build('distributedwordreps-data/glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)

def randvec(w, n=40, lower=-0.5, upper=0.5):
    """Returns a random vector of length n. w is ignored."""
    return np.array([random.uniform(lower, upper) for i in range(n)])

def vec_concatenate(u, v):
    return np.concatenate((u, v))
    
def data_prep(
        src_filename='wordentail_data.pickle',
        vector_func=None,        # Should be a map from the strings in vocab to vectors (e.g., randvec). 
        vector_combo_func=None): # Use vec_concatenate or write something better!
    # Load in the dataset:
    vocab, d = pickle.load(file(src_filename))

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Make vectors a mapping from words (as strings) to their vector
    # representations, as determined by vector_func.
    vectors = {}

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Here, we create a dataset in the format required by the neural
    # network:
    #
    # {'train': [(vec, [cls]), (vec, [cls]), ...],
    #  'test':  [(vec, [cls]), (vec, [cls]), ...],
    #  'disjoint_vocab_test': [(vec, [cls]), (vec, [cls]), ...]}    
    dataset = defaultdict(list)
    for split, data in d.items():
        for clsname, word_pairs in data.items():
            for w1, w2 in word_pairs:
                          
                #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                # Use vector_combo_func to combine the word vectors
                # for w1 and w2, as given by the vectors dictionary
                # above, and pair it with the singleton array containing
                # clsname. item should be a pair consisting of a single
                # vector and a list containing only clsname:
                item = [] 
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                
                dataset[split].append(item)
    return dataset
    
def train_and_evaluate(dataset):
    train = dataset['train']
    test = dataset['test']
    disjoint_vocab_test = dataset['disjoint_vocab_test']

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Set up the neural network so that input_dim is the length of
    # your training inputs, hidden_dim is set by you (make it a
    # keyword argument to this function, and output_dim is 1:

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Train the network, with the number of iterations set you by you
    # (make it a keyword argument to this function). You might want
    # to use display_progress=True to track errors and speed.
    # USE ONLY train FOR THE TRAINING!!!

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # The following is evaluation code. You won't have to alter it
    # unless you did something unanticipated like transform the output
    # variables before training.
    for typ, data in (('train', train), ('test', test), ('disjoint_vocab_test', disjoint_vocab_test)):
        predictions = []
        cats = []
        for ex, cat in data:            
            # The raw prediction is a singleton list containing a float in (-1,1).
            # We want only its contents:
            prediction = net.predict(ex)[0]
            # Categorize the prediction for accuracy comparison:
            prediction = SUPERSET if prediction <= 0.0 else SUBSET
            predictions.append(prediction)
            # Store the gold label for the classification report:
            cats.append(cat[0])
        # Report:
        print "======================================================================"
        print typ
        print sklearn.metrics.classification_report(cats, predictions, target_names=['SUPERSET', 'SUBSET'])

if __name__ == '__main__':

    # This is a complete run. You'll probably want to make keyword
    # arguments available here to tune good networks heuristically.
    dataset = data_prep()
    train_and_evaluate(dataset)


