# -*- coding: utf-8 -*-

import re
import sys
import csv
import copy
import numpy as np
import itertools
from collections import Counter

try:
    import sklearn
except ImportError:
    sys.stderr.write("scikit-learn version 0.16.* is required\n")
    sys.exit(2)
if sklearn.__version__[:4] != '0.16':
    sys.stderr.write("scikit-learn version 0.16.* is required. You're at %s.\n" % sklearn.__version__)
    sys.exit(2)

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

from distributedwordreps import build, ShallowNeuralNetwork 







LABELS = ['ENTAILMENT', 'CONTRADICTION', 'NEUTRAL']





WORD_RE = re.compile(r"([^ \(\)]+)", re.UNICODE)

def str2tree(s):
    """Turns labeled bracketing s into a tree structure (tuple of tuples)"""
    s = WORD_RE.sub(r'"\1",', s)
    s = s.replace(")", "),").strip(",")
    s = s.strip(",")
    return eval(s)




if __name__ == '__main__': # Prevent this example from loading on import of this module.
    
    print str2tree("( ( A child ) ( is ( playing ( in ( a yard ) ) ) ) )")




def leaves(t):
    """Returns all of the words (terminal nodes) in tree t"""
    words = []
    for x in t:
        if isinstance(x, str):
            words.append(x)
        else:
            words += leaves(x)
    return words




if __name__ == '__main__': # Prevent this example from loading on import of this module.
    
    t = str2tree("( ( A child ) ( is ( playing ( in ( a yard ) ) ) ) )")
    print leaves(t)





data_dir = 'nli-data/'

def sick_reader(src_filename):
    for example in csv.reader(file(src_filename), delimiter="\t"):
        label, t1, t2 = example[:3]
        if not label.startswith('%'): # Some files use leading % for comments.           
            yield (label, str2tree(t1), str2tree(t2))




def sick_train_reader():
    return sick_reader(src_filename=data_dir+"SICK_train_parsed.txt")

def sick_dev_reader():
    return sick_reader(src_filename=data_dir+"SICK_trial_parsed.txt")




def sick_test_reader():
    return sick_reader(src_filename=data_dir+"SICK_test_parsed.txt")






def word_overlap_features(t1, t2):
    overlap = [w1 for w1 in leaves(t1) if w1 in leaves(t2)]
    return Counter(overlap)




def word_cross_product_features(t1, t2):
    return Counter([(w1, w2) for w1, w2 in itertools.product(leaves(t1), leaves(t2))])






def featurizer(reader=sick_train_reader, feature_function=word_overlap_features):
    """Map the data in reader to a list of features according to feature_function,
    and create the gold label vector."""
    feats = []
    labels = []
    split_index = None
    for label, t1, t2 in reader():
        d = feature_function(t1, t2)
        feats.append(d)
        labels.append(label)              
    return (feats, labels)




def train_classifier(
        reader=sick_train_reader,
        feature_function=word_overlap_features,
        feature_selector=SelectFpr(chi2, alpha=0.05), # Use None to stop feature selection
        cv=10, # Number of folds used in cross-validation
        priorlims=np.arange(.1, 3.1, .1)): # regularization priors to explore (we expect something around 1)
    # Featurize the data:
    feats, labels = featurizer(reader=reader, feature_function=feature_function) 
    
    # Map the count dictionaries to a sparse feature matrix:
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(feats)

    ##### FEATURE SELECTION    
    # (An optional step; not always productive). By default, we select all
    # the features that pass the chi2 test of association with the
    # class labels at p < 0.05. sklearn.feature_selection has other
    # methods that are worth trying. I've seen particularly good results
    # with the model-based methods, which require some changes to the
    # current code.
    feat_matrix = None
    if feature_selector:
        feat_matrix = feature_selector.fit_transform(X, labels)
    else:
        feat_matrix = X
    
    ##### HYPER-PARAMETER SEARCH
    # Define the basic model to use for parameter search:
    searchmod = LogisticRegression(fit_intercept=True, intercept_scaling=1)
    # Parameters to grid-search over:
    parameters = {'C':priorlims, 'penalty':['l1','l2']}  
    # Cross-validation grid search to find the best hyper-parameters:     
    clf = GridSearchCV(searchmod, parameters, cv=cv)
    clf.fit(feat_matrix, labels)
    params = clf.best_params_

    # Establish the model we want using the parameters obtained from the search:
    mod = LogisticRegression(fit_intercept=True, intercept_scaling=1, C=params['C'], penalty=params['penalty'])

    ##### ASSESSMENT              
    # Cross-validation of our favored model; for other summaries, use different
    # values for scoring: http://scikit-learn.org/dev/modules/model_evaluation.html
    scores = cross_val_score(mod, feat_matrix, labels, cv=cv, scoring="f1_macro")       
    print 'Best model', mod
    print '%s features selected out of %s total' % (feat_matrix.shape[1], X.shape[1])
    print 'F1 mean: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2)

    # TRAIN OUR MODEL:
    mod.fit(feat_matrix, labels)

    # Return the trained model along with the objects we need to
    # featurize test data in a way that aligns with our training
    # matrix:
    return (mod, vectorizer, feature_selector, feature_function)




if __name__ == '__main__': # Prevent this example from loading on import of this module.
    
    overlapmodel = train_classifier(feature_function=word_overlap_features)




def evaluate_trained_classifier(model=None, reader=sick_dev_reader):
    """Evaluate model, the output of train_classifier, on the data in reader."""
    mod, vectorizer, feature_selector, feature_function = model
    feats, labels = featurizer(reader=reader, feature_function=feature_function)
    feat_matrix = vectorizer.transform(feats)
    if feature_selector:
        feat_matrix = feature_selector.transform(feat_matrix)
    predictions = mod.predict(feat_matrix)
    return metrics.classification_report(labels, predictions)




if __name__ == '__main__': # Prevent this example from loading on import of this module.
    
    for readername, reader in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
        print "======================================================================"
        print readername
        print evaluate_trained_classifier(model=overlapmodel, reader=reader)




if __name__ == '__main__': # Prevent this example from loading on import of this module.
    
    crossmodel = train_classifier(feature_function=word_cross_product_features)
    
    for readername, reader in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
        print "======================================================================"
        print readername
        print evaluate_trained_classifier(model=crossmodel, reader=reader)








GLOVE_MAT, GLOVE_VOCAB, _ = build('../distributedwordreps-data/glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)



def glvvec(w):
    """Return the GloVe vector for w."""
    i = GLOVE_VOCAB.index(w)
    return GLOVE_MAT[i]

def glove_features(t):
    """Return the mean glove vector of the leaves in tree t."""
    return np.mean([glvvec(w) for w in leaves(t) if w in GLOVE_VOCAB], axis=0)

def vec_concatenate(u, v):
    return np.concatenate((u, v)) 

def glove_featurizer(t1, t2):
    """Combined input vector based on glove_features and concatenation."""
    return vec_concatenate(glove_features(t1), glove_features(t2))





def labelvec(label):
    """Return output vectors like [1,-1,-1], where the unique 1 is the true label."""
    vec = np.repeat(-1.0, 3)
    vec[LABELS.index(label)] = 1.0
    return vec





def data_prep(reader=sick_train_reader, featurizer=glove_featurizer):    
    dataset = []
    for label, t1, t2 in reader():     
        dataset.append([featurizer(t1, t2), labelvec(label)])
    return dataset




def train_network(
        hidden_dim=100, 
        maxiter=1000, 
        reader=sick_train_reader,
        featurizer=glove_featurizer,
        display_progress=False):  
    dataset = data_prep(reader=reader, featurizer=featurizer)
    net = ShallowNeuralNetwork(input_dim=len(dataset[0][0]), hidden_dim=hidden_dim, output_dim=len(LABELS))
    net.train(dataset, maxiter=maxiter, display_progress=display_progress)
    return (net, featurizer)




def evaluate_trained_network(network=None, reader=sick_dev_reader):
    """Evaluate network, the output of train_network, on the data in reader"""
    net, featurizer = network
    dataset = data_prep(reader=reader, featurizer=featurizer)
    predictions = []
    cats = []
    for ex, cat in dataset:            
        # The raw prediction is a triple of real numbers:
        prediction = net.predict(ex)
        # Argmax dimension for the prediction; this could be done better with
        # an explicit softmax objective:
        prediction = LABELS[np.argmax(prediction)]
        predictions.append(prediction)
        # Store the gold label for the classification report:
        cats.append(LABELS[np.argmax(cat)])        
    # Report:
    print "======================================================================"
    print metrics.classification_report(cats, predictions, target_names=LABELS)




if __name__ == '__main__': # Prevent this example from loading on import of this module.

    network = train_network(hidden_dim=10, maxiter=1000, reader=sick_train_reader, featurizer=glove_featurizer)
    
    for readername, reader in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
        print "======================================================================"
        print readername
        evaluate_trained_network(network=network, reader=reader)





