
# coding: utf-8

# In[1]:




# # Natural language inference

# In[2]:

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
    sys.stderr.write("scikit-learn version 0.16.* is required. You have version %s.\n" % sklearn.__version__[:4])
    sys.exit(2)

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

from distributedwordreps import build, ShallowNeuralNetwork 


# In the context of NLP/NLU, Natural Language Inference (NLI) is the task of predicting the logical
# relationships between words, phrases, sentences, (paragraphs, documents, ...). Such relationships
# are crucial for all kinds of reasoning in natural language: arguing, debating, problem solving, 
# summarization, extrapolation, and so forth. 
# 
# NLI is a great task for this course. It requires serious linguistic analysis to do well, 
# there are good (albeit small) publicly available data sets, and there are some natural baselines
# that help with getting a model up and running, and with understanding the performance of more 
# sophisticated approaches. 
# 
# NLI was also the topic of [Bill's thesis](http://nlp.stanford.edu/~wcmac/papers/nli-diss.pdf) 
# (he popularized the name "NLI"), so you can forever endear yourself to him by working on it!
# 
# The purpose of this codebook is to introduce the problem of NLI more fully in the context of 
# the [SemEval 2014 semantic relatedness task](http://alt.qcri.org/semeval2014/task1/). 
# This data set is called "Sentences Involving Compositional Knowledge" or, for better or worse, 
# "SICK". It's [freely available from the SemEval site](http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools), 
# but we're going to work with a parsed version created by [Sam Bowman](http://stanford.edu/~sbowman/) 
# as part of [his research on neural models of semantic composition](https://github.com/sleepinyourhat/vector-entailment/releases/tag/W15-R1). 
# This data is in the subfolder `nli-data` of this Github repository.
# 
# This codebook explores two general approaches: a standard classifier, and 
# a shallow neural network using distributed representations.
# 
# The [Classifier training and assessment](#Classifier-training-and-assessment) section also 
# serves as a general illustration of how to take advantage of the 
# [scikit-learn](http://scikit-learn.org/stable/) functions for 
# doing feature selection, cross-validation, hyper-parameter optimization, and evaluation in the 
# context of multi-class classification. That code could be easily modified to work with 
# any classification problem.

# 0. [Working with the parsed SICK data](#Working-with-the-parsed-SICK-data)
#    0. [Trees](#Trees)
#    0. [Readers](#Readers)
# 0. [MaxEnt classifier approach](#MaxEnt-classifier-approach)
#    0. [Baseline classifier features](#Baseline-classifier-features)
#    0. [Classifier training and assessment](#Classifier-training-and-assessment)
# 0. [Shallow neural network approach](#Shallow-neural-network-approach)
#    0. [Baseline distributed features](#Baseline-distributed-features)                      
#    0. [Output label vectors](#Output-label-vectors) 
#    0. [Network training and assessment](#Network-training-and-assessment)
# 0. [Some further reading](#Some-further-reading)

# ## Working with the parsed SICK data

# Our parsed version of the SICK data contains triples like this (tab-separated fields):
#     
# `
# ENTAILMENT  ( ( A child ) ( is ( playing ( in ( a yard ) ) ) ) )  ( ( A child ) ( is playing ) )
# `
# 
# `
# NEUTRAL  ( ( A child ) ( is ( playing ( in ( a yard ) ) ) ) )  ( ( A child ) ( is ( wearing ( blue jeans ) ) ) )
# `
# 
# `
# CONTRADICTION  ( ( A child ) ( is ( playing ) ) )  ( ( A child ) ( is sleeping ) )
# `
# 
# The brackets encode a label-free constituency structure of each sentence. The three labels on the left 
# are the classes that we want to learn to predict. We'll frequently need access to them, so let's define
# them as a list:

# In[3]:

LABELS = ['ENTAILMENT', 'CONTRADICTION', 'NEUTRAL']


# ### Trees

# The baseline models that I define here ignore all of the tree structure, but you'll likely want 
# to take advantage of it. So the following function can be used to turn bracketed strings like
# the above into tuples of tuples encoding the syntactic structure.

# In[4]:

WORD_RE = re.compile(r"([^ \(\)]+)", re.UNICODE)

def str2tree(s):
    """Turns labeled bracketing s into a tree structure (tuple of tuples)"""
    s = WORD_RE.sub(r'"\1",', s)
    s = s.replace(")", "),").strip(",")
    s = s.strip(",")
    return eval(s)


# Here's an example:

# In[5]:

if __name__ == '__main__': # Prevent this example from loading on import of this module.
    
    print str2tree("( ( A child ) ( is ( playing ( in ( a yard ) ) ) ) )")


# For baseline models, we often want just the words, also called terminal nodes or _leaves_.
# This function gives us access to them as a list:

# In[6]:

def leaves(t):
    """Returns all of the words (terminal nodes) in tree t"""
    words = []
    for x in t:
        if isinstance(x, str):
            words.append(x)
        else:
            words += leaves(x)
    return words


# An example:

# In[7]:

if __name__ == '__main__': # Prevent this example from loading on import of this module.
    
    t = str2tree("( ( A child ) ( is ( playing ( in ( a yard ) ) ) ) )")
    print leaves(t)


# ### Readers

# To make it easy to run through the corpus, let's define general readers for the SICK
# data. The general function for this yields triples consisting of the label, the left
# tree, and the right tree, as parsed by `str2tree`:

# In[8]:

data_dir = 'nli-data/'

def sick_reader(src_filename):
    for example in csv.reader(file(src_filename), delimiter="\t"):
        label, t1, t2 = example[:3]
        if not label.startswith('%'): # Some files use leading % for comments.           
            yield (label, str2tree(t1), str2tree(t2))


# Now we define separate readers for the training and development data:

# In[9]:

def sick_train_reader():
    return sick_reader(src_filename=data_dir+"SICK_train_parsed.txt")

def sick_dev_reader():
    return sick_reader(src_filename=data_dir+"SICK_trial_parsed.txt")


# Eventually, we'll want a test-set reader. As Bill discussed in class, though, 
# __we swear on our honor as scholars that we won't use this data until
# system development is complete and we are ready to conduct our final
# assessment__!

# In[10]:

def sick_test_reader():
    return sick_reader(src_filename=data_dir+"SICK_test_parsed.txt")


# ## MaxEnt classifier approach

# ### Baseline classifier features

# The first baseline we define is the _word overlap_ baseline. It simply uses as
# features the words that appear in both sentences.

# In[11]:

def word_overlap_features(t1, t2):
    overlap = [w1 for w1 in leaves(t1) if w1 in leaves(t2)]
    return Counter(overlap)


# Another popular baseline is to use as features the full cross-product of
# words from both sentences:    

# In[12]:

def word_cross_product_features(t1, t2):
    return Counter([(w1, w2) for w1, w2 in itertools.product(leaves(t1), leaves(t2))])


# Both of these feature functions return count dictionaries mapping feature names to 
# the number of times they occur in the data. This is the representation we'll work
# with throughout; scikit-learn will handle the further processing it needs to build
# linear classifiers.
# 
# Naturally, you can do better than these feature functions! 
# Both of these feature classes might be useful even in a more advanced model, though.

# ### Classifier training and assessment

# The first step in training a classifier is using a feature function like the one above
# to turn the data into a list of _training instances_: feature representations and their 
# associated labels:

# In[13]:

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


# The main work of training a model is done by the following `train_classifier` method. At a high-level, 
# this function is doing a few things, with the goal of finding the right model (in the
# class of models we're exploring) for the data we have to train on. The major steps:                                                                        
# 
# 0. Use scikit-learn to turn the feature dictionaries into a big sparse matrix.
# 0. Optionally use a bit of statistical testing to select features that seem discriminating. 
# 0. Use cross-validation within the training data to try to find the right regularization scheme.
# 0. Train using the best-looking regularization scheme found in the previous step.
# 0. Report cross-valided F1 scores to provide a sense for how we're doing.
# 0. Return the fitted model plus the objects we need to align with future evaluation data.
# 
# Part of the thinking behind this approach is that one can work as follows:
# 
# * By and large, you check models just by training them and seeing how they do in cross-validation.
# * Only ccasionally, and only with good reason, do you check how you're doing on the dev set. If this is done only rarely, then it will help prevent you from over-fitting to quirks of this data or the training data.
# * At the very end, one runs the test. 
# * The final paper can report, for the final model,
#   * Mean cross-validation values on the training data with standard errors
#   * Dev set performance
#   * Test set performance

# In[14]:

def train_classifier(
        reader=sick_train_reader,
        feature_function=word_overlap_features,
        feature_selector=SelectFpr(chi2, alpha=0.05), # Use None to stop feature selection
        cv=10, # Number of folds used in cross-validation
        priorlims=range(1,10)): # regularization priors to explore (we expect something around 1)
    # Featurize the data:
    feats, labels = featurizer(reader=reader, feature_function=feature_function) 
    
    # Map the count dictionaries to a sparse feature matrix:
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(feats)

    ##### FEATURE SELECTION    
    # (An optional step; not always productive). By default, we select all
    # the features that pass the chi2 test of association with the
    # class labels at p < 0.05. sklearn.feature_selection has other
    # methods that are worth trying.
    feat_matrix = None
    if feature_selector:
        feat_matrix = feature_selector.fit_transform(X, labels)
    else:
        feat_matrix = X
    
    ##### HYPER-PARAMETER SEARCH
    # Define the basic model to use for parameter search:
    searchmod = LogisticRegression(fit_intercept=True, intercept_scaling=1)
    # Parameters to grid-search over:
    parameters = {'C':[x / 10.0 for x in priorlims], 'penalty':['l1','l2']}  
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


# Now we can train either of our baseline models. Here's the run for the word-overlap one 
# (`word_cross_product_features` does better, since it has more information, but it takes
# a while to train and goes from 89,038 candidate features to 3,680 during feature
# selection).

# In[15]:

if __name__ == '__main__': # Prevent this example from loading on import of this module.
    
    model = train_classifier(feature_function=word_overlap_features)


# Finally, the following code assess the output of `train_classifier` on new data. The default is to do this
# on the dev set, but this same code would be used to evaluate the final model on the test set.

# In[16]:

def evaluate_trained_classifier(model=None, reader=sick_dev_reader):
    """Evaluate model, the output of train_classifier, on the data in reader."""
    mod, vectorizer, feature_selector, feature_function = model
    feats, labels = featurizer(reader=reader, feature_function=feature_function)
    feat_matrix = vectorizer.transform(feats)
    if feature_selector:
        feat_matrix = feature_selector.transform(feat_matrix)
    predictions = mod.predict(feat_matrix)
    return metrics.classification_report(labels, predictions)


# Let's see how we do on the training data as well as on the development data:

# In[17]:

if __name__ == '__main__': # Prevent this example from loading on import of this module.
    
    for readername, reader in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
        print "======================================================================"
        print readername
        print evaluate_trained_classifier(model=model, reader=reader)


# ## Shallow neural network approach

# ### Baseline distributed features

# The baseline I define here just turns each sentence into the average of all its word vectors.
# To create inputs to the network, we just concatenate the output of `glove_featurizer` for the
# two trees being compared. Pretty simplistic, but it is at least a start!

# In[18]:

GLOVE_MAT, GLOVE_VOCAB, _ = build('../distributedwordreps-data/glove.6B.50d.txt', delimiter=' ', header=False, quoting=csv.QUOTE_NONE)


# In[19]:

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


# ### Output label vectors

# To shoehorn the current problem into our current neural network implementation, I 
# define our output vectors as having a 1 for the dimension of the correct class, and a -1 
# for the other two classes:

# In[20]:

def labelvec(label):
    """Return output vectors like [1,-1,-1], where the unique 1 is the true label."""
    vec = np.repeat(-1.0, 3)
    vec[LABELS.index(label)] = 1.0
    return vec


# ### Network training and assessment

# To train the network, we first create the training data by pairing
# our sentence-pair vectors with the output vector:

# In[21]:

def data_prep(reader=sick_train_reader, featurizer=glove_featurizer):    
    dataset = []
    for label, t1, t2 in reader():     
        dataset.append([featurizer(t1, t2), labelvec(label)])
    return dataset


# Training is then straightforward: it just uses `ShallowNeuralNetwork`:

# In[22]:

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


# The evaluation function evaluates the output of `train_network` using a new data reader.
# The `featurizer` is return by `train_network` to ensure that it is used consistently in
# training and evalution.

# In[23]:

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


# Here's a summary train/dev evaluation. The network has trouble dealing with the
# class imbalances, but it seems to be doing okay overall.

# In[24]:

if __name__ == '__main__': # Prevent this example from loading on import of this module.

    network = train_network(hidden_dim=20, maxiter=1000, reader=sick_train_reader, featurizer=glove_featurizer)
    
    for readername, reader in (('Train', sick_train_reader), ('Dev', sick_dev_reader)):
        print "======================================================================"
        print readername
        evaluate_trained_network(network=network, reader=reader)


# ## Some further reading

# Bowman, Samuel R.; Christopher Potts; and Christopher D. Manning. 2014. 
# [Recursive neural networks for learning logical semantics](http://arxiv.org/abs/1406.1827). 
# arXiv manuscript 1406.1827. 
# 
# Dagan, Ido; Oren Glickman; and  Bernardo Magnini. 2006. 
# [The PASCAL recognising textual entailment challenge](http://eprints.pascal-network.org/archive/00001298/01/dagan_et_al_rte05.pdf).
# In J. Quinonero-Candela, I. Dagan, B. Magnini, F. d'Alché-Buc, ed., _Machine Learning Challenges_, 
# 177-190. Springer-Verlag.
# 
# Icard, Thomas F. 2012. [Inclusion and exclusion in natural language](http://link.springer.com/article/10.1007%2Fs11225-012-9425-8). _Studia Logica_ 100(4): 705-725.
# 
# MacCartney, Bill and Christopher D. Manning. 2009. 
# [An extended model of natural logic](http://www.aclweb.org/anthology/W09-3714). 
# In  _Proceedings of the Eighth International Conference on Computational Semantics_, 140-156. 
# Tilburg, The Netherlands: Association for Computational Linguistics.
