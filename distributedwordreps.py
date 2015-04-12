#!/usr/bin/env python
# -*- coding: utf-8 -*-

# For CS224u, Stanford, Spring 2015 (Chris Potts)
# Exported from distributedwordreps.ipynb, which can
# also be viewed in HTML: distributedwordreps.html

######################################################################

import os
import sys
import csv
import copy
import random
import itertools
from operator import itemgetter
from collections import defaultdict
# Make sure you've got Numpy and Scipy installed:
import numpy as np
import scipy
import scipy.spatial.distance
from numpy.linalg import svd
# For visualization:
from tsne import tsne # See http://lvdmaaten.github.io/tsne/#implementations
import matplotlib.pyplot as plt
# For clustering in the 'Word-sense ambiguities' section:
from sklearn.cluster import AffinityPropagation

######################################################################
# Reading in matrices

def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):    
    reader = csv.reader(file(src_filename), delimiter=delimiter, quoting=quoting)
    colnames = None
    if header:
        colnames = reader.next()
        colnames = colnames[1: ]
    mat = []    
    rownames = []
    for line in reader:        
        rownames.append(line[0])            
        mat.append(np.array(map(float, line[1: ])))
    return (np.array(mat), rownames, colnames)

######################################################################
# Vector comparison

def euclidean(u, v):
    # Use scipy's method:
    return scipy.spatial.distance.euclidean(u, v)
    # Or define it yourself:
    # return vector_length(u - v)    

def vector_length(u):
    return np.sqrt(np.dot(u, u))

def length_norm(u):
    return u / vector_length(u)

def cosine(u, v):
    # Use scipy's method:
    return scipy.spatial.distance.cosine(u, v)
    # Or define it yourself:
    # return 1.0 - (np.dot(u, v) / (vector_length(u) * vector_length(v)))

def matching(u, v):
    # The scipy implementation is for binary vectors only. This version is more general.
    return np.sum(np.minimum(u, v))

def jaccard(u, v):
    # The scipy implementation is for binary vectors only. This version is more general.
    return 1.0 - (matching(u, v) / np.sum(np.maximum(u, v)))

def neighbors(word=None, mat=None, rownames=None, distfunc=cosine):
    if word not in rownames:
        raise ValueError('%s is not in this VSM' % word)
    w = mat[rownames.index(word)]
    dists = [(rownames[i], distfunc(w, mat[i])) for i in xrange(len(mat))]
    return sorted(dists, key=itemgetter(1), reverse=False)

######################################################################
# Reweighting

def prob_norm(u):
    return u / np.sum(u)

def pmi(mat=None, rownames=None, positive=True):
    """PMI on mat; positive=True does PPMI. rownames is not used; it's 
    an argument only for consistency with other methods used here"""
    # Joint probability table:
    p = mat / np.sum(mat, axis=None)
    # Pre-compute column sums:
    colprobs = np.sum(p, axis=0)
    # Vectorize this function so that it can be applied rowwise:
    np_pmi_log = np.vectorize((lambda x : _pmi_log(x, positive=positive)))
    p = np.array([np_pmi_log(row / (np.sum(row)*colprobs)) for row in p])   
    return (p, rownames)

def _pmi_log(x, positive=True):
    """With positive=False, return log(x) if possible, else 0.
    With positive=True, log(x) is mapped to 0 where negative."""
    val = 0.0
    if x > 0.0:
        val = np.log(x)
    if positive:
        val = max([val,0.0])
    return val

def tfidf(mat=None, rownames=None):
    """TF-IDF on mat. rownames is unused; it's an argument only 
    for consistency with other methods used here"""
    colsums = np.sum(mat, axis=0)
    doccount = mat.shape[1]
    w = np.array([_tfidf_row_func(row, colsums, doccount) for row in mat])
    return (w, rownames)

def _tfidf_row_func(row, colsums, doccount):
    df = float(len([x for x in row if x > 0]))
    idf = 0.0
    # This ensures a defined IDF value >= 0.0:
    if df > 0.0 and df != doccount:
        idf = np.log(doccount / df)
    tfs = row/colsums
    return tfs * idf

######################################################################
# Dimensionality reduction

def lsa(mat=None, rownames=None, k=100):
    """svd with a column-wise truncation to k dimensions; rownames 
    is passed through only for consistency with other methods"""
    rowmat, singvals, colmat = svd(mat, full_matrices=False)
    singvals = np.diag(singvals)
    trunc = np.dot(rowmat[:, 0:k], singvals[0:k, 0:k])
    return (trunc, rownames)

######################################################################
# Visualization

def tsne_viz(
        mat=None,
        rownames=None,
        indices=None,
        colors=None,
        output_filename=None,
        figheight=40,
        figwidth=50,
        display_progress=False): 
    """2d plot of mat using tsne, with the points labeled by rownames,
    aligned with colors (defaults to all black).
    If indices is a list of indices into mat and rownames,
    then it determines a subspace of mat and rownames to display.
    Give output_filename a string argument to save the image to disk.
    figheight and figwidth set the figure dimensions.
    display_progress=True shows the information that the tsne method prints out."""
    if not colors:
        colors = ['black' for i in range(len(rownames))]
    temp = sys.stdout
    if not display_progress:
        # Redirect stdout so that tsne doesn't fill the screen with its iteration info:
        f = open(os.devnull, 'w')
        sys.stdout = f
    tsnemat = tsne(mat)
    sys.stdout = temp
    # Plot coordinates:
    if not indices:
        indices = range(len(rownames))        
    vocab = np.array(rownames)[indices]
    xvals = tsnemat[indices, 0] 
    yvals = tsnemat[indices, 1]
    # Plotting:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(40)
    fig.set_figwidth(50)
    ax.plot(xvals, yvals, marker='', linestyle='')
    # Text labels:
    for word, x, y, color in zip(vocab, xvals, yvals, colors):
        ax.annotate(word, (x, y), fontsize=8, color=color)
    # Output:
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
    else:
        plt.show()

######################################################################
# Semantic orientation method
        
def semantic_orientation(
        mat=None, 
        rownames=None,
        seeds1=['bad', 'nasty', 'poor', 'negative', 'unfortunate', 'wrong', 'inferior'],
        seeds2=['good', 'nice', 'excellent', 'positive', 'fortunate', 'correct', 'superior'],
        distfunc=cosine):
    sm1 = so_seed_matrix(seeds1, mat, rownames)
    sm2 = so_seed_matrix(seeds2, mat, rownames)
    scores = [(rownames[i], so_row_func(mat[i], sm1, sm2, distfunc)) for i in xrange(len(mat))]
    return sorted(scores, key=itemgetter(1), reverse=False)

def so_seed_matrix(seeds, mat, rownames):
    indices = [rownames.index(word) for word in seeds if word in rownames]
    if not indices:
        raise ValueError('The matrix contains no members of the seed set: %s' % ",".join(seeds))
    return mat[np.array(indices)]
    
def so_row_func(row, sm1, sm2, distfunc):
    val1 = np.sum([distfunc(row, srow) for srow in sm1])
    val2 = np.sum([distfunc(row, srow) for srow in sm2])
    return val1 - val2    

######################################################################
# Disambiguation

def disambiguate(mat=None, rownames=None, minval=0.0):
    """Basic unsupervised disambiguation. minval sets what it means to occur in a column"""
    clustered = defaultdict(lambda : defaultdict(int))
    # For each word, cluster the documents containing it:
    for w_index, w in enumerate(rownames):
        doc_indices = np.array([j for j in range(mat.shape[1]) if mat[w_index,j] > minval])
        clust = cluster(mat, doc_indices) 
        for doc_index, c_index in clust:
            w_sense = "%s_%s" % (w, c_index)
            clustered[w_sense][doc_index] = mat[w_index, doc_index]
    # Build the new matrix:
    new_rownames = sorted(clustered.keys())
    new_mat = np.zeros((len(new_rownames), mat.shape[1]))
    for i, w in enumerate(new_rownames):
        for j in clustered[w]:            
            new_mat[i,j] = clustered[w][j]
    return (new_mat, new_rownames)

def cluster(mat, doc_indices):    
    X = mat[:, doc_indices].T
    # Other clustering algorithms can easily be swapped in: 
    # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
    clust = AffinityPropagation()
    clust.fit(X)    
    return zip(doc_indices,  clust.labels_)     

######################################################################
# GloVe word representations

def randmatrix(m, n, lower=-0.5, upper=0.5):
    """Creates an m x n matrix of random values in [lower, upper]"""
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)

def glove(
        mat=None, rownames=None, 
        n=100, xmax=100, alpha=0.75, 
        iterations=100, learning_rate=0.05, 
        display_progress=False):
    """Basic GloVe. rownames is passed through unused for compatibility
    with other methods. n sets the dimensionality of the output vectors.
    xmax and alpha controls the weighting function (see the paper, eq. (9)).
    iterations and learning_rate control the SGD training.
    display_progress=True prints iterations and current error to stdout."""    
    m = mat.shape[0]
    W = randmatrix(m, n) # Word weights.
    C = randmatrix(m, n) # Context weights.
    B = randmatrix(2, m) # Word and context biases.
    indices = range(m)
    for iteration in range(iterations):
        error = 0.0        
        random.shuffle(indices)
        for i, j in itertools.product(indices, indices):
            if mat[i,j] > 0.0:     
                # Weighting function from eq. (9)
                weight = (mat[i,j] / xmax)**alpha if mat[i,j] < xmax else 1.0
                # Cost is J' based on eq. (8) in the paper:
                diff = np.dot(W[i], C[j]) + B[0,i] + B[1,j] - np.log(mat[i,j])                
                fdiff = diff * weight                
                # Gradients:
                wgrad = fdiff * C[j]
                cgrad = fdiff * W[i]
                wbgrad = fdiff
                wcgrad = fdiff
                # Updates:
                W[i] -= (learning_rate * wgrad) 
                C[j] -= (learning_rate * cgrad) 
                B[0,i] -= (learning_rate * wbgrad) 
                B[1,j] -= (learning_rate * wcgrad)                 
                # One-half squared error term:                              
                error += 0.5 * weight * (diff**2)
        if display_progress:
            print "iteration %s: error %s" % (iteration, error)
    # Return the sum of the word and context matrices, per the advice 
    # in section 4.2:
    return (W + C, rownames)

def glove_viz(mat=None, rownames=None, word_count=1000, iterations=10, n=50, display_progress=True):
    glove_indices = random.sample(range(len(rownames)), word_count)
    glovemat, _ = glove(mat=mat[glove_indices, :], iterations=iterations, n=n)
    tsne_viz(mat=glovemat, rownames=np.array(rownames)[glove_indices])

######################################################################
# Shallow neural networks
    
from numpy import dot, outer

class ShallowNeuralNetwork:
    def __init__(self, input_dim=0, hidden_dim=0, output_dim=0, afunc=np.tanh, d_afunc=(lambda z : 1.0 - z**2)):        
        self.afunc = afunc 
        self.d_afunc = d_afunc      
        self.input = np.ones(input_dim+1)   # +1 for the bias                                         
        self.hidden = np.ones(hidden_dim+1) # +1 for the bias        
        self.output = np.ones(output_dim)        
        self.iweights = randmatrix(input_dim+1, hidden_dim)
        self.oweights = randmatrix(hidden_dim+1, output_dim)        
        self.oerr = np.zeros(output_dim)
        self.ierr = np.zeros(input_dim+1)
        
    def forward_propagation(self, ex):        
        self.input[ : -1] = ex # ignore the bias
        self.hidden[ : -1] = self.afunc(dot(self.input, self.iweights)) # ignore the bias
        self.output = self.afunc(dot(self.hidden, self.oweights))
        return copy.deepcopy(self.output)
        
    def backward_propagation(self, labels, alpha=0.5):
        labels = np.array(labels)       
        self.oerr = (labels-self.output) * self.d_afunc(self.output)
        herr = dot(self.oerr, self.oweights.T) * self.d_afunc(self.hidden)
        self.oweights += alpha * outer(self.hidden, self.oerr)
        self.iweights += alpha * outer(self.input, herr[:-1]) # ignore the bias
        return np.sum(0.5 * (labels-self.output)**2)

    def train(self, training_data, maxiter=5000, alpha=0.05, epsilon=1.5e-8, display_progress=False):       
        iteration = 0
        error = sys.float_info.max
        while error > epsilon and iteration < maxiter:            
            error = 0.0
            random.shuffle(training_data)
            for ex, labels in training_data:
                self.forward_propagation(ex)
                error += self.backward_propagation(labels, alpha=alpha)           
            if display_progress:
                print 'completed iteration %s; error is %s' % (iteration, error)
            iteration += 1
                    
    def predict(self, ex):
        self.forward_propagation(ex)
        return copy.deepcopy(self.output)
        
    def hidden_representation(self, ex):
        self.forward_propagation(ex)
        return self.hidden

def read_valence_arousal_dominance_lexicon(src_filename='distributedwordreps-data/Warriner_et_al emot ratings.csv'):
    rescaler = (lambda x : np.tanh(float(x)-5))
    lex = {}
    for d in csv.DictReader(file(src_filename)):
        vals = {'valence': rescaler(d['V.Mean.Sum']), 
                'arousal': rescaler(d['A.Mean.Sum']), 
                'dominance': rescaler(d['D.Mean.Sum'])}
        lex[d['Word']] = vals
    return lex

def build_supervised_dataset(mat=None, rownames=None, lex=None):
    data = []
    vocab = []
    for word, vals in lex.items():
        if word in rownames:
            vocab.append(word)
            data.append((mat[rownames.index(word)], [y for _, y in sorted(vals.items())]))
    return (data, vocab)

def sentiment_lexicon_example(
        mat=None, 
        rownames=None, 
        hidden_dim=100, 
        maxiter=1000, 
        output_filename=None, 
        display_progress=False):
    # Get the lexicon:
    lex = read_valence_arousal_dominance_lexicon()
    # Build the training data:
    sentidata, sentivocab = build_supervised_dataset(mat=mat, rownames=rownames, lex=lex)
    # Set up the network:
    sentinet = ShallowNeuralNetwork(input_dim=len(sentidata[0][0]), hidden_dim=hidden_dim, output_dim=len(sentidata[0][1]))
    # Train the network:
    sentinet.train(copy.deepcopy(sentidata), maxiter=maxiter, display_progress=display_progress)
    # Build the new matrix of hidden representations:
    inputs, labels = zip(*sentidata)
    sentihidden = np.array([sentinet.hidden_representation(x) for x in inputs])
    # Visualize the results with t-SNE:
    def colormap(vals):
        """Simple way to distinguish the 2x2x2 possible labels -- could be done much better!"""
        signs = ['CC' if x < 0.0 else '00' for _, x in sorted(vals.items())]
        return "#" + "".join(signs)    
    colors = [colormap(lex[word]) for word in sentivocab]
    tsne_viz(mat=sentihidden, rownames=sentivocab, colors=colors, display_progress=display_progress, output_filename=output_filename)

######################################################################
# Word similarity task   

def word_similarity_evaluation(src_filename="distributedwordreps-data/wordsim353/combined.csv", 
        mat=None, rownames=None, distfunc=cosine):
    # Read in the data:
    reader = csv.DictReader(file(src_filename))
    sims = defaultdict(list)
    vocab = set([])
    for d in reader:
        w1 = d['Word 1']
        w2 = d['Word 2']
        if w1 in rownames and w2 in rownames:
            # Use negative of scores to align intuitively with distance functions:
            sims[w1].append((w2, -float(d['Human (mean)'])))
            sims[w2].append((w1, -float(d['Human (mean)'])))
            vocab.add(w1)
            vocab.add(w2)
    # Evaluate the matrix by creating a vector of all_scores for the wordsim353 data
    # and all_dists for mat's distances. 
    all_scores = []
    all_dists = []
    for word in vocab:
        vec = mat[rownames.index(word)]
        vals = sims[word]
        cmps, scores = zip(*vals)
        all_scores += scores
        all_dists += [distfunc(vec, mat[rownames.index(w)]) for w in cmps]
    # Return just the rank correlation coefficient (index [1] would be the p-value):
    return scipy.stats.spearmanr(all_scores, all_dists)[0]   

######################################################################
# Analogy completion task

def analogy_completion(a, b, c, mat=None, rownames=None, distfunc=cosine):
    """a is to be as c is to predicted, where predicted is the closest to (b-a) + c"""
    for x in (a, b, c):
        if x not in rownames:
            raise ValueError('%s is not in this VSM' % x)
    avec = mat[rownames.index(a)]
    bvec = mat[rownames.index(b)]
    cvec = mat[rownames.index(c)]
    newvec = (bvec - avec) + cvec
    dists = [(w, distfunc(newvec, mat[i])) for i, w in enumerate(rownames) if w not in (a, b, c)]
    return sorted(dists, key=itemgetter(1), reverse=False)    

def analogy_evaluation(src_filename="distributedwordreps-data/question-data/gram1-adjective-to-adverb.txt", 
        mat=None, rownames=None, distfunc=cosine):
    # Read in the data and restrict to problems we can solve:
    data = [line.split() for line in open(src_filename).read().splitlines()]
    data = [prob for prob in data if set(prob) <= set(rownames)]
    # Run the evaluation, collecting accuracy and rankings:
    results = defaultdict(int)
    ranks = []
    for a, b, c, d in data:
        predicted = analogy_completion(a, b, c, mat=mat, rownames=rownames, distfunc=distfunc)
        # print "%s is to %s as %s is to %s (actual is %s)" % (a, b, c, predicted, d)
        results[predicted[0][0] == d] += 1
        predicted_words, _ = zip(*predicted)
        ranks.append(predicted_words.index(d))
    # Return the mean reciprocal rank and the accuracy results:
    mrr = np.mean(1.0/(np.array(ranks)+1))
    return (mrr, results)

