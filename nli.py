from collections import defaultdict
import json
from nltk.tree import Tree
import numpy as np
import os
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2022"


def str2tree(s, binarize=False):
    """Map str `s` to an `nltk.tree.Tree` instance.

    Parameters
    ----------
    s : str
    binarize : bool
        Use `binarize=True` to handle the SNLI/MultiNLI binarized
        tree format.

    Returns
    -------
    nltk.tree.Tree
    """
    if not s.startswith('('):
        s = "( {} )".format(s)
    if binarize:
        s = s.replace("(", "(X")
    return Tree.fromstring(s)


def get_pair_overlap_size(wordentail_data):
    train = {tuple(x) for x, y in wordentail_data['train']}
    dev = {tuple(x) for x, y in wordentail_data['dev']}
    return len(train & dev)


def get_vocab_overlap_size(wordentail_data):
    train = {w for x, y in wordentail_data['train'] for w in x}
    dev = {w for x, y in wordentail_data['dev'] for w in x}
    return len(train & dev)


class NLIExample(object):
    """For processing examples from SNLI or MultiNLI.

    Parameters
    ----------
    d : dict
        Derived from a dataset example. Each key-value pair becomes
        an attribute-value pair for the class. The tree strings are
        converted to `nltk.tree.Tree` instances using `str2tree`.

    """

    label_map = {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction',
        -1: '-'}

    def __init__(self, d):
        for k, v in d.items():
            if k == 'label':
                v = self.label_map[v]
            if '_binary_parse' in k:
                v = str2tree(v, binarize=True)
            elif '_parse' in k:
                v = str2tree(v, binarize=False)
            setattr(self, k, v)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"NLIExample({})""".format(d)


class NLIReader(object):
    """Reader for SNLI/MultiNLI data.

    Parameters
    ----------
    splits : DatasetDict or arg list of DatasetDict
        The NLI dataset split(s) as read by the Hugging Face function
        `datasets.load_dataset` with the split key filled in
        (e.g., "train", "validation", "test"). All the splits must have
        the same fields.
    filter_unlabeled : bool
        Whether to leave out cases without a gold label.
    samp_percentage : float or None
        If not None, randomly sample approximately this percentage
        of lines.
    random_state : int or None
        Optionally set the random seed for consistent sampling.

    Raises
    ------
    ValueError, if the splits don't have all the same fields

    """
    def __init__(self,
            *splits,
            filter_unlabeled=True,
            samp_percentage=None,
            random_state=None,
            gold_label_attr_name='gold_label'):
        self.splits = splits

        fields = set(self.splits[0].features.keys())
        for split in self.splits[1: ]:
            if set(split.features.keys()) != fields:
                raise ValueError(
                    "All provided splits must have the same keys.")

        self.filter_unlabeled = filter_unlabeled
        self.samp_percentage = samp_percentage
        self.random_state = random_state

    def read(self):
        """Primary interface.

        Yields
        ------
        `NLIExample`

        """
        random.seed(self.random_state)
        for split in self.splits:
            names = list(split.features.keys())
            fields = zip(*[split[k] for k in names])
            for ex in fields:
                if (not self.samp_percentage) or random.random() <= self.samp_percentage:
                    d = dict(zip(names, ex))
                    ex = NLIExample(d)
                    if (not self.filter_unlabeled) or ex.label != '-':
                        yield ex

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"NLIReader({})""".format(d)



def read_annotated_subset(src_filename, mnli_dev_split):
    """Given an annotation filename from MultiNLI's separate
    annotation distribution, associate it with the appropriate
    dev examples.

    Parameters
    ----------
    src_filename : str
        Full pat to the annotation file.
    mnli_dev_split : str
        The MultiNLI dataset split as read by the Hugging Face
        function `datasets.load_dataset` with the split key as
        either "validation_matched" or "validation_mismatched".

    Returns
    -------
    dict
        Maps pairID values to dicts with keys 'annotations' and
        'example', where 'annotations' gives a list of str and
        'example' gives an `NLIExample` instance.

    """
    reader = NLIReader(mnli_dev_split)
    id2ex = {ex.pairID: ex for ex in reader.read()}
    data = {}
    with open(src_filename, encoding='utf8') as f:
        for line in f:
            fields = line.split()
            pair_id = fields[0]
            data[pair_id] = {
                'annotations': fields[1: ],
                'example': id2ex[pair_id]}
    return data


def build_dataset(reader, phi, vectorizer=None, vectorize=True):
    """Create a dataset for training classifiers using `sklearn`.

    Parameters
    ----------
    reader : `NLIReader` instance or one of its subclasses.
    phi : feature function
        Any function that maps `NLIExample` instances to
        bool/int/float-valued dicts.
    assess_reader : `NLIReader` or one of its subclasses.
        If None, then random train/test splits are performed.
    vectorizer : `sklearn.feature_extraction.DictVectorizer`
        If this is None, then a new `DictVectorizer` is created and
        used to turn the list of dicts created by `phi` into a
        feature matrix. This happens when we are training.
        If this is not None, then it's assumed to be a `DictVectorizer`
        and used to transform the list of dicts. This happens in
        assessment, when we take in new instances and need to
        featurize them as we did in training.
    vectorize : bool
        Whether or not to use a `DictVectorizer` to create the feature
        matrix. If False, then it is assumed that `phi` does this,
        which is appropriate for models that featurize their own data.

    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels), 'vectorizer' (the `DictVectorizer`), and
        'raw_examples' (the original tree pairs, for error analysis).

    """
    feats = []
    labels = []
    raw_examples = []
    for ex in reader.read():
        label = ex.label
        d = phi(ex)
        feats.append(d)
        labels.append(label)
        raw_examples.append((ex.__dict__))
    if vectorize:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=True)
            feat_matrix = vectorizer.fit_transform(feats)
        else:
            feat_matrix = vectorizer.transform(feats)
    else:
        feat_matrix = feats
    return {'X': feat_matrix,
            'y': labels,
            'vectorizer': vectorizer,
            'raw_examples': raw_examples}


def experiment(
        train_reader,
        phi,
        train_func,
        assess_reader=None,
        train_size=0.7,
        score_func=utils.safe_macro_f1,
        vectorize=True,
        verbose=True,
        random_state=None):
    """Generic experimental framework for NLI. Either assesses with a
    random train/test split of `train_reader` or with `assess_reader` if
    it is given.

    Parameters
    ----------
    train_reader : `NLIReader`
        Iterator for training data.
    phi : feature function
        Any function that maps `NLIExample` instances to
        bool/int/float-valued dicts.
    train_func : model wrapper (default: `fit_maxent_classifier`)
        Any function that takes a feature matrix and a label list
        as its values and returns a fitted model with a `predict`
        function that operates on feature matrices.
    assess_reader : None, or `NLIReader` or one of its subclasses
        If None, then the data from `train_reader` are split into
        a random train/test split, with the the train percentage
        determined by `train_size`.
    train_size : float
        If `assess_reader` is None, then this is the percentage of
        `train_reader` devoted to training. If `assess_reader` is
        not None, then this value is ignored.
    score_metric : function name
        This should be an `sklearn.metrics` scoring function. The
        default is weighted average F1 (macro-averaged F1). For
        comparison with the SST literature, `accuracy_score` might
        be used instead. For micro-averaged F1, use
          (lambda y, y_pred : f1_score(y, y_pred, average='micro', pos_label=None))
        For other metrics that can be used here, see
        see http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    vectorize : bool
       Whether to use a DictVectorizer. Set this to False for
       deep learning models that process their own input.
    verbose : bool
        Whether to print out the model assessment to standard output.
        Set to False for statistical testing via repeated runs.
    random_state : int or None
        Optionally set the random seed for consistent sampling.

    Prints
    -------
    To standard output, if `verbose=True`
        Model precision/recall/F1 report. Accuracy is micro-F1 and is
        reported because many NLI papers report that figure, but the
        precision/recall/F1 are better given the slight class imbalances.

    Returns
    -------
    dict with keys
        'model': trained model
        'phi': the function used for featurization
        'train_dataset': a dataset as returned by `build_dataset`
        'assess_dataset': a dataset as returned by `build_dataset`
        'predictions': predictions on the assessment data
        'metric': `score_func.__name__`
        'score': the `score_func` score on the assessment data

    """
    # Train dataset:
    train = build_dataset(
        train_reader,
        phi,
        vectorizer=None,
        vectorize=vectorize)
    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    raw_train = train['raw_examples']
    if assess_reader == None:
         X_train, X_assess, y_train, y_assess, raw_train, raw_assess = train_test_split(
            X_train, y_train, raw_train,
            train_size=train_size, test_size=None, random_state=random_state)
         assess = {
            'X': X_assess,
            'y': y_assess,
            'vectorizer': train['vectorizer'],
            'raw_examples': raw_assess}
    else:
        # Assessment dataset using the training vectorizer:
        assess = build_dataset(
            assess_reader,
            phi,
            vectorizer=train['vectorizer'],
            vectorize=vectorize)
        X_assess, y_assess = assess['X'], assess['y']
    # Train:
    mod = train_func(X_train, y_train)
    # Predictions:
    predictions = mod.predict(X_assess)
    # Report:
    if verbose:
        print(classification_report(y_assess, predictions, digits=3))
    # Return the overall score and experimental info:
    return {
        'model': mod,
        'phi': phi,
        'train_dataset': train,
        'assess_dataset': assess,
        'predictions': predictions,
        'metric': score_func.__name__,
        'score': score_func(y_assess, predictions)}
