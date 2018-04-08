from collections import defaultdict
import json
from nltk.tree import Tree
import numpy as np
import os
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import utils


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2018"


BAKEOFF_CONDITION_NAMES = ['edge_disjoint', 'word_disjoint', 'word_disjoint_balanced']


def build_bakeoff_dataset(wordentail_data, vector_func, vector_combo_func):
    """
    Parameters
    ----------
    wordentail_data
        The contents of `wordentail_filename` loaded from JSON.
    vector_func : function
        Any function mapping words in the vocab for `wordentail_data`
        to vector representations
    vector_combo_func : function
        Any function for combining two vectors into a new vector
        of fixed dimensionality.

    Returns
    -------
    A dict in the same format as `wordentail_data` but with the
    pairs of strings for each example replaced by a single vector.
    """
    # A mapping from words (as strings) to their vector
    # representations, as determined by vector_func:
    vocab = wordentail_data['vocab']
    vectors = {w: vector_func(w) for w in vocab}
    # Dataset in the format required by the neural network:
    dataset = defaultdict(lambda: defaultdict(list))
    for condition in BAKEOFF_CONDITION_NAMES:
        for split, data in wordentail_data[condition].items():
            for (w1, w2), label in data:
                # Use vector_combo_func to combine the word vectors for
                # w1 and w2, as given by the vectors dictionary above,
                # and pair it with the singleton array containing clsname:
                rep = vector_combo_func(vectors[w1], vectors[w2])
                example = [rep, label]
                dataset[condition][split].append(example)
    dataset['vocab'] = vocab
    return dataset


def bakeoff_experiment(dataset, model, conditions=None):
    """Train and evaluation code for the word-level entailment task.

    Parameters
    ----------
    dataset : dict
        With keys `BAKEOFF_CONDITION_NAMES`, each with values that are lists of
        vector pairs, the first giving the example representation and the second
        giving its 1d output vector. The expectation is that this was created
        by `build_bakeoff_dataset`.
    model : class with `fit` and `predict` methods
    conditions : list or None
        If None, then all of `BAKEOFF_CONDITION_NAMES` are evaluated.
        If this is a list, then it should be a subset of
        `BAKEOFF_CONDITION_NAMES`.

    Prints
    ------
    To standard ouput
        An sklearn classification report for all three splits.

    """
    if conditions is None:
        conditions = BAKEOFF_CONDITION_NAMES
    else:
        for c in conditions:
            if c not in BAKEOFF_CONDITION_NAMES:
                raise ValueError(
                    "Condition {} is not recogized. Conditions must "
                    "be in {}".format(c, BAKEOFF_CONDITION_NAMES))
    # Train the network:
    for condition in conditions:
        cond_data = dataset[condition]
        X_train, y_train = zip(*cond_data['train'])
        model.fit(X_train, y_train)
        X_dev, y_dev = zip(*cond_data['dev'])
        predictions = model.predict(X_dev)
        # Report:
        print("="*70)
        print("{}".format(condition))
        print(classification_report(y_dev, predictions))
        if condition == 'word_disjoint_balanced':
            X_train, y_train = zip(*dataset['word_disjoint']['train'])
            model.fit(X_train, y_train)
            predictions = model.predict(X_dev)
            # Report:
            print("="*70)
            print("{}, training on word_disjoint".format(condition))
            print(classification_report(y_dev, predictions))


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


def get_edge_overlap_size(wordentail_data, split):
    train = {tuple(x) for x, y in wordentail_data[split]['train']}
    dev = {tuple(x) for x, y in wordentail_data[split]['dev']}
    return len(train & dev)


def get_vocab_overlap_size(wordentail_data, split):
    train = {w for x, y in wordentail_data[split]['train'] for w in x}
    dev = {w for x, y in wordentail_data[split]['dev'] for w in x}
    return len(train & dev)


class NLIExample(object):
    """For processing examples from SNLI or MultiNLI.

    Parameters
    ----------
    d : dict
        Derived from a JSON line in one of the corpus files. Each
        key-value pair becomes an attribute-value pair for the
        class. The tree strings are converted to `nltk.tree.Tree`
        instances using `str2tree`.

    """
    def __init__(self, d):
        for k, v in d.items():
            if '_binary_parse' in k:
                v = str2tree(v, binarize=True)
            elif '_parse' in k:
                v = str2tree(v, binarize=False)
            setattr(self, k, v)

    def __str__(self):
        return """{}\n{}\n{}""".format(
            self.sentence1, self.gold_label, self.sentence2)

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"NLIExample({})""".format(d)


class NLIReader(object):
    """Reader for SNLI/MultiNLI data.

    Parameters
    ----------
    src_filename : str
        Full path to the file to process.
    filter_unlabeled : bool
        Whether to leave out cases without a gold label.
    samp_percentage : float or None
        If not None, randomly sample approximately this percentage
        of lines.
    random_state : int or None
        Optionally set the random seed for consistent sampling.

    """
    def __init__(self,
            src_filename,
            filter_unlabeled=True,
            samp_percentage=None,
            random_state=None):
        self.src_filename = src_filename
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
        for line in open(self.src_filename):
            if (not self.samp_percentage) or random.random() <= self.samp_percentage:
                d = json.loads(line)
                ex = NLIExample(d)
                if (not self.filter_unlabeled) or ex.gold_label != '-':
                    yield ex

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"NLIReader({})""".format(d)


SNLI_HOME = os.path.join("nlidata", "snli_1.0")

MULTINLI_HOME = os.path.join("nlidata", "multinli_1.0")


class SNLITrainReader(NLIReader):
    def __init__(self, snli_home=SNLI_HOME, **kwargs):
        src_filename = os.path.join(
            snli_home, "snli_1.0_train.jsonl")
        super(SNLITrainReader, self).__init__(src_filename, **kwargs)


class SNLIDevReader(NLIReader):
    def __init__(self, snli_home=SNLI_HOME, **kwargs):
        src_filename = os.path.join(
            snli_home, "snli_1.0_dev.jsonl")
        super(SNLIDevReader, self).__init__(src_filename, **kwargs)


class MultiNLITrainReader(NLIReader):
    def __init__(self, snli_home=MULTINLI_HOME, **kwargs):
        src_filename = os.path.join(
            snli_home, "multinli_1.0_train.jsonl")
        super(MultiNLITrainReader, self).__init__(src_filename, **kwargs)


class MultiNLIMatchedDevReader(NLIReader):
    def __init__(self, multinli_home=MULTINLI_HOME, **kwargs):
        src_filename = os.path.join(
            multinli_home, "multinli_1.0_dev_matched.jsonl")
        super(MultiNLIMatchedDevReader, self).__init__(src_filename, **kwargs)


class MultiNLIMismatchedDevReader(NLIReader):
    def __init__(self, multinli_home=MULTINLI_HOME, **kwargs):
        src_filename = os.path.join(
            multinli_home, "multinli_1.0_dev_mismatched.jsonl")
        super(MultiNLIMismatchedDevReader, self).__init__(src_filename, **kwargs)


def read_annotated_subset(src_filename):
    """Given an annotation filename from MultiNLI's separate
    annotation distribution, associate it with the appropriate
    dev examples.

    Parameters
    ----------
    src_filename : str
        Full pat to the annotation file.

    Returns
    -------
    dict
        Maps pairID values to dicts with keys 'annotations' and
        'example', where 'annotations' gives a list of str and
        'example' gives an `NLIExample` instance.

    """
    if 'mismatched' in src_filename:
        reader = MultiNLIMismatchedDevReader()
    else:
        reader = MultiNLIMatchedDevReader()
    id2ex = {ex.pairID: ex for ex in reader.read()}
    data = {}
    with open(src_filename) as f:
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
        Maps trees to count dictionaries.
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
        t1 = ex.sentence1_parse
        t2 = ex.sentence2_parse
        label = ex.gold_label
        d = phi(t1, t2)
        feats.append(d)
        labels.append(label)
        raw_examples.append((t1, t2))
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
    train_reader : `NLIReader` (or one of its subclasses
        Iterator for training data.
    phi : feature function
        Any function that takes an `nltk.Tree` instance as input
        and returns a bool/int/float-valued dict as output.
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
        Model accuracy and a model precision/recall/F1 report. Accuracy is
        reported because many SST papers report that figure, but the
        precision/recall/F1 is better given the class imbalances and the
        fact that performance across the classes can be highly variable.

    Returns
    -------
    float
        The overall scoring metric as determined by `score_metric`.

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
    X_assess = None
    y_assess = None
    if assess_reader == None:
         X_train, X_assess, y_train, y_assess = train_test_split(
             X_train, y_train, train_size=train_size, test_size=None,
             random_state=random_state)
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
    # Return the overall score:
    return score_func(y_assess, predictions)
