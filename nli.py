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
__version__ = "CS224u, Stanford, Spring 2020"


CONDITION_NAMES = [
    'edge_disjoint',
    'word_disjoint',
    'word_disjoint_balanced']


def word_entail_featurize(data, vector_func, vector_combo_func):
    X = []
    y = []
    for (w1, w2), label in data:
        rep = vector_combo_func(vector_func(w1), vector_func(w2))
        X.append(rep)
        y.append(label)
    return X, y


def wordentail_experiment(
        train_data,
        assess_data,
        vector_func,
        vector_combo_func,
        model):
    """Train and evaluation code for the word-level entailment task.

    Parameters
    ----------
    train_data : list
    assess_data : list
    vector_func : function
        Any function mapping words in the vocab for `wordentail_data`
        to vector representations
    vector_combo_func : function
        Any function for combining two vectors into a new vector
        of fixed dimensionality.
    model : class with `fit` and `predict` methods

    Prints
    ------
    To standard ouput
        An sklearn classification report for all three splits.

    Returns
    -------
    dict with structure

        'model': the trained model
        'train_condition': train_condition
        'assess_condition': assess_condition
        'macro-F1': score for 'assess_condition'
        'vector_func': vector_func
        'vector_combo_func': vector_combo_func

    We pass 'vector_func' and 'vector_combo_func' through to ensure alignment
    between these experiments and the bake-off evaluation.

    """
    X_train, y_train = word_entail_featurize(
        train_data,  vector_func, vector_combo_func)
    X_dev, y_dev = word_entail_featurize(
        assess_data, vector_func, vector_combo_func)
    model.fit(X_train, y_train)
    predictions = model.predict(X_dev)
    # Report:
    print(classification_report(y_dev, predictions, digits=3))
    macrof1 = utils.safe_macro_f1(y_dev, predictions)
    return {
        'model': model,
        'train_data': train_data,
        'assess_data': assess_data,
        'macro-F1': macrof1,
        'vector_func': vector_func,
        'vector_combo_func': vector_combo_func}


def bake_off_evaluation(experiment_results, test_data_filename=None):
    """Function for evaluating a trained model on the bake-off test set.

    Parameters
    ----------
    experiment_results : dict
        This should be the return value of `experiment` with at least
        keys 'model', 'vector_func', and 'vector_combo_func'.
    test_data_filename : str or None
        Full path to the test data. If `None`, then we assume the file is
        'data/nlidata/nli_wordentail_bakeoff_data-test.json'.

    Prints
    ------
    To standard ouput
        An sklearn classification report for all three splits.

    """
    if test_data_filename is None:
        test_data_filename = os.path.join(
            'data', 'nlidata', 'nli_wordentail_bakeoff_data-test.json')
    with open(test_data_filename, encoding='utf8') as f:
        wordentail_data = json.load(f)
    X_test, y_test = word_entail_featurize(
        wordentail_data['word_disjoint']['test'],
        vector_func=experiment_results['vector_func'],
        vector_combo_func=experiment_results['vector_combo_func'])
    predictions = experiment_results['model'].predict(X_test)
    # Report:
    print(classification_report(y_test, predictions, digits=3))



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
        return repr(self)

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
    gold_label_attr_name : str
        To accommodate different field names across NLI datasets.
        The default is 'gold_label', as in SNLI and MultiNLI.

    """
    def __init__(self,
            src_filename,
            filter_unlabeled=True,
            samp_percentage=None,
            random_state=None,
            gold_label_attr_name='gold_label'):
        self.src_filename = src_filename
        self.filter_unlabeled = filter_unlabeled
        self.samp_percentage = samp_percentage
        self.random_state = random_state
        self.gold_label_attr_name = gold_label_attr_name

    def read(self):
        """Primary interface.

        Yields
        ------
        `NLIExample`

        """
        if isinstance(self.src_filename, str):
            src_filenames = [self.src_filename]
        else:
            src_filenames = self.src_filename
        random.seed(self.random_state)
        for src_filename in src_filenames:
            for line in open(src_filename, encoding='utf8'):
                if (not self.samp_percentage) or random.random() <= self.samp_percentage:
                    d = json.loads(line)
                    ex = NLIExample(d)
                    gold_label = getattr(ex, self.gold_label_attr_name)
                    if (not self.filter_unlabeled) or gold_label != '-':
                        yield ex

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"NLIReader({})""".format(d)



class SNLITrainReader(NLIReader):
    def __init__(self, snli_home, **kwargs):
        src_filename = os.path.join(
            snli_home, "snli_1.0_train.jsonl")
        super(SNLITrainReader, self).__init__(src_filename, **kwargs)


class SNLIDevReader(NLIReader):
    def __init__(self, snli_home, **kwargs):
        src_filename = os.path.join(
            snli_home, "snli_1.0_dev.jsonl")
        super(SNLIDevReader, self).__init__(src_filename, **kwargs)


class MultiNLITrainReader(NLIReader):
    def __init__(self, snli_home, **kwargs):
        src_filename = os.path.join(
            snli_home, "multinli_1.0_train.jsonl")
        super(MultiNLITrainReader, self).__init__(src_filename, **kwargs)


class MultiNLIMatchedDevReader(NLIReader):
    def __init__(self, multinli_home, **kwargs):
        src_filename = os.path.join(
            multinli_home, "multinli_1.0_dev_matched.jsonl")
        super(MultiNLIMatchedDevReader, self).__init__(src_filename, **kwargs)


class MultiNLIMismatchedDevReader(NLIReader):
    def __init__(self, multinli_home, **kwargs):
        src_filename = os.path.join(
            multinli_home, "multinli_1.0_dev_mismatched.jsonl")
        super(MultiNLIMismatchedDevReader, self).__init__(src_filename, **kwargs)


class ANLIReader(NLIReader):
    def __init__(self, anli_home, anli_type, rounds=(1,2,3), **kwargs):
        if not all(int(i) in {1,2,3} for i in rounds):
            raise ValueError("Available ANLI rounds are {1,2,3}.")
        self.rounds = rounds
        self.src_filename = []
        for r in self.rounds:
            self.src_filename.append(
                os.path.join(
                    anli_home,
                    "R{}".format(r),
                    "{}.jsonl".format(anli_type)))
        super().__init__(
            self.src_filename,
            gold_label_attr_name='label',
            **kwargs)


class ANLITrainReader(ANLIReader):
    def __init__(self, anli_home, **kwargs):
        super().__init__(anli_home, 'train', **kwargs)


class ANLIDevReader(ANLIReader):
    def __init__(self, anli_home, **kwargs):
        super().__init__(anli_home, 'dev', **kwargs)



def read_annotated_subset(src_filename, multinli_home):
    """Given an annotation filename from MultiNLI's separate
    annotation distribution, associate it with the appropriate
    dev examples.

    Parameters
    ----------
    src_filename : str
        Full pat to the annotation file.
    multinli_home : str
        Full path to the MultiNLI corpus directory.

    Returns
    -------
    dict
        Maps pairID values to dicts with keys 'annotations' and
        'example', where 'annotations' gives a list of str and
        'example' gives an `NLIExample` instance.

    """
    if 'mismatched' in src_filename:
        reader = MultiNLIMismatchedDevReader(multinli_home)
    else:
        reader = MultiNLIMatchedDevReader(multinli_home)
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
