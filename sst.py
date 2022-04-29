import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import scipy.stats
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2022"


def sentiment_reader(src_filename, include_subtrees=True, dedup=False):
    """
    Iterator for our distribution of the SST-3 and other files in
    that format.

    Parameters
    ----------
    src_filename : str
        Full path to the file to be read.

    include_subtrees : bool
        If True, then the subtrees are returned as separate examples.
        This affects only the train split. For dev and test, only
        the full examples are included.

    dedup : bool
        If True, only one copy of each (example, label) pair is included.
        This mainly affects the train set, though there is one repeated
        example in the dev set.

    Yields
    ------
    pd.DataFrame with columns ['example_id', 'sentence', 'label']

    """
    df = pd.read_csv(src_filename)
    if not include_subtrees:
        df = df[df.is_subtree == 0]
    if dedup:
        df = df.groupby(['sentence', 'label']).apply(lambda x: x.iloc[0])
        df = df.reset_index(drop=True)
    return df


def train_reader(sst_home, include_subtrees=False, dedup=False):
    """
    Convenience function for reading the SST-3 train file.

    """
    src = os.path.join(sst_home, 'sst3-train.csv')
    return sentiment_reader(
        src, include_subtrees=include_subtrees, dedup=dedup)


def dev_reader(sst_home, include_subtrees=False, dedup=False):
    """
    Convenience function for reading the SST-3 dev file.

    """
    src = os.path.join(sst_home, 'sst3-dev.csv')
    return sentiment_reader(
        src, include_subtrees=include_subtrees, dedup=dedup)


def test_reader(sst_home, include_subtrees=False, dedup=False):
    """
    Convenience function for reading the SST-3 test file, unlabeled.
    This function should be used only for the final stages of a
    project, to obtain a submission to be evaluated. If you need
    to do an evaluation yourself with the labeled dataset, use
    `sentiment_reader` pointing to the labeled version of this
    dataset.

    """
    src = os.path.join(sst_home, 'sst3-test-unlabeled.csv')
    return sentiment_reader(
        src, include_subtrees=include_subtrees, dedup=dedup)


def bakeoff_dev_reader(sst_home, include_subtrees=False, dedup=False):
    """
    Convenience function for reading the bakeoff dev file.

    """
    src = os.path.join(sst_home, 'cs224u-sentiment-dev.csv')
    return sentiment_reader(
        src, include_subtrees=include_subtrees, dedup=dedup)


def bakeoff_test_reader(sst_home, include_subtrees=False, dedup=False):
    """
    Convenience function for reading the bakeoff test file, unlabeled.

    """
    src = os.path.join(sst_home, 'cs224u-sentiment-test-unlabeled.csv')
    return sentiment_reader(
        src, include_subtrees=include_subtrees, dedup=dedup)


def build_dataset(dataframes, phi, vectorizer=None, vectorize=True):
    """
    Core general function for building experimental datasets.

    Parameters
    ----------
    dataframes : pd.DataFrame or list of pd.DataFrame
        The dataset or datasets to process, as read in by
        `sentiment_reader`.

    phi : feature function
       Any function that takes a string as input and returns a
       bool/int/float-valued dict as output.

    vectorizer : sklearn.feature_extraction.DictVectorizer
       If this is None, then a new `DictVectorizer` is created and
       used to turn the list of dicts created by `phi` into a
       feature matrix. This happens when we are training.
       If this is not None, then it's assumed to be a `DictVectorizer`
       and used to transform the list of dicts. This happens in
       assessment, when we take in new instances and need to
       featurize them as we did in training.

    vectorize : bool
       Whether to use a DictVectorizer. Set this to False for
       deep learning models that process their own input.

    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels), 'vectorizer' (the `DictVectorizer`), and
        'raw_examples' (the `nltk.Tree` objects, for error analysis).

    """
    if isinstance(dataframes, (list, tuple)):
        df = pd.concat(dataframes)
    else:
        df = dataframes

    raw_examples = list(df.sentence.values)

    feat_dicts = list(df.sentence.apply(phi).values)

    if 'label' in df.columns:
        labels = list(df.label.values)
    else:
        labels = None

    feat_matrix = None
    if vectorize:
        # In training, we want a new vectorizer:
        if vectorizer is None:
            vectorizer = DictVectorizer(sparse=False)
            feat_matrix = vectorizer.fit_transform(feat_dicts)
        # In assessment, we featurize using the existing vectorizer:
        else:
            feat_matrix = vectorizer.transform(feat_dicts)
    else:
        feat_matrix = feat_dicts

    return {'X': feat_matrix,
            'y': labels,
            'vectorizer': vectorizer,
            'raw_examples': raw_examples}


def experiment(
        train_dataframes,
        phi,
        train_func,
        assess_dataframes=None,
        train_size=0.7,
        score_func=utils.safe_macro_f1,
        vectorize=True,
        verbose=True,
        random_state=None):
    """
    Generic experimental framework. Either assesses with a random
    train/test split of `train_reader` or with `assess_reader` if
    it is given.

    Parameters
    ----------
    train_dataframes : pd.DataFrame or list of pd.DataFrame
        The dataset or datasets to process, as read in by
        `sentiment_reader`.

    phi : feature function
        Any function that takes an `nltk.Tree` instance as input
        and returns a bool/int/float-valued dict as output.

    train_func : model wrapper
        Any function that takes a feature matrix and a label list
        as its values and returns a fitted model with a `predict`
        function that operates on feature matrices.

    assess_dataframes : pd.DataFrame, list of pd.DataFrame or None
        If None, then the df from `train_dataframes` is split into
        a random train/test split, with the the train percentage
        determined by `train_size`. If not None, then this should
        be a dataset or datasets to process, as read in by
        `sentiment_reader`. Each such dataset will be read and
        used in a separate evaluation.

    train_size : float (default: 0.7)
        If `assess_reader` is None, then this is the percentage of
        `train_reader` devoted to training. If `assess_reader` is
        not None, then this value is ignored.

    score_metric : function name (default: `utils.safe_macro_f1`)
        This should be an `sklearn.metrics` scoring function. The
        default is weighted average F1 (macro-averaged F1). For
        comparison with the SST literature, `accuracy_score` might
        be used instead. For other metrics that can be used here,
        see http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    vectorize : bool
        Whether to use a DictVectorizer. Set this to False for
        deep learning models that process their own input.

    verbose : bool (default: True)
        Whether to print out the model assessment to standard output.
        Set to False for statistical testing via repeated runs.

    random_state : int or None
        Optionally set the random seed for consistent sampling
        where random train/test splits are being created.

    Prints
    -------
    To standard output, if `verbose=True`
        Model precision/recall/F1 report. Accuracy is micro-F1 and is
        reported because many SST papers report that figure, but macro
        precision/recall/F1 is better given the class imbalances and the
        fact that performance across the classes can be highly variable.

    Returns
    -------
    dict with keys
        'model': trained model
        'phi': the function used for featurization
        'train_dataset': a dataset as returned by `build_dataset`
        'assess_datasets': list of datasets as returned by `build_dataset`
        'predictions': list of lists of predictions on the assessment datasets
        'metric': `score_func.__name__`
        'score': the `score_func` score on each of the assessment datasets

    """
    # Train dataset:
    train = build_dataset(
        train_dataframes,
        phi,
        vectorizer=None,
        vectorize=vectorize)

    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    raw_train = train['raw_examples']
    assess_datasets = []
    if assess_dataframes is None:
        X_train, X_assess, y_train, y_assess, raw_train, raw_assess = train_test_split(
            X_train, y_train, raw_train,
            train_size=train_size,
            test_size=None,
            random_state=random_state)
        assess_datasets.append({
            'X': X_assess,
            'y': y_assess,
            'vectorizer': train['vectorizer'],
            'raw_examples': raw_assess})
    else:
        if not isinstance(assess_dataframes, (tuple, list)):
            assess_dataframes = [assess_dataframes]
        for assess_df in assess_dataframes:
            # Assessment dataset using the training vectorizer:
            assess = build_dataset(
                assess_df,
                phi,
                vectorizer=train['vectorizer'],
                vectorize=vectorize)
            assess_datasets.append(assess)

    # Train:
    mod = train_func(X_train, y_train)

    # Predictions if we have labels:
    predictions = []
    scores = []
    for dataset_num, assess in enumerate(assess_datasets, start=1):
        preds = mod.predict(assess['X'])
        if assess['y'] is None:
            predictions.append(None)
            scores.append(None)
        else:
            if verbose:
                if len(assess_datasets) > 1:
                    print("Assessment dataset {}".format(dataset_num))
                print(classification_report(assess['y'], preds, digits=3))
            predictions.append(preds)
            scores.append(score_func(assess['y'], preds))
    true_scores = [s for s in scores if s is not None]
    if len(true_scores) > 1 and verbose:
        mean_score = np.mean(true_scores)
        print("Mean of macro-F1 scores: {0:.03f}".format(mean_score))


    # Return the overall scores and other experimental info:
    return {
        'model': mod,
        'phi': phi,
        'train_dataset': train,
        'assess_datasets': assess_datasets,
        'predictions': predictions,
        'metric': score_func.__name__,
        'scores': scores}


def compare_models(
        dataframes,
        phi1,
        train_func1,
        phi2=None,
        train_func2=None,
        vectorize1=True,
        vectorize2=True,
        stats_test=scipy.stats.wilcoxon,
        trials=10,
        train_size=0.7,
        score_func=utils.safe_macro_f1):
    """
    Wrapper for comparing models. The parameters are like those of
    `experiment`, with the same defaults, except

    Parameters
    ----------
    dataframes : pd.DataFrame or list of pd.DataFrame
        The dataset or datasets to process, as read in by
        `sentiment_reader`.

    phi1, phi2
        Just like `phi` for `experiment`. `phi1` defaults to
        `unigrams_phi`. If `phi2` is None, then it is set equal
        to `phi1`.

    train_func1, train_func2
        Just like `train_func` for `experiment`. If `train_func2`
        is None, then it is set equal to `train_func`.

    vectorize1, vectorize1 : bool
        Whether to vectorize the respective inputs. Use `False` for
        deep learning models that featurize their own input.

    stats_test : scipy.stats function
        Defaults to `scipy.stats.wilcoxon`, a non-parametric version
        of the paired t-test.

    trials : int (default: 10)
        Number of runs on random train/test splits of `reader`,
        with `train_size` controlling the amount of training data.

    train_size : float
        Percentage of data to use for training.

    Prints
    ------
    To standard output
        A report of the assessment.

    Returns
    -------
    (np.array, np.array, float)
        The first two are the scores from each model (length `trials`),
        and the third is the p-value returned by `stats_test`.

    """
    if phi2 == None:
        phi2 = phi1
    if train_func2 == None:
        train_func2 = train_func1
    experiments1 = [experiment(dataframes,
        phi=phi1,
        train_func=train_func1,
        score_func=score_func,
        vectorize=vectorize1,
        verbose=False) for _ in range(trials)]
    experiments2 = [experiment(dataframes,
        phi=phi2,
        train_func=train_func2,
        score_func=score_func,
        vectorize=vectorize2,
        verbose=False) for _ in range(trials)]
    scores1 = np.array([d['scores'][0] for d in experiments1])
    scores2 = np.array([d['scores'][0] for d in experiments2])
    # stats_test returns (test_statistic, p-value). We keep just the p-value:
    pval = stats_test(scores1, scores2)[1]
    # Report:
    print('Model 1 mean: {0:.03f}'.format(scores1.mean()))
    print('Model 2 mean: {0:.03f}'.format(scores2.mean()))
    print('p = {0:.03f}'.format(pval if pval >= 0.001 else 'p < 0.001'))
    # Return the scores for later analysis, and the p value:
    return scores1, scores2, pval


def build_rnn_dataset(dataframes, tokenizer=lambda s: s.split()):
    """
    Given an SST reader, return the dataset as (X, y) training pairs.

    Parameters
    ----------
    dataframes : pd.DataFrame or list of pd.DataFrame
        The dataset or datasets to process, as read in by
        `sentiment_reader`.

    tokenizer : function from str to list of str
        Defaults to a whitespace tokenizer.

    Returns
    -------
    X, y
        Where X is a list of list of str, and y is the output label list.

    """
    if isinstance(dataframes, (list, tuple)):
        df = pd.concat(dataframes)
    else:
        df = dataframes
    X = list(df.sentence.apply(tokenizer))
    y = list(df.label.values)
    return X, y
