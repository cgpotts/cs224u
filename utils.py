__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2016"


import unicodecsv as csv
import random
import numpy as np
from sklearn.metrics import f1_score



def build(src_filename, delimiter=',', header=True, quoting=csv.QUOTE_MINIMAL):
    """Reads in matrices from CSV or space-delimited files.
    
    Parameters
    ----------
    src_filename : str
        Full path to the file to read.
        
    delimiter : str (default: ',')
        Delimiter for fields in src_filename. Use delimter=' '
        for GloVe files.
        
    header : bool (default: True)
        Whether the file's first row contains column names. 
        Use header=False for GloVe files.
    
    quoting : csv style (default: QUOTE_MINIMAL)
        Use the default for normal csv files and csv.QUOTE_NONE for
        GloVe files.

    Returns
    -------
    (np.array, list of str, list of str)
       The first member is a dense 2d Numpy array, and the second 
       and third are lists of strings (row names and column names, 
       respectively). The third (column names) is None if the 
       input file has no header. The row names are assumed always 
       to be present in the leftmost column.    
    """
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


def build_glove(src_filename):
    return build(src_filename, delimiter=' ', header=False, quoting=csv.QUOTE_NONE)


def randmatrix(m, n, lower=-0.5, upper=0.5):
    """Creates an m x n matrix of random values in [lower, upper]"""
    return np.array([random.uniform(lower, upper) for i in range(m*n)]).reshape(m, n)


def safe_macro_f1(y, y_pred):
    """Macro-averaged F1, forcing `sklearn` to report as a multiclass
    problem even when there are just two classes. `y` is the list of 
    gold labels and `y_pred` is the list of predicted labels."""
    return f1_score(y, y_pred, average='macro', pos_label=None)


def multirun_classification_report(results, digits=3, labels=None):
    results = np.array(results).mean(0).T
    labels = map(str, range(results.shape(0))) if labels == None else labels    
    header = ('', 'precision', 'recall', 'f1', 'support')
    width = 9
    labelmaxwidth = max([len(x) for x in labels])
    
    def cellfmt(x, typ='float', width=width):
        val = ""
        if typ=='int':
            val = "%0d" % x
        elif typ=='str':
            val = x
        else:
            val "%0.03f" % x
        return val.rjust(width)

    def rowfmt(row, label):
        s = [cellfmt(label, typ='str'),
             " ".join([cellfmt(x, typ='float') for x in row[:-1]]),
             cellfmt(row[-1], typ='int')]
        return " ".join(s)

    rows = [" ".join([x.rjust(width) for x in header])]
    rows += [rowfmt(row, label) for row, label in zip(results, labels)]
    return "\n".join(rows)
