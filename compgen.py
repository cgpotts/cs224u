from collections import defaultdict
from itertools import product
import re

__author__ = "Christopher Potts and Zhengxuan Wu"
__version__ = "CS224u, Stanford, Spring 2023"


def recogs_exact_match(gold, pred, flag="000000"):
    gold = normalize_formula(gold)
    pred = normalize_formula(pred)
    gold_conj_set = get_conj_set(gold)
    # Loop over all viable mappings from pred_vars to gold_vars:
    for this_map in _candidate_variable_maps(gold, pred):
        phi = pred
        for sourcevar, targetvar in this_map.items():
            # The flag makes sure we don't accidentally do a chain
            # of replacements via successive changes in situations
            # where the domain and range of `this_map` share vars.
            phi = variable_change(phi, sourcevar, targetvar, flag=flag)
        phi = phi.replace(flag, "")
        phi_conj_set = get_conj_set(phi)
        # This step assumes that we have no conjuncts that are
        # tautologies, contradictions, or equality predications. If
        # such are introduced, they need to be identified ahead of
        # time and treated separately -- tautologies would be removed,
        # contradictions would reduce to comparisons of only those
        # conjuncts, and equality statements would call for special
        # handling related to variables mapping.
        if phi_conj_set == gold_conj_set:
            return True
    return False


def normalize_formula(phi):
    return phi.replace(" ", "").replace("AND" , " AND ")


binary_pred_re = re.compile(r"""
    (\w+)
    \s*
    \(
    \s*
    (\d+)
    \s*
    ,
    \s*
    (\d+)
    \s*
    \)""", re.VERBOSE)


unary_pred_re = re.compile(r"""
    (\w+)
    \s*
    \(
    \s*
    (\d+)
    \s*
    \)""", re.VERBOSE)


def _candidate_variable_maps(gold, pred):
    # This creates a mapping from tuples of predicates into their
    # associated variables. These serve as equivalence classes over
    # variables that could possibly be translations of each other.
    gold_map = _map_get_preds_to_vars(gold)
    pred_map = _map_get_preds_to_vars(pred)

    # For each prediction variable, get the set of potential
    # translations for it:
    pred2gold = defaultdict(list)
    for preds, pvars in pred_map.items():
        gvars = gold_map[preds]
        for pvar in pvars:
            pred2gold[pvar] = gold_map[preds]

    # Variable sets:
    gold_vars = set(get_variables(gold))
    pred_vars = set(get_variables(pred))

    # Now generate potentially viable mappings:
    for vals in list(product(*list(pred2gold.values()))):
        d = dict(zip(pred2gold.keys(), vals))
        if set(d.keys()) == pred_vars and set(d.values()) == gold_vars:
            yield d


def _map_get_preds_to_vars(phi):
    var2pred = defaultdict(list)
    for pred, var in unary_pred_re.findall(phi):
        var2pred[var].append(pred)
    # We could do somewhat less search by specializing to first and
    # second position for these predicates, but I think it's fine
    # as-is.
    for pred, var1, var2 in binary_pred_re.findall(phi):
        var2pred[var1].append(pred)
        var2pred[var2].append(pred)
    pred2var = defaultdict(list)
    for var, preds in var2pred.items():
        pred2var[tuple(sorted(preds))].append(var)
    return pred2var


def get_variables(phi):
    variable_re = re.compile(r"(\d+)")
    return variable_re.findall(phi)


def get_conj_set(phi):
    conj_splitter_re  = re.compile(r"\s*(?:AND|;)\s*")
    return set(conj_splitter_re.split(phi))


def variable_change(phi, sourcevar, targetvar, flag="000000"):
    replace_re = re.compile(rf"\b{sourcevar}\b")
    return replace_re.sub(f"{flag}{targetvar}", phi)
