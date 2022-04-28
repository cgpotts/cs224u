import numpy as np
import random
import torch
from utils import randvec

__author__ = "Atticus Geiger"
__version__ = "CS224u, Stanford, Spring 2022"


def get_IIT_equality_dataset_both(embed_dim, size):
    train_dataset = IIT_PremackDatasetBoth(
        embed_dim=embed_dim,
        size=size)
    X_base_train, X_sources_train,  y_base_train, y_IIT_train, interventions = train_dataset.create()
    X_base_train = torch.tensor(X_base_train)
    X_sources_train = [torch.tensor(X_source_train) for X_source_train in X_sources_train]
    y_base_train = torch.tensor(y_base_train)
    y_IIT_train = torch.tensor(y_IIT_train)
    interventions = torch.tensor(interventions)
    return X_base_train, X_sources_train, y_base_train, y_IIT_train, interventions


def get_IIT_equality_dataset(variable, embed_dim, size):
    class_size = size/2
    train_dataset = IIT_PremackDataset(
        variable,
        embed_dim=embed_dim,
        n_pos=class_size,
        n_neg=class_size)
    X_base_train, X_sources_train, y_base_train, y_IIT_train, interventions = train_dataset.create()
    X_base_train = torch.tensor(X_base_train)
    X_sources_train = [torch.tensor(X_source_train) for X_source_train in X_sources_train]
    y_base_train = torch.tensor(y_base_train)
    y_IIT_train = torch.tensor(y_IIT_train)
    interventions = torch.tensor(interventions)
    return X_base_train, X_sources_train, y_base_train, y_IIT_train, interventions


def get_equality_dataset(embed_dim, size):
    class_size = size/2
    train_dataset = PremackDataset(
        embed_dim=embed_dim,
        n_pos=class_size,
        n_neg=class_size)
    X_train, y_train = train_dataset.create()

    test_dataset = PremackDataset(
        embed_dim=embed_dim,
        n_pos=class_size,
        n_neg=class_size)
    X_test, y_test = test_dataset.create()

    train_dataset.test_disjoint(test_dataset)
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)

    return X_train, X_test, y_train, y_test, test_dataset


class EqualityDataset:

    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self, embed_dim=50, n_pos=500, n_neg=500, flatten=True):
        """Creates simple equality datasets, which are basically lists
        of `((vec1, vec2), label)` instances, where `label == POS_LABEL`
        if `vec1 == vec2`, else `label == NEG_LABEL`. With `flatten=True`,
        the instances become `(vec1;vec2, label)`.

        Parameters
        ----------
        embed_dim : int
            Sets the dimensionality of the individual component vectors.
        n_pos : int
        n_neg : int
        flatten : bool
            If False, instances are of the form ((vec1, vec2), label).
            If True, vec1 and vec2 are concatenated, creating instances
            (x, label) where len(x) == embed_dim*2.

        Usage
        -----
        dataset = EqualityDataset()
        X, y = dataset.create()

        Attributes
        ----------
        embed_dim : int
        n_pos : int
        n_neg : int
        flatten : bool

        """
        self.embed_dim = embed_dim
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.flatten = flatten

    def create(self):
        """Main interface

        Attributes
        ----------
        data : list
            Shuffled version of the raw instances, ignoring `self.flatten`.
            Thus, these are all of the form `((vec1, vec2), label)`

        X : np.array
            The dimensionality depends on `self.flatten`. If it is
            False, then `X.shape == (n_pos+n_neg, 2, embed_dim)`. If it
            is True, then `X.shape == (n_pos+n_neg, embed_dim*2)`.

        y : list
            Containing `POS_LABEL` and `NEG_LABEL`. Length: n_pos+n_neg

        Returns
        -------
        self.X, self.y

        """
        self.data = []
        self.data += self._create_pos()
        self.data += self._create_neg()
        random.shuffle(self.data)
        data = self.data.copy()
        if self.flatten:
            data = [(np.concatenate(x), label) for x, label in data]
        X, y = zip(*data)
        self.X = np.array(X)
        self.y = y
        return self.X, self.y

    def test_disjoint(self, other_dataset):
        these_vecs = {tuple(x) for pair, label in self.data for x in pair}
        other_vecs = {tuple(x) for pair, label in other_dataset.data for x in pair}
        shared = these_vecs & other_vecs
        assert len(shared) == 0, \
            f"This dataset and the other dataset shared {len(shared)} word-level reps."

    def _create_pos(self):
        data = []
        for _ in range(self.n_pos):
            vec = randvec(self.embed_dim)
            rep = (vec, vec)
            data.append((rep, self.POS_LABEL))
        return data

    def _create_neg(self):
        data = []
        for _ in range(self.n_neg):
            vec1 = randvec(self.embed_dim)
            vec2 = vec1.copy()
            while np.array_equal(vec1, vec2):
                vec2 = randvec(self.embed_dim)
            rep = (vec1, vec2)
            data.append((rep, self.NEG_LABEL))
        return data


class PremackDataset:

    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self, embed_dim=50, n_pos=500, n_neg=500,
                 flatten_root=True, flatten_leaves=True, intermediate=False):
        """Creates Premack datasets. Conceptually, the instances are

        (((a, b), (c, d)), label)

        where `label == POS_LABEL` if (a == b) == (c == d), else
        `label == NEG_LABEL`. With `flatten_leaves=True`, these become

        ((a;b, c;d), label)

        and with `flatten_root=True`, these become

        (a;b;c;d, label)

        and `flatten_root=True` means that `flatten_leaves=True`, since
        we can't flatten the roof without flattening the leaves.

        Parameters
        ----------
        embed_dim : int
            Sets the dimensionality of the individual component vectors.
        n_pos : int
        n_neg : int
        flatten_root : bool
        flatten_leaves : bool

        Usage
        -----
        dataset = EqualityDataset()
        X, y = dataset.create()

        Attributes
        ----------
        embed_dim : int
        n_pos : int
        n_neg : int
        flatten_root : bool
        flatten_leaves : bool
        n_same_same : n_pos / 2
        n_diff_diff : n_pos / 2
        n_same_diff : n_neg / 2
        n_diff_same : n_neg / 2

        Raises
        ------
        ValueError
            If `n_pos` or `n_neg` is not even, since this means we
            can't get an even distribtion of the two sub-types of
            each of those classes while also staying faithful to
            user's expected number of examples for each class.

        """
        self.embed_dim = embed_dim
        self.n_pos = n_pos
        self.n_neg = n_neg

        for n, v in ((n_pos, 'n_pos'), (n_neg, 'n_neg')):
            if n % 2 != 0:
                raise ValueError(
                    f"The value of {v} must be even to ensure a balanced "
                    f"split across its two sub-types of the {v} class.")

        self.n_same_same = int(n_pos / 2)
        self.n_diff_diff = int(n_pos / 2)
        self.n_same_diff = int(n_neg / 2)
        self.n_diff_same = int(n_neg / 2)
        self.flatten_root = flatten_root
        self.flatten_leaves = flatten_leaves
        self.intermediate = intermediate

    def create(self):
        """Main interface

        Attributes
        ----------
        data : list
            Shuffled version of the raw instances, ignoring
            `self.flatten_root` and `self.flatten_leaves`.
            Thus, these are all of the form `(((a, b), (c, d)), label)`

        X : np.array
            The dimensionality depends on `self.flatten_root` and
            `self.flatten_leaves`.

            If both are False, then

            `X.shape == (n_pos+n_neg, 2, 2, embed_dim)`

            If `self.flatten_root`, then

            `X.shape == (n_pos+n_neg, embed_dim*4)`

            If only `self.flatten_leaves`, then

            `X.shape == (n_pos+n_neg, 2, embed_dim*2)`

        y : list
            Containing `POS_LABEL` and `NEG_LABEL`. Length: n_pos+n_neg

        Returns
        -------
        self.X, self.y

        """
        self.data = []
        self.data += self._create_same_same()
        self.data += self._create_diff_diff()
        self.data += self._create_same_diff()
        self.data += self._create_diff_same()
        random.shuffle(self.data)
        data = self.data.copy()
        if self.flatten_root or self.flatten_leaves:
            data = [((np.concatenate(x1), np.concatenate(x2)), label)
                    for (x1, x2), label in data]
        if self.flatten_root:
            data = [(np.concatenate(x), label) for x, label in data]
        X, y = zip(*data)
        self.X = np.array(X)
        self.y = y
        return self.X, self.y

    def test_disjoint(self, other_dataset):
        these_vecs = {tuple(x) for root_pair, label in self.data
                               for pair in root_pair for x in pair}
        other_vecs = {tuple(x) for root_pair, label in other_dataset.data
                               for pair in root_pair for x in pair}
        shared = these_vecs & other_vecs
        assert len(shared) == 0, \
            f"This dataset and the other dataset shared {len(shared)} word-level reps."

    def _create_same_same(self):
        data = []
        for _ in range(self.n_same_same):
            left = self._create_same_pair()
            right = self._create_same_pair()
            rep = (left, right)
            data.append((rep, self.POS_LABEL))
        return data

    def _create_diff_diff(self):
        data = []
        for _ in range(self.n_diff_diff):
            left = self._create_diff_pair()
            right = self._create_diff_pair()
            rep = (left, right)
            data.append((rep, self.POS_LABEL))
        return data

    def _create_same_diff(self):
        data = []
        for _ in range(self.n_same_diff):
            left = self._create_same_pair()
            right = self._create_diff_pair()
            rep = (left, right)
            data.append((rep, self.NEG_LABEL))
        return data

    def _create_diff_same(self):
        data = []
        for _ in range(self.n_diff_same):
            left = self._create_diff_pair()
            right = self._create_same_pair()
            rep = (left, right)
            data.append((rep, self.NEG_LABEL))
        return data

    def _create_same_pair(self):
        vec = randvec(self.embed_dim)
        return (vec, vec)

    def _create_diff_pair(self):
        vec1 = randvec(self.embed_dim)
        vec2 = randvec(self.embed_dim)
        assert not np.array_equal(vec1, vec2)
        return (vec1, vec2)


class PremackDatasetLeafFlattened(PremackDataset):
    def __init__(self, embed_dim=50, n_pos=500, n_neg=500):
        super().__init__(
            embed_dim=embed_dim,
            n_pos=n_pos,
            n_neg=n_neg,
            flatten_leaves=True,
            flatten_root=False,
            intermediate=False)

class IIT_PremackDataset:

    V1 = 0
    V2 = 1
    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self, variable, embed_dim=50, n_pos=500, n_neg=500,
                 flatten_root=True, flatten_leaves=True, intermediate=False):

        self.variable = variable
        self.embed_dim = embed_dim
        self.n_pos = n_pos
        self.n_neg = n_neg

        for n, v in ((n_pos, 'n_pos'), (n_neg, 'n_neg')):
            if n % 2 != 0:
                raise ValueError(
                    f"The value of {v} must be even to ensure a balanced "
                    f"split across its two sub-types of the {v} class.")

        self.n_same_same_to_same = int(n_pos / 4)
        self.n_diff_diff_to_same = int(n_neg / 4)
        self.n_same_diff_to_same = int(n_neg / 4)
        self.n_diff_same_to_same = int(n_neg / 4)

        self.n_same_same_to_diff = int(n_neg / 4)
        self.n_diff_diff_to_diff = int(n_neg / 4)
        self.n_same_diff_to_diff = int(n_neg / 4)
        self.n_diff_same_to_diff = int(n_neg / 4)

        self.flatten_root = flatten_root
        self.flatten_leaves = flatten_leaves
        self.intermediate = intermediate

    def create(self):
        self.data = []
        self.data += self._create_same_same_to_same()
        self.data += self._create_diff_diff_to_same()
        self.data += self._create_same_diff_to_same()
        self.data += self._create_diff_same_to_same()
        self.data += self._create_same_same_to_diff()
        self.data += self._create_diff_diff_to_diff()
        self.data += self._create_same_diff_to_diff()
        self.data += self._create_diff_same_to_diff()
        random.shuffle(self.data)
        data = self.data.copy()
        if self.flatten_root or self.flatten_leaves:
            data = [((np.concatenate(x1), np.concatenate(x2)),(np.concatenate(x3), np.concatenate(x4)), base_label, IIT_label, intervention)
                    for (x1, x2,x3,x4), base_label, IIT_label, intervention in data]
        if self.flatten_root:
            data = [(np.concatenate(base), np.concatenate(source), label, IIT_label, intervention)
                    for base, source, label, IIT_label, intervention in data]
        base, source, y, IIT_y, interventions = zip(*data)
        self.base = np.array(base)
        self.source = np.array(source)
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        self.sources = list()
        self.sources.append(self.source)
        return self.base, self.sources, self.y, self.IIT_y, self.interventions

    def _create_same_same_to_same(self):
        data = []
        for _ in range(self.n_same_same_to_same):
            base_left = self._create_same_pair()
            base_right = self._create_same_pair()
            base_label = self.POS_LABEL
            if self.variable == "V1":
                source_left = self._create_same_pair()
                source_right = self._create_random_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_same_pair()
                intervention = self.V2
                IIT_label = self.POS_LABEL
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_diff_diff_to_same(self):
        data = []
        for _ in range(self.n_diff_diff_to_same):
            base_left = self._create_diff_pair()
            base_right = self._create_diff_pair()
            base_label = self.POS_LABEL
            if self.variable == "V1":
                source_left = self._create_same_pair()
                source_right = self._create_random_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_same_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_same_diff_to_same(self):
        data = []
        for _ in range(self.n_same_diff_to_same):
            base_left = self._create_same_pair()
            base_right = self._create_diff_pair()
            base_label = self.NEG_LABEL
            if self.variable == "V1":
                source_left = self._create_same_pair()
                source_right = self._create_random_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_same_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_diff_same_to_same(self):
        data = []
        for _ in range(self.n_diff_same_to_same):
            base_left = self._create_diff_pair()
            base_right = self._create_same_pair()
            base_label = self.NEG_LABEL
            if self.variable == "V1":
                source_left = self._create_same_pair()
                source_right = self._create_random_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_same_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_same_same_to_diff(self):
        data = []
        for _ in range(self.n_same_same_to_diff):
            base_left = self._create_same_pair()
            base_right = self._create_same_pair()
            base_label = self.POS_LABEL
            if self.variable == "V1":
                source_left = self._create_diff_pair()
                source_right = self._create_random_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_diff_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_diff_diff_to_diff(self):
        data = []
        for _ in range(self.n_diff_diff_to_diff):
            base_left = self._create_diff_pair()
            base_right = self._create_diff_pair()
            base_label = self.POS_LABEL
            if self.variable == "V1":
                source_left = self._create_diff_pair()
                source_right = self._create_random_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_diff_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_same_diff_to_diff(self):
        data = []
        for _ in range(self.n_same_diff_to_diff):
            base_left = self._create_same_pair()
            base_right = self._create_diff_pair()
            base_label = self.NEG_LABEL
            if self.variable == "V1":
                source_left = self._create_diff_pair()
                source_right = self._create_random_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_diff_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_diff_same_to_diff(self):
        data = []
        for _ in range(self.n_diff_same_to_diff):
            base_left = self._create_diff_pair()
            base_right = self._create_same_pair()
            base_label = self.NEG_LABEL
            if self.variable == "V1":
                source_left = self._create_diff_pair()
                source_right = self._create_random_pair()
                IIT_label = self.NEG_LABEL
                intervention = self.V1
            if self.variable == "V2":
                source_left = self._create_random_pair()
                source_right = self._create_diff_pair()
                IIT_label = self.POS_LABEL
                intervention = self.V2
            rep = (base_left, base_right, source_left,source_right)
            data.append((rep, base_label, IIT_label, intervention))
        return data

    def _create_random_pair(self):
        if random.choice([True,False]):
            return self._create_same_pair()
        else:
            return self._create_diff_pair()

    def _create_same_pair(self):
        vec = randvec(self.embed_dim)
        return (vec, vec)

    def _create_diff_pair(self):
        vec1 = randvec(self.embed_dim)
        vec2 = randvec(self.embed_dim)
        assert not np.array_equal(vec1, vec2)
        return (vec1, vec2)


class IIT_PremackDatasetBoth:

    V1 = 0
    V2 = 1
    POS_LABEL = 1
    NEG_LABEL = 0
    both_coord_id = 2

    def __init__(self, size= 1000, embed_dim=50,  flatten_root=True, flatten_leaves=True, intermediate=False):

        self.embed_dim = embed_dim
        self.size= size


        self.flatten_root = flatten_root
        self.flatten_leaves = flatten_leaves
        self.intermediate = intermediate

    def create(self):
        data = []
        for _ in range(self.size):
            rep = [self._create_random_pair() for _ in range(6)]
            if (rep[0][0] == rep[0][1]).all() == (rep[1][0] == rep[1][1]).all():
                base_label = self.POS_LABEL
            else:
                base_label = self.NEG_LABEL
            if (rep[2][0] == rep[2][1]).all() == (rep[5][0] == rep[5][1]).all():
                IIT_label = self.POS_LABEL
            else:
                IIT_label = self.NEG_LABEL
            data.append((rep,base_label, IIT_label, self.both_coord_id))
        random.shuffle(data)
        data = data.copy()
        if self.flatten_root or self.flatten_leaves:
            data = [
                (
                    (
                        (np.concatenate(x1), np.concatenate(x2)),
                        (np.concatenate(x3), np.concatenate(x4)),
                        (np.concatenate(x5), np.concatenate(x6))
                    ),
                    base_label, IIT_label, intervention
                )
                for (x1, x2,x3,x4,x5,x6), base_label, IIT_label, intervention in data
            ]
        if self.flatten_root:
            data = [(np.concatenate(base), np.concatenate(source),np.concatenate(source2), label, IIT_label, intervention)
                    for (base, source, source2), label, IIT_label, intervention in data]
        base, source, source2, y, IIT_y, interventions = zip(*data)
        self.base = np.array(base)
        self.source = np.array(source)
        self.source2 = np.array(source2)
        self.y = np.array(y)
        self.IIT_y = np.array(IIT_y)
        self.interventions = np.array(interventions)
        return self.base, [self.source, self.source2], self.y, self.IIT_y, self.interventions


    def _create_random_pair(self):
        if random.choice([True,False]):
            return self._create_same_pair()
        else:
            return self._create_diff_pair()

    def _create_same_pair(self):
        vec = randvec(self.embed_dim)
        return (vec, vec)

    def _create_diff_pair(self):
        vec1 = randvec(self.embed_dim)
        vec2 = randvec(self.embed_dim)
        assert not np.array_equal(vec1, vec2)
        return (vec1, vec2)
