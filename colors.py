from collections import defaultdict
import colorsys
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


TURN_BOUNDARY =  " ### "


class ColorsCorpusReader:
    """Basic interface for the Stanford Colors in Context corpus:

    https://cocolab.stanford.edu/datasets/colors.html

    Parameters
    ----------
    src_filename : str
        Full path to the corpus file.
    word_count : int or None
        If int, then only examples with `word_count` words in their
        'contents' field are included (as estimated by the number of
        whitespqce tokens). If None, then all examples are returned.
    normalize_colors : bool
         The colors in the corpus are in HLS format with values
         [0, 360], [0, 100], [0, 100]. If `normalize_colors=True`,
         these are scaled into [0, 1], [0, 1], [0, 1].

    Usage
    -----
    corpus = ColorsCorpusReader('filteredCorpus.csv')

    for ex in corpus.read():
        # ...

    """
    def __init__(self, src_filename, word_count=None, normalize_colors=True):
        self.src_filename = src_filename
        self.word_count = word_count
        self.normalize_colors = normalize_colors

    def read(self):
        """The main interface to the corpus.

        As in the paper, turns taken in the same game and round are
        grouped together into a single `ColorsCorpusExample` instance
        with the turn texts separated by `TURN_BOUNDARY`, formatted
        as a string.

        Yields
        ------
        `ColorsCorpusExample` with the `normalize_colors` attribute set
        as in `self.normalize_colors` in this class.

        """
        grouped = defaultdict(list)
        with open(self.src_filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['role'] == 'speaker' and self._word_count_filter(row):
                    grouped[(row['gameid'], row['roundNum'])].append(row)
        for rows in grouped.values():
            yield ColorsCorpusExample(
                rows, normalize_colors=self.normalize_colors)

    def _word_count_filter(self, row):
        return self.word_count is None or \
          row['contents'].count(" ") == (self.word_count-1)


class ColorsCorpusExample:
    """Interface to individual examples in the Stanford Colors in
    Context corpus.

    Parameters
    ----------
    rows : list of dict
        This contains all of the turns associated with a given game
        and round. The assumption is that all of the key-value pairs
        in these dicts are the same except for the 'contents' key.
    normalize_colors : bool
         The colors in the corpus are in HLS format with values
         [0, 360], [0, 100], [0, 100]. If `normalize_colors=True`,
         these are scaled into [0, 1], [0, 1], [0, 1].

    Usage
    -----
    We assume that these instances are created by `ColorsCorpusReader`.
    For an example of one being created directly, see
    `test/test_colors.py::test_color_corpus_example`.

    Note
    ----
    There are values in the corpus that are present in `rows` but
    not captured in attributes right now, to keep this code from
    growing very complex. It should be straightforward to bring
    in these additional attributes by subclassing this class.

    """
    def __init__(self, rows, normalize_colors=True):
        self.normalize_colors = normalize_colors
        self.contents = TURN_BOUNDARY.join([r['contents'] for r in rows])
        # Make sure our assumptions about these rows are correct:
        self._check_row_alignment(rows)
        row = rows[0]
        self.gameid = row['gameid']
        self.roundNum = int(row['roundNum'])
        self.condition = row['condition']
        self.outcome = row['outcome'] == 'true'
        self.clickStatus = row['clickStatus']
        self.color_data = []
        for typ in ['click', 'alt1', 'alt2']:
            self.color_data.append({
                'type': typ,
                'Status': row['{}Status'.format(typ)],
                'rep': self._get_color_rep(row, typ),
                'speaker': int(row['{}LocS'.format(typ)]),
                'listener': int(row['{}LocL'.format(typ)])})
        self.colors = self._get_reps_in_order('Status')
        self.listener_context = self._get_reps_in_order('listener')
        self.speaker_context = self._get_reps_in_order('speaker')

    def parse_turns(self):
        """"Turns the `contents` string into a list by splitting on
        `TURN_BOUNDARY`.

        Returns
        -------
        list of str

        """
        return self.contents.split(TURN_BOUNDARY)

    def display(self, typ='model'):
        """Prints examples to the screen in an intuitive format: the
        utterance text appears first, following by the three color
        patches, with the target identified by a black border in the
        'speaker' and 'model' variants.

        Parameters
        ----------
        typ : str
            Should be 'model', 'speaker', or 'listener'. This
            determines the order the color patches are given. For
            'speaker' and 'listener', this is the order in the corpus.
            For 'model', it is a version with the two distractors
            printed in their canonical order and the target given last.

        Raises
        ------
        ValueError
            If `typ` isn't one of 'model', 'speaker', 'listener'.

        Prints
        ------
        text to standard output and three color patches as a
        `matplotlib.pyplot` image. For notebook usage, this should
        all embed nicely.

        """
        print(self.contents)
        if typ == 'model':
            colors = self.colors
            target_index = 2
        elif typ == 'listener':
            colors = self.listener_context
            target_index = None
        elif typ == 'speaker':
            colors = self.speaker_context
            target_index = self._get_target_index('speaker')
        else:
            raise ValueError('`typ` options: "model", "listener", "speaker"')

        rgbs = [self._convert_hls_to_rgb(*c) for c in colors]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(3, 1))

        for i, c in enumerate(rgbs):
            ec = c if (i != target_index or typ == 'listener') else "black"
            patch = mpatch.Rectangle((0, 0), 1, 1, color=c, ec=ec, lw=8)
            axes[i].add_patch(patch)
            axes[i].axis('off')

    def _get_color_rep(self, row, typ):
        rep = []
        for dim in ['H', 'L', 'S']:
            colname = "{}Col{}".format(typ, dim)
            rep.append(float(row[colname]))
        if self.normalize_colors:
            rep = self._scale_color(*rep)
        return rep

    def _convert_hls_to_rgb(self, h, l, s):
        if not self.normalize_colors:
            h, l, s = self._scale_color(h, l, s)
        return colorsys.hls_to_rgb(h, l, s)

    @staticmethod
    def _scale_color(h, l, s):
        return [h/360, l/100, s/100]

    def _get_reps_in_order(self, field):
        colors = [(d[field], d['rep']) for d in self.color_data]
        return [rep for s, rep in sorted(colors)]

    def _get_target_index(self, field):
        for d in self.color_data:
            if d['Status'] == 'target':
                return d[field] - 1

    @staticmethod
    def _check_row_alignment(rows):
        """We expect all the dicts in `rows` to have the same
        keys and values except for the keys associated with the
        messages. This function tests this assumption holds.

        """
        keys = set(rows[0].keys())
        for row in rows[1:]:
            if set(row.keys()) != keys:
                raise RuntimeError(
                    "The dicts in the `rows` argument to `ColorsCorpusExample` "
                    "must have all the same keys.")
        exempted = {'contents', 'msgTime',
                    'numRawWords', 'numRawChars',
                    'numCleanWords', 'numCleanChars'}
        keys = keys - exempted
        for row in rows[1: ]:
            for key in keys:
                if rows[0][key] != row[key]:
                    raise RuntimeError(
                        "The dicts in the `rows` argument to `ColorsCorpusExample` "
                        "must have all the same key values except for the keys "
                        "associated with the message. The key {} has values {} "
                        "and {}".format(key, rows[0][key], row[key]))

    def __str__(self):
        return self.contents
