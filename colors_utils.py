import colorsys
import re

WORD_RE_STR = r"""
(?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
|
(?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
|
(?:[\w_]+)                     # Words without apostrophes or dashes.
|
(?:\.(?:\s*\.){1,})            # Ellipsis dots.
|
(?:\*{1,})                     # Asterisk runs.
|
(?:\S)                         # Everything else that isn't whitespace.
"""

WORD_RE = re.compile(r"(%s)" % WORD_RE_STR, re.VERBOSE | re.I | re.UNICODE)


def basic_unigram_tokenizer(s, lower=True):
    words = WORD_RE.findall(s)
    if lower:
        words = [w.lower() for w in words]
    return words


def heuristic_ending_tokenizer(s, lower=True):
    words = basic_unigram_tokenizer(s, lower=lower)
    return [seg for w in words for seg in heuristic_segmenter(w)]


ENDINGS = ['er', 'est', 'ish']


def heuristic_segmenter(word):
    for ending in ENDINGS:
        if word.endswith(ending):
            return [word[:-len(ending)], '+' + ending]
    return [word]


def whitespace_tokenizer(s, lower=True):
    if lower:
        s = s.lower()
    return s.split()


def rgb_to_hsv(rgb):
    rgb_0_1 = [d / 255.0 for d in rgb[:3]]
    hsv_0_1 = colorsys.rgb_to_hsv(*rgb_0_1)
    return tuple(d * r for d, r in zip(hsv_0_1, [360.0, 100.0, 100.0]))


# HSL <-> HSV conversion based on C code by Ariya Hidayat:
#   http://ariya.ofilabs.com/2008/07/converting-between-hsl-and-hsv.html

def hsl_to_hsv(color):
    '''
    >>> hsl_to_hsv((120, 100, 50))
    (120.0, 100.0, 100.0)
    >>> hsl_to_hsv((0, 100, 100))
    (0.0, 0.0, 100.0)

    Saturation in HSV is undefined and arbitrarily 0 for black:

    >>> hsl_to_hsv((240, 100, 0))
    (240.0, 0.0, 0.0)
    '''
    hi, si, li = [float(d) for d in color]

    ho = hi
    si *= (li / 100.0) if li <= 50.0 else (1.0 - li / 100.0)
    vo = li + si
    so = (200.0 * si / vo) if vo else 0.0

    return (ho, so, vo)


def hsv_to_hsl(color):
    '''
    >>> hsv_to_hsl((120, 100, 100))
    (120.0, 100.0, 50.0)

    Saturation in HSL is undefined and arbitrarily 0 for black and white:

    >>> hsv_to_hsl((240, 0, 0))
    (240.0, 0.0, 0.0)
    >>> hsv_to_hsl((0, 0, 100))
    (0.0, 0.0, 100.0)
    '''
    hi, si, vi = [float(d) for d in color]

    ho = hi
    lo = (200.0 - si) * vi / 200.0
    so = si * vi / 200.0
    if lo >= 100.0 or lo <= 0.0:
        so = 0.0
    else:
        so /= (lo / 100.0) if lo <= 50.0 else (1.0 - lo / 100.0)

    return (ho, so, lo)

