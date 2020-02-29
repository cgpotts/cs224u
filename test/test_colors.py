from colors import ColorsCorpusReader, ColorsCorpusExample, TURN_BOUNDARY
import json
import os
import pytest

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2020"


@pytest.fixture
def test_rows():
    src_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'colors-test-data.json')
    with open(src_filename) as f:
        data = json.load(f)
    return data


# These are the colors in the test data file loaded by `test_rows`:
alt1 = [226.0, 50.0, 81.0]
alt2 = [283.0, 50.0, 87.0]
click = [248.0, 50.0, 92.0]


@pytest.mark.parametrize("attr, expected", [
    ['contents', 'Blue{}The darker blue one'.format(TURN_BOUNDARY)],
    ['gameid', '1124-1'],
    ['roundNum', 1],
    ['outcome', False],
    ['clickStatus', 'distr2'],
    ['colors', [alt2, click, alt1]],
    ['listener_context', [click, alt1, alt2]],
    ['speaker_context', [alt1, alt2, click]]
])
def test_color_corpus_example(attr, expected, test_rows):
    ex = ColorsCorpusExample(test_rows, normalize_colors=False)
    result = getattr(ex, attr)
    assert result == expected


def test_normalize_colors(test_rows):
    ex = ColorsCorpusExample(test_rows, normalize_colors=True)
    result = ex.colors[0]
    h, l, s = alt2
    expected = [h/360, l/100, s/100]
    assert result == expected


def test_parse_turns(test_rows):
    ex = ColorsCorpusExample(test_rows)
    result = ex.parse_turns()
    expected = ['Blue', 'The darker blue one']
    assert result == expected


def test_check_row_alignment(test_rows):
    rows = test_rows.copy()
    rows[0]['clickStatus'] = 'deliberate change'
    with pytest.raises(RuntimeError):
        ex = ColorsCorpusExample(rows)
