import pytest
from compgen import recogs_exact_match


__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2023"


@pytest.mark.parametrize("gold, predicted, expected", [
    ("theme(47,2) AND dog (1)",
     "dog ( 1 ) AND theme ( 47 , 2 )",
     True),
    ("theme ( 47 , 2 ) AND dog ( 1 )",
     "dog(1) AND theme(47,2)",
     True),
    ("   theme (47 , 2 ) AND dog ( 1)",
     "dog    ( 1 ) AND    theme ( 47 ,2 )",
     True),
    ("theme ( 47 , 2 ) AND dog ( 1 )",
     "dog ( 1 ) AND theme ( 47 , 2 )",
     True),
    ("dog ( 1 ) AND happy ( 47 ) AND theme ( 47 , 2 )",
     "happy ( 47 ) AND dog ( 1 ) AND theme ( 47 , 2 )",
     True),
    ("dog ( 1 ) AND happy ( 47 )",
     "happy ( 47 ) AND dog ( 1 ) AND theme ( 47 , 2 )",
     False),
    ("dog ( 1 ) AND happy ( 47 ) AND theme ( 47 , 2 )",
     "dog ( 4 ) AND happy ( 30 ) AND theme ( 30 , 2 )",
     True),
    ("dog ( 2 ) AND happy ( 47 ) AND theme ( 47 , 2 )",
     "dog ( 4 ) AND happy ( 30 ) AND theme ( 30 , 2 )",
     False),
    ("dog ( 1 ) AND happy ( 1 ) AND theme ( 47 , 2 )",
     "dog ( 2 ) AND happy ( 30 ) AND theme ( 30 , 2 )",
     False),
     # Submitted by a student:
    (" boy ( 10 ) ; sandwich ( 36 ) ; Liam ( 42 ) ; like ( 21 ) AND agent ( 21 , 10 ) AND ccomp ( 21 , 0 ) AND respect ( 0 ) AND theme ( 0 , 36 ) AND agent ( 0 , 42 )",
     " boy ( 10 ) ; sandwich ( 36 ) ; Liam ( 42 ) ; like ( 21 ) AND agent ( 21 , 10 ) AND ccomp ( 21 , 0 ) AND respect ( 0 ) AND agent ( 0 , 42 ) AND theme ( 0 , 36 )",
     True),
    # Conjunct ordering:
    ("baby ( 5 ) ; Ava ( 9 ) ; eat ( 42 ) AND agent ( 42 , 5 ) AND theme ( 42 , 9 )",
     "agent ( 11 , 26 ) AND eat ( 11 ) AND Ava ( 32 ) AND theme ( 11 , 32 ) AND baby ( 26 )",
     True),
    ("baby ( 35 ) ; * pumpkin ( 11 ) ; believe ( 13 ) AND agent ( 13 , 35 ) AND ccomp ( 13 , 45 ) AND hold ( 45 ) AND theme ( 45 , 11 )",
     "believe ( 35 ) AND ccomp ( 35 , 2 ) AND hold ( 2 ) AND agent ( 35 , 21 ) AND * pumpkin ( 4 ) AND theme ( 2 , 4 ) AND baby ( 21 )",
     True),
    # Check that wrong predictions are called wrong:
    ('* puppy ( 11 ) ; need ( 53 ) AND agent ( 53 , 11 ) AND xcomp ( 53 , 45 ) AND crawl ( 45 ) AND agent ( 45 , 11 )',
     '* puppy ( 19 ) ; need ( 1 ) AND agent ( 1 , 19 ) AND xcomp ( 1 , 15 ) AND call ( 15 ) AND agent ( 15 , 19 )',
     False),
    # Fix of the above:
    ('* puppy ( 11 ) ; need ( 53 ) AND agent ( 53 , 11 ) AND xcomp ( 53 , 45 ) AND crawl ( 45 ) AND agent ( 45 , 11 )',
     '* puppy ( 19 ) ; need ( 1 ) AND agent ( 1 , 19 ) AND xcomp ( 1 , 15 ) AND crawl ( 15 ) AND agent ( 15 , 19 )',
     True),
    ('* shark ( 51 ) ; * cookie ( 32 ) ; * bag ( 3 ) ; nmod . in ( 32 , 3 ) AND eat ( 15 ) AND agent ( 15 , 51 ) AND theme ( 15 , 32 )',
     '* chalk ( 19 ) ; * cookie ( 50 ) ; * bag ( 13 ) ; nmod . in ( 50 , 13 ) AND eat ( 5 ) AND agent ( 5 , 19 ) AND theme ( 5 , 50 )',
     False),
    # Fix of the above
    ('* shark ( 51 ) ; * cookie ( 32 ) ; * bag ( 3 ) ; nmod . in ( 32 , 3 ) AND eat ( 15 ) AND agent ( 15 , 51 ) AND theme ( 15 , 32 )',
     '* shark ( 19 ) ; * cookie ( 50 ) ; * bag ( 13 ) ; nmod . in ( 50 , 13 ) AND eat ( 5 ) AND agent ( 5 , 19 ) AND theme ( 5 , 50 )',
     True),
    ('Mia ( 25 ) ; * raisin ( 7 ) ; * box ( 23 ) ; nmod . on ( 7 , 23 ) AND bake ( 32 ) AND agent ( 32 , 25 ) AND theme ( 32 , 7 )',
     'Mia ( 4 ) ; * raisin ( 40 ) ; * box ( 20 ) ; nmod . on ( 40 , 20 ) AND snap ( 17 ) AND agent ( 17 , 4 ) AND theme ( 17 , 40 )',
     False),
    ('Emma ( 53 ) ; Ella ( 43 ) ; Liam ( 47 ) ; Oliver ( 1 ) ; girl ( 51 ) ; Bella ( 10 ) ; Ava ( 11 ) ; hero ( 28 ) ; * cake ( 46 ) ; * truck ( 20 ) ; * crib ( 0 ) ; * room ( 25 ) ; nmod . on ( 46 , 20 ) AND nmod . in ( 20 , 0 ) AND nmod . in ( 0 , 25 ) AND like ( 54 ) AND agent ( 54 , 53 ) AND ccomp ( 54 , 35 ) AND like ( 35 ) AND agent ( 35 , 43 ) AND ccomp ( 35 , 9 ) AND respect ( 9 ) AND agent ( 9 , 47 ) AND ccomp ( 9 , 22 ) AND notice ( 22 ) AND agent ( 22 , 1 ) AND ccomp ( 22 , 16 ) AND imagine ( 16 ) AND agent ( 16 , 51 ) AND ccomp ( 16 , 37 ) AND mean ( 37 ) AND agent ( 37 , 10 ) AND ccomp ( 37 , 59 ) AND say ( 59 ) AND agent ( 59 , 11 ) AND ccomp ( 59 , 2 ) AND award ( 2 ) AND recipient ( 2 , 28 ) AND theme ( 2 , 46 )',
     'Emma ( 25 ) ; Ella ( 50 ) ; Liam ( 8 ) ; Oliver ( 9 ) ; girl ( 14 ) ; girl ( 6 ) ; Ava ( 45 ) ; Ava ( 9 ) ; hero ( 29 ) ; * cake ( 42 ) ; * truck ( 13 ) ; * room ( 56 ) ; * room ( 16 ) ; nmod . on ( 42 , 56 ) AND nmod . in ( 13 , 33 ) AND like ( 23 ) AND agent ( 23 , 25 ) AND ccomp ( 23 , 21 ) AND like ( 21 ) AND agent ( 21 , 50 ) AND ccomp ( 21 , 20 ) AND respect ( 20 ) AND agent ( 20 , 8 ) AND ccomp ( 20 , 50 ) AND imagine ( 50 ) AND agent ( 50 , 14 ) AND ccomp ( 50 , 50 ) AND mean ( 50 ) AND agent ( 50 , 6 ) AND ccomp ( 50 , 54 ) AND say ( 54 ) AND agent ( 54 , 45 ) AND ccomp ( 54 , 53 ) AND award ( 53 ) AND recipient ( 53 , 29 ) AND theme ( 53 , 42 ) AND award ( 22 ) AND agent ( 22 , 29 ) AND theme ( 22 , 42 )',
     False)
])
def test_recogs_exact_match(gold, predicted, expected):
    result = recogs_exact_match(gold, predicted)
    assert result == expected
