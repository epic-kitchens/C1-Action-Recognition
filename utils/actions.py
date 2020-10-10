from typing import Union

import numpy as np

_ACTION_VERB_MULTIPLIER = 1000


def action_id_from_verb_noun(verb: Union[int, np.array], noun: Union[int, np.array]):
    """
    Examples:
    >>> action_id_from_verb_noun(0, 0)
    0
    >>> action_id_from_verb_noun(0, 1)
    1
    >>> action_id_from_verb_noun(0, 351)
    351
    >>> action_id_from_verb_noun(1, 0)
    1000
    >>> action_id_from_verb_noun(1, 1)
    1001
    >>> action_id_from_verb_noun(np.array([0, 1, 2]), np.array([0, 1, 2]))
    array([   0, 1001, 2002])
    """
    return verb * _ACTION_VERB_MULTIPLIER + noun
