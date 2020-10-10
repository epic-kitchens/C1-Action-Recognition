from typing import Dict, List, Union

import numpy as np
from scipy.special import softmax


def scores_dict_to_ranks(scores_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: scores_to_ranks(scores) for key, scores in scores_dict.items()}


def scores_to_ranks(scores: Union[np.ndarray, List[Dict[int, float]]]) -> np.ndarray:
    if isinstance(scores, np.ndarray):
        return _scores_array_to_ranks(scores)
    elif isinstance(scores, list):
        return _scores_dict_to_ranks(scores)
    raise ValueError("Cannot compute ranks for type {}".format(type(scores)))


def top_scores(scores: np.ndarray, top_n: int = 100):
    """
    Examples:
        >>> top_scores(np.array([0.2, 0.6, 0.1, 0.04, 0.06]), top_n=3)
        (array([1, 0, 2]), array([0.6, 0.2, 0.1]))
    """
    if scores.ndim == 1:
        top_n_idx = scores.argsort()[::-1][:top_n]
        return top_n_idx, scores[top_n_idx]
    else:
        top_n_scores_idx = np.argsort(scores)[:, ::-1][:, :top_n]
        top_n_scores = scores[
            np.arange(0, len(scores)).reshape(-1, 1), top_n_scores_idx
        ]
        return top_n_scores_idx, top_n_scores


def _scores_array_to_ranks(scores: np.ndarray):
    """
    The rank vector contains classes and is indexed by the rank

    Examples:
        >>> _scores_array_to_ranks(np.array([[0.1, 0.15, 0.25,  0.3, 0.5], \
                                             [0.5, 0.3, 0.25,  0.15, 0.1], \
                                             [0.2, 0.4,  0.1,  0.25, 0.05]]))
        array([[4, 3, 2, 1, 0],
               [0, 1, 2, 3, 4],
               [1, 3, 0, 2, 4]])
    """
    if scores.ndim != 2:
        raise ValueError(
            "Expected scores to be 2 dimensional: [n_instances, n_classes]"
        )
    return scores.argsort(axis=-1)[:, ::-1]


def _scores_dict_to_ranks(scores: List[Dict[int, float]]) -> np.ndarray:
    """
    Compute ranking from class to score dictionary

    Examples:
        >>> _scores_dict_to_ranks([{0: 0.15, 10: 0.75, 5: 0.1},\
                                   {0: 0.85, 10: 0.10, 5: 0.05}])
        array([[10,  0,  5],
               [ 0, 10,  5]])
    """
    ranks = []
    for score in scores:
        class_ids = np.array(list(score.keys()))
        score_array = np.array([score[class_id] for class_id in class_ids])
        ranks.append(class_ids[np.argsort(score_array)[::-1]])
    return np.array(ranks)


def compute_action_scores(verb_scores: np.ndarray, noun_scores: np.ndarray, top_n=100):
    top_verbs, top_verb_scores = top_scores(verb_scores, top_n=top_n)
    top_nouns, top_noun_scores = top_scores(noun_scores, top_n=top_n)
    top_verb_probs = softmax(top_verb_scores)
    top_noun_probs = softmax(top_noun_scores)
    action_probs_matrix = (
        top_verb_probs[:, :, np.newaxis] * top_noun_probs[:, np.newaxis, :]
    )
    instance_count = action_probs_matrix.shape[0]
    action_ranks = action_probs_matrix.reshape(instance_count, -1).argsort(axis=-1)[
        :, ::-1
    ]
    verb_ranks_idx, noun_ranks_idx = np.unravel_index(
        action_ranks[:, :top_n], shape=(action_probs_matrix.shape[1:])
    )

    segments = np.arange(0, instance_count).reshape(-1, 1)
    return (
        (top_verbs[segments, verb_ranks_idx], top_nouns[segments, noun_ranks_idx]),
        action_probs_matrix.reshape(instance_count, -1)[
            segments, action_ranks[:, :top_n]
        ],
    )
