import warnings
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from utils.scoring import scores_dict_to_ranks


def _check_label_predictions_preconditions(rankings: np.ndarray, labels: np.ndarray):
    if len(rankings) < 1:
        raise ValueError(
            f"Need at least one instance to evaluate, but input shape "
            f"was {rankings.shape}"
        )
    if not rankings.ndim == 2:
        raise ValueError(f"Rankings should be a 2D matrix but was {rankings.ndim}D")
    if not labels.ndim == 1:
        raise ValueError(f"Labels should be a 1D vector but was {labels.ndim}D")
    if not labels.shape[0] == rankings.shape[0]:
        raise ValueError(
            f"Number of labels ({labels.shape[0]}) provided does not match number of "
            f"predictions ({rankings.shape[0]})"
        )


def topk_accuracy(
    rankings: np.ndarray, labels: np.ndarray, ks: Union[Tuple[int, ...], int] = (1, 5)
) -> List[float]:
    """Computes Top-K accuracies for different values of k

    Args:
        rankings: 2D rankings array: shape = (instance_count, label_count)
        labels: 1D correct labels array: shape = (instance_count,)
        ks: The k values in top-k, either an int or a list of ints.

    Returns:
        list of float: TOP-K accuracy for each k in ks

    Raises:
        ValueError
             If the dimensionality of the rankings or labels is incorrect, or
             if the length of rankings and labels aren't equal
    """
    if isinstance(ks, int):
        ks = (ks,)
    _check_label_predictions_preconditions(rankings, labels)

    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    accuracies = [tp[:, :k].max(1).mean() for k in ks]
    if any(np.isnan(accuracies)):
        raise ValueError(f"NaN present in accuracies {accuracies}")
    return accuracies


def compute_metrics(
    groundtruth_df: pd.DataFrame,
    scores: Dict[str, np.ndarray],
    tail_verbs: Sequence[int],
    tail_nouns: Sequence[int],
    unseen_participant_ids: Sequence[str],
):
    """
    Args:
        groundtruth_df: DataFrame containing 'verb_class': int, 'noun_class': int and 'action_class': Tuple[int, int] columns.
        scores: Dictionary containing three entries: 'verb', 'noun' and 'action' entries should map to a 2D
            np.ndarray of shape (instance_count, class_count) where each element is the predicted score
            of that class.
        tail_verbs: The set of verb classes that are considered to be tail classes
        tail_nouns: The set of noun classes that are considered to be tail classes
        unseen_participant_ids: The set of participant IDs who do not have videos in the training set.

    Returns:
        A dictionary containing nested metrics.

    Raises:
        ValueError
            If the shapes of the score arrays are not correct, or the lengths of the groundtruth_df and the
            scores array are not equal, or if the grountruth_df doesn't have the specified columns.
    """
    for entry in "verb", "noun", "action":
        class_col = entry + "_class"
        if class_col not in groundtruth_df.columns:
            raise ValueError("Expected '{}' column in groundtruth_df".format(class_col))

    ranks = scores_dict_to_ranks(scores)
    top_k = (1, 5)

    all_accuracies = {
        "verb": topk_accuracy(
            ranks["verb"], groundtruth_df["verb_class"].values, ks=top_k
        ),
        "noun": topk_accuracy(
            ranks["noun"], groundtruth_df["noun_class"].values, ks=top_k
        ),
        "action": topk_accuracy(
            ranks["action"], groundtruth_df["action_class"].values, ks=top_k
        ),
    }
    unseen_bool_idx = groundtruth_df.participant_id.isin(unseen_participant_ids)
    unseen_groundtruth_df = groundtruth_df[unseen_bool_idx]
    if unseen_bool_idx.sum() == 0:
        warnings.warn("There are no unseen segments to evaluate")
        unseen_accuracies = None
    else:
        unseen_ranks = {
            task: task_scores[unseen_bool_idx] for task, task_scores in ranks.items()
        }
        unseen_accuracies = {
            "verb": topk_accuracy(
                unseen_ranks["verb"],
                unseen_groundtruth_df["verb_class"].values,
                ks=(1,),
            ),
            "noun": topk_accuracy(
                unseen_ranks["noun"],
                unseen_groundtruth_df["noun_class"].values,
                ks=(1,),
            ),
            "action": topk_accuracy(
                unseen_ranks["action"],
                unseen_groundtruth_df["action_class"].values,
                ks=(1,),
            ),
        }

    tail_verb_bool_idx = groundtruth_df.verb_class.isin(tail_verbs)
    tail_noun_bool_idx = groundtruth_df.noun_class.isin(tail_nouns)
    tail_action_bool_idx = tail_verb_bool_idx | tail_noun_bool_idx

    tail_verb_groundtruth_df = groundtruth_df[tail_verb_bool_idx]
    tail_noun_groundtruth_df = groundtruth_df[tail_noun_bool_idx]
    tail_action_groundtruth_df = groundtruth_df[tail_action_bool_idx]

    tail_verb_ranks = ranks["verb"][tail_verb_bool_idx.values]
    tail_noun_ranks = ranks["noun"][tail_noun_bool_idx.values]
    tail_action_ranks = ranks["action"][tail_action_bool_idx.values]

    tail_accuracies = {
        "verb": topk_accuracy(
            tail_verb_ranks, tail_verb_groundtruth_df["verb_class"].values, ks=(1,)
        ),
        "noun": topk_accuracy(
            tail_noun_ranks, tail_noun_groundtruth_df["noun_class"].values, ks=(1,)
        ),
        "action": topk_accuracy(
            tail_action_ranks,
            tail_action_groundtruth_df["action_class"].values,
            ks=(1,),
        ),
    }

    accuracies = {
        "all": all_accuracies,
        "tail": tail_accuracies,
    }
    if unseen_accuracies is not None:
        accuracies["unseen"] = unseen_accuracies

    return accuracies


def compute_accuracies(
    groundtruth_df: pd.DataFrame,
    ranks: Dict[str, np.ndarray],
    top_k: Union[int, Tuple[int, ...]] = (1, 5),
) -> Dict[str, Union[float, Union[float, List[float]]]]:
    """
    Compute class aware metrics dictionary

    Args:
        groundtruth_df: DataFrame containing 'verb_class': int, 'noun_class': int and 'action_class': Tuple[int, int] columns.
        ranks: Dictionary containing three entries: 'verb', 'noun' and 'action' entries should map to a 2D
            np.ndarray of shape (instance_count, class_count) where each element is the predicted rank of that class.
            The 'action' rank array should be
        top_k: The set of k values to compute top-k accuracy for.
    """
    verb_accuracies = topk_accuracy(
        ranks["verb"], groundtruth_df["verb_class"].values, ks=top_k
    )
    noun_accuracies = topk_accuracy(
        ranks["noun"], groundtruth_df["noun_class"].values, ks=top_k
    )
    action_accuracies = topk_accuracy(
        ranks["action"], groundtruth_df["action_class"].values
    )
    return {
        "verb": verb_accuracies,
        "noun": noun_accuracies,
        "action": action_accuracies,
    }
