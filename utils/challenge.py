import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def make_action_recognition_submission(
    verb_scores: np.ndarray,
    noun_scores: np.ndarray,
    narration_ids: Union[np.ndarray, List[str]],
    challenge: str = "action_recognition",
    sls_pt: int = 5,
    sls_tl: int = 5,
    sls_td: int = 5,
):
    """
    Args:
        verb_scores: Array containing verb scores of shape :math:`(N, C_v)`.
        noun_scores: Array containing noun scores of shape :math:(N, C_n)`.
        narration_ids:  Array or list of length :math:`N` containing narration IDs
            for each score.
        challenge: The challenge being submitted to.
        sls_pt:  Supervision level: pretraining (0--5)
        sls_tl:  Supervision level: training labels (0--5)
        sls_td:  Supervision level: training data (0--5)

    Returns:
        Submission dictionary ready to be serialised to JSON.
    """
    return {
        "version": "0.2",
        "challenge": challenge,
        "sls_pt": sls_pt,
        "sls_tl": sls_tl,
        "sls_td": sls_td,
        "results": make_action_recognition_submission_scores_dict(
            verb_scores, noun_scores, narration_ids
        ),
    }


def make_action_recognition_submission_scores_dict(
    verb_scores: np.ndarray,
    noun_scores: np.ndarray,
    narration_ids: Union[np.ndarray, List[str]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Args:
        verb_scores: Array containing verb scores of shape :math:`(N, C_v)`.
        noun_scores: Array containing noun scores of shape :math:(N, C_n)`.
        narration_ids:  Array or list of length :math:`N` containing narration IDs
            for each score.

    Returns:
        Dictionary mapping narration ids to a dictionary containing verb and noun
        scores, e.g.

        .. code-block:: python

            "P01_101_0": {
              "verb": {
                "0": 1.223,
                "1": 4.278,
                ...
                "96": 0.023
              },
              "noun": {
                "0": 0.804,
                "1": 1.870,
                ...
                "299": 0.023
              }
            }

    """
    results = dict()
    for example_verb_scores, example_noun_scores, narration_id in zip(
        verb_scores, noun_scores, narration_ids
    ):
        results[str(narration_id)] = {
            "verb": make_scores_dict(example_verb_scores),
            "noun": make_scores_dict(example_noun_scores),
        }
    return results


def make_scores_dict(scores: np.ndarray) -> Dict[str, float]:
    assert scores.ndim == 1
    return {str(i): float(s) for i, s in enumerate(scores)}


def write_submission_file(submission_dict: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED, compresslevel=5) as f:
        f.writestr("test.json", json.dumps(submission_dict))


def read_pickle(path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
