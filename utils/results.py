from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def load_results(results_path: Path) -> Dict[str, Any]:
    """
    Load results created by running `test.py`

    Args:
        results_path: Path to PT results file

    Returns:
        A dictionary with the structure

        .. code-block::

            {
               'verb_output': np.ndarray [N, C_v],
               'noun_output': np.ndarray [N, C_n],
               'narration_id': np.ndarray [N,],
            }


    """
    # These are all in python lists, we turn them into np arrays for convenience
    # We first have to collate them.
    results: List[Dict[str, Any]] = torch.load(
        results_path, map_location=torch.device("cpu")
    )
    new_results = dict()
    first_item = results[0]
    for key in first_item.keys():
        new_results[key] = np.array([r[key] for r in results])
    return new_results
