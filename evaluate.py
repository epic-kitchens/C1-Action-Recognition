import argparse
from pathlib import Path
from textwrap import dedent

import pandas as pd
import numpy as np
import torch
import yaml

from utils.actions import action_id_from_verb_noun
from utils.metrics import compute_metrics
from utils.results import load_results
from utils.scoring import compute_action_scores

parser = argparse.ArgumentParser(
    description="Evaluate model results on the validation set",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "results",
    type=Path,
    help=dedent(
        """\
        Path to a results file with either a .pt suffix (loadable via `torch.load`) or 
        with a .pkl suffix (loadable via `pickle.load(open(path, 'rb'))`). The 
        loaded data should be in one of the following formats:

            [
                {
                    'verb_output': np.ndarray of float32, shape [97],
                    'noun_output': np.ndarray of float32, shape [300],
                    'narration_id': str, e.g. 'P01_101_1'
                }, ... # repeated entries 
            ]

        or 

            {
                'verb_output': np.ndarray of float32, shape [N, 97],
                'noun_output': np.ndarray of float32, shape [N, 300],
                'narration_id': np.ndarray of str, shape [N,]
            }
        """
    ),
)
parser.add_argument("labels", type=Path, help="Labels (pickled dataframe)")
parser.add_argument(
    "--tail-verb-classes-csv", type=Path,
        default="annotations/EPIC_100_tail_verbs.csv"
)
parser.add_argument(
    "--tail-noun-classes-csv", type=Path,
        default="annotations/EPIC_100_tail_nouns.csv"
)
parser.add_argument(
    "--unseen-participant-ids-csv",
    type=Path,
    default="annotaions/EPIC_100_unseen_participant_ids.csv",
)


def collate(results):
    return {k: [r[k] for r in results] for k in results[0].keys()}


def add_action_class_column(groundtruth_df):
    groundtruth_df["action_class"] = action_id_from_verb_noun(
        groundtruth_df["verb_class"], groundtruth_df["noun_class"]
    )
    return groundtruth_df


def main(args):
    labels: pd.DataFrame = pd.read_pickle(args.labels)
    if "narration_id" in labels.columns:
        labels.set_index("narration_id", inplace=True)
    labels = add_action_class_column(labels)
    unseen_participants: np.ndarray = pd.read_csv(
        args.unseen_participant_ids_csv, index_col="participant_id"
    ).index.values
    tail_verb_classes: np.ndarray = pd.read_csv(
        args.tail_verb_classes_csv, index_col="verb"
    ).index.values
    tail_noun_classes: np.ndarray = pd.read_csv(
        args.tail_noun_classes_csv, index_col="noun"
    ).index.values

    results = load_results(args.results)
    verb_output = results["verb_output"]
    noun_output = results["noun_output"]
    narration_ids = results["narration_id"]
    scores = {
        "verb": verb_output,
        "noun": noun_output,
    }
    (verbs, nouns), _scores = compute_action_scores(
        scores["verb"], scores["noun"], top_n=100
    )
    scores["action"] = [
        {
            action_id_from_verb_noun(verb, noun): score
            for verb, noun, score in zip(segment_verbs, segment_nouns, segment_score)
        }
        for segment_verbs, segment_nouns, segment_score in zip(verbs, nouns, _scores)
    ]
    accuracies = compute_metrics(
        labels.loc[narration_ids],
        scores,
        tail_verb_classes,
        tail_noun_classes,
        unseen_participants,
    )

    display_metrics = dict()
    for split in accuracies.keys():
        for task in ["verb", "noun", "action"]:
            task_accuracies = accuracies[split][task]
            for k, task_accuracy in zip((1, 5), task_accuracies):
                display_metrics[f"{split}_{task}_accuracy_at_{k}"] = task_accuracy
    display_metrics = {
        metric: float(value * 100) for metric, value in display_metrics.items()
    }

    print(yaml.dump(display_metrics))


if __name__ == "__main__":
    main(parser.parse_args())
