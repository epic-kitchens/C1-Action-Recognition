import argparse
import sys
from pathlib import Path
from textwrap import dedent

from utils.challenge import (
    make_action_recognition_submission,
    write_submission_file,
)
from utils.results import load_results

parser = argparse.ArgumentParser(
        description="Export model scores to JSON for submission to leaderboard.",
        formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
        "scores",
        type=Path,
        help=dedent("""\
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

        """),
)
parser.add_argument(
        "scores_json_zip",
        type=Path,
        help="Path to zip file to be created containing the JSON submission.",
)
parser.add_argument(
        "--sls-pt",
        choices=[0, 1, 2, 3, 4, 5],
        default=2,
        type=int,
        help=dedent("""\
        Level of pre-training supervision used by your model:
        
        0: No pretraining
        1: Pretrained on public image dataset
        2: Pretrained on public image and video datasets
        3: Pretrained using self-supervision on public data
        4: Pretrained using self-supervision on public data and on EPIC-Kitchens
        5: Pretrained on private data
        
        See https://github.com/epic-kitchens/sls for the canonical reference on SLS-PT.
        
        (default: 2)
        """),
)
parser.add_argument(
        "--sls-tl",
        choices=[0, 1, 2, 3, 4, 5],
        default=3,
        type=int,
        help=dedent("""\
        Level of training labels used by your model:
        
        0: No supervision 
        1: Weak-supervision [video-level] (Can use narration / verb_class / 
           noun_class, but no temporal or spatial annotations)
        2: Weak-supervision [temporal] (Can use narration_timestamp in addition to 
           supervision in 1)
        3: Full-supervision [temporal] (Can use start_frame / stop_frame in addition 
           to supervision in 1/2)
        4: Full-supervision [spatio-temporal] (Can use spatial-annotations produced 
           by pretrained detection/segmentation model in addition to supervision 
           defined in 1/2/3)
        5: Full-supervision [spatio-temporal+] (Can use prior knowledge outside of 
           labels specified in addition to supervision defined in 1/2/3/4)
            
        All additional labels used in training the model should be made available.
            
        See https://github.com/epic-kitchens/sls for the canonical reference on SLS-TL .
        
        (default: 3)"""),
)
parser.add_argument(
        "--sls-td",
        choices=[0, 1, 2, 3, 4, 5],
        default=4,
        type=int,
        help=dedent("""\
        Level of training supervision used by your model:
        
        0: Zero-shot learning (no training data used, only class knowledge incorporated)
        1: Few-shot learning (trained with up to 5 examples of each verb-class or 
           noun-class in the dataset)
        2: Efficient learning (A random sample of no more than 25%% of the training 
           data  was used to train the model and the sample was not optimised)
        3: Training set (The full training split was used to train the model)
        4: Train+val (The training and validation sets were used to train the model, 
           typically after optimising hyperparameters using the validation set.)
        5: Train+val+ (All labelled data in train+val was used in addition to other 
           labelled or unlabelled data from additional source [different from 
           pretraining])
        
        All additional data used in training the model should be made available.
        
        See https://github.com/epic-kitchens/sls for the canonical reference on SLS-TD 
        
        (default: 4)"""),
)
parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target zip file if it already exists.",
)


EXIT_CODE_OUTPUT_FILE_EXISTS = 1
EXIT_CODE_KEYS_NOT_PRESENT = 2
EXIT_CODE_DATA_SCHEMA_VIOLATED = 3

N_VERBS = 97
N_NOUNS = 300


def main(args):
    if args.scores_json_zip.exists() and not args.force:
        print(
                f"The output file {args.scores_json_zip} already exists, use --force to overwrite it."
        )
        sys.exit(EXIT_CODE_OUTPUT_FILE_EXISTS)

    scores_dict = load_results(args.scores_pt)

    if not verify_keys_are_present(scores_dict):
        sys.exit(EXIT_CODE_KEYS_NOT_PRESENT)

    verb_scores = scores_dict["verb_output"]
    noun_scores = scores_dict["noun_output"]
    narration_ids = scores_dict["narration_id"]

    if not validate_data_schema(noun_scores, verb_scores, narration_ids):
        sys.exit(EXIT_CODE_DATA_SCHEMA_VIOLATED)

    submission = make_action_recognition_submission(
            verb_scores,
            noun_scores,
            narration_ids,
            challenge="action_recognition",
            sls_pt=args.sls_pt,
            sls_tl=args.sls_tl,
            sls_td=args.sls_td,
    )

    write_submission_file(submission, args.scores_json_zip)


def verify_keys_are_present(scores_dict):
    validation_passed = True
    for key in ["verb_output", "noun_output", "narration_id"]:
        if key not in scores_dict:
            validation_passed = False
            print(f"Expected an entry for {key!r} but it was not present.")
    return validation_passed


def validate_data_schema(noun_scores, verb_scores, narration_ids):
    n_verb_scores = verb_scores.shape[0]
    n_noun_scores = noun_scores.shape[0]
    n_narration_ids = len(narration_ids)
    validation_passed = True
    if n_verb_scores != n_noun_scores:
        print(
                f"Expected no. of verb scores ({n_verb_scores}) to match no. of noun "
                f"scores ({n_noun_scores})."
        )
        validation_passed = False
    if n_verb_scores != n_narration_ids:
        print(
                f"Expected no. of scores ({n_verb_scores}) to match no. of narration IDs "
                f"({n_narration_ids})."
        )
        validation_passed = False
    if verb_scores.shape[1] != N_VERBS:
        print(
                f"Expected verb scores to have shape (N, {N_VERBS}) but was "
                f"{verb_scores.shape}."
        )
        validation_passed = False
    if noun_scores.shape[1] != N_NOUNS:
        print(
                f"Expected noun scores to have shape (N, {N_NOUNS}) but was "
                f"{noun_scores.shape}."
        )
        validation_passed = False
    return validation_passed


if __name__ == "__main__":
    main(parser.parse_args())
