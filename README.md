# EPIC-KITCHENS Action Recognition baselines

Train/Val/Test splits and annotations are available at [Annotations Repo](https://github.com/epic-kitchens/epic-kitchens-100-annotations)

To participate and submit to this challenge, register at [Action Recognition Codalab Challenge](https://competitions.codalab.org/competitions/25923)

## Released Models

- For TBN see https://github.com/ekazakos/temporal-binding-network
- For SlowFast see https://github.com/epic-kitchens/SlowFast (to be released)
- For TSN/TRN/TSM see (coming soon)

## Result data formats

We support two formats for model results.

- *List format*:
  ```
  [
      {
          'verb_output': Iterable of float, shape [97],
          'noun_output': Iterable of float, shape [300],
          'narration_id': str, e.g. 'P01_101_1'
      }, ... # repeated for all segments in the val/test set.
  ]
  ```
- *Dict format*:
  ```
  {
      'verb_output': np.ndarray of float32, shape [N, 97],
      'noun_output': np.ndarray of float32, shape [N, 300],
      'narration_id': np.ndarray of str, shape [N,]
  }
  ```

Either of these formats can saved via `torch.save` with `.pt` suffix or with
`pickle.dump` with a `.pkl` suffix.

We provide two example files (from a TSN trained on RGB) following both these formats for reference.

- Dict format saved as `.pkl`: https://www.dropbox.com/s/5bfn8p0lw5quy68/dict_format.pkl?dl=1
- List format saved as `.pt`: https://www.dropbox.com/s/4ouop6n267l3ip9/list_format.pt?dl=1

Note that either of these layouts can be stored in a `.pkl`/`.pt` file--the dict
format doesn't necessarily have to be in a `.pkl`.


## Evaluating model results

We provide an evaluation script to compute the metrics we report in the paper on
the validation set. You will also need to clone the [annotations repo](https://github.com/epic-kitchens/epic-kitchens-100-annotations)

```bash
$ python evaluate.py \
    results.pt \
    /path/to/annotations/EPIC_100_validation.pkl \
    --tail-verb-classes-csv /path/to/annotations/EPIC_100_tail_verbs.csv \
    --tail-noun-classes-csv /path/to/annotations/EPIC_100_tail_nouns.csv \
    --unseen-participant-ids /path/to/annotations/EPIC_100_unseen_participant_ids_test.csv \
```


## Creating competition submission files

Once you have evaluated your model on the test set, you can simply run this
script to generate a zip file containing the JSON representation of your model's
scores that can be submitted to the 
[challenge leaderboard](https://competitions.codalab.org/competitions/25923).

```bash
$ python create_submission.py test_results.pt test_results.zip
```
